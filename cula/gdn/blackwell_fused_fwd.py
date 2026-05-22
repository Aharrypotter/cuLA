# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cuLA GDN chunked prefill API for Blackwell.

The public entry point mirrors FlashInfer's packed-varlen GDN prefill contract,
while keeping the integration surface local to cuLA. The SM100 kernel is a CuTe
DSL port of FlashInfer's Blackwell chunked prefill implementation.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch


__all__ = [
    "chunk_gated_delta_rule",
    "is_sm100_gdn_prefill_available",
]

_SUPPORTED_IO_DTYPES = (torch.bfloat16,)
_SUPPORTED_STATE_DTYPES = (torch.float32,)
_HEAD_SIZE = 128
_CHUNK_SIZE = 64
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"


@dataclass(frozen=True)
class _GDNPrefillInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    output: torch.Tensor
    cu_seqlens_i32: torch.Tensor
    initial_state: Optional[torch.Tensor]
    output_state: Optional[torch.Tensor]
    scale: float
    num_seqs: int
    num_o_heads: int


@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    num_q_heads: int,
    num_v_heads: int,
    is_gqa: bool,
    use_initial_state: bool,
    store_final_state: bool,
):
    return {}


@functools.cache
def _get_num_sm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _cuda_major_version() -> int:
    if torch.version.cuda is None:
        return 0
    return int(torch.version.cuda.split(".")[0])


def is_sm100_gdn_prefill_available(device: torch.device | int | str | None = None) -> bool:
    """Return whether the current environment can run the SM100 GDN prefill path."""
    if _cuda_major_version() < 13 or not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major == 10 and props.minor in (0, 3)


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    import cutlass

    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    if torch_dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(f"Unsupported GDN IO dtype {torch_dtype}")


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    import cutlass

    if torch_dtype == torch.float32:
        return cutlass.Float32
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    raise ValueError(f"Unsupported GDN state dtype {torch_dtype}")


def _check_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    ndim: int,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {tensor.ndim}D")
    if isinstance(dtype, tuple):
        if tensor.dtype not in dtype:
            expected = ", ".join(str(d) for d in dtype)
            raise ValueError(f"{name} dtype must be one of ({expected}), got {tensor.dtype}")
    elif tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_checkpoint_args(
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> None:
    if checkpoint_every_n_tokens < 0:
        raise ValueError(f"checkpoint_every_n_tokens must be non-negative, got {checkpoint_every_n_tokens}")
    if checkpoint_every_n_tokens > 0 and checkpoint_every_n_tokens % _CHUNK_SIZE != 0:
        raise ValueError(
            f"checkpoint_every_n_tokens must be a multiple of {_CHUNK_SIZE}, "
            f"got {checkpoint_every_n_tokens}"
        )
    if checkpoint_every_n_tokens != 0 or state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise NotImplementedError("GDN prefill checkpointing is deferred until after the no-checkpoint SM100 path passes")


def _validate_grouping(num_q_heads: int, num_k_heads: int, num_v_heads: int) -> tuple[bool, int]:
    if num_q_heads >= num_v_heads:
        if num_k_heads != num_v_heads:
            raise ValueError(
                "GQA requires num_k_heads == num_v_heads when num_q_heads >= num_v_heads; "
                f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
            )
        if num_k_heads == 0 or num_q_heads % num_k_heads != 0:
            raise ValueError(
                "GQA requires num_q_heads to be divisible by num_k_heads; "
                f"got q={num_q_heads}, k={num_k_heads}"
            )
        return True, num_q_heads

    if num_k_heads != num_q_heads:
        raise ValueError(
            "GVA requires num_k_heads == num_q_heads when num_v_heads > num_q_heads; "
            f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
        )
    if num_q_heads == 0 or num_v_heads % num_q_heads != 0:
        raise ValueError(
            "GVA requires num_v_heads to be divisible by num_q_heads; "
            f"got q={num_q_heads}, v={num_v_heads}"
        )
    return False, num_v_heads


def _prepare_gdn_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    use_qk_l2norm_in_kernel: bool,
) -> _GDNPrefillInputs:
    device = q.device if isinstance(q, torch.Tensor) else torch.device("cuda")

    _check_cuda_tensor("q", q, device=device, ndim=3, dtype=_SUPPORTED_IO_DTYPES)
    _check_cuda_tensor("k", k, device=device, ndim=3, dtype=q.dtype)
    _check_cuda_tensor("v", v, device=device, ndim=3, dtype=q.dtype)

    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("GDN prefill does not support use_qk_l2norm_in_kernel in the SM100-first port")
    _validate_checkpoint_args(state_checkpoints, checkpoint_cu_starts, checkpoint_every_n_tokens)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for GDN packed varlen prefill")

    total_tokens, num_q_heads, head_size = q.shape
    if total_tokens <= 0:
        raise ValueError("q, k, and v must contain at least one token")
    if num_q_heads <= 0 or k.shape[1] <= 0 or v.shape[1] <= 0:
        raise ValueError(
            "q, k, and v must have at least one head; "
            f"got q={num_q_heads}, k={k.shape[1]}, v={v.shape[1]}"
        )
    if k.shape[0] != total_tokens or v.shape[0] != total_tokens:
        raise ValueError(
            "q, k, and v must have the same total token dimension; "
            f"got q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )
    if k.shape[2] != head_size or v.shape[2] != head_size:
        raise ValueError(
            "q, k, and v must have the same head_size; "
            f"got q={head_size}, k={k.shape[2]}, v={v.shape[2]}"
        )
    if head_size != _HEAD_SIZE:
        raise ValueError(f"SM100 GDN prefill currently requires head_size={_HEAD_SIZE}, got {head_size}")

    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    _, num_o_heads = _validate_grouping(num_q_heads, num_k_heads, num_v_heads)

    _check_cuda_tensor("cu_seqlens", cu_seqlens, device=device, ndim=1, dtype=(torch.int32, torch.int64))
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have at least two elements")
    num_seqs = cu_seqlens.numel() - 1
    cu_seqlens_i32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    if not cu_seqlens_i32.is_contiguous():
        cu_seqlens_i32 = cu_seqlens_i32.contiguous()

    gate_shape = (total_tokens, num_o_heads)
    if g is None:
        g = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _check_cuda_tensor("g", g, device=device, ndim=2, dtype=torch.float32)
        if tuple(g.shape) != gate_shape:
            raise ValueError(f"g must have shape {gate_shape}, got {tuple(g.shape)}")

    if beta is None:
        beta = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _check_cuda_tensor("beta", beta, device=device, ndim=2, dtype=torch.float32)
        if tuple(beta.shape) != gate_shape:
            raise ValueError(f"beta must have shape {gate_shape}, got {tuple(beta.shape)}")

    output_shape = (total_tokens, num_o_heads, head_size)
    if output is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=device)
    else:
        _check_cuda_tensor("output", output, device=device, ndim=3, dtype=q.dtype)
        if tuple(output.shape) != output_shape:
            raise ValueError(f"output must have shape {output_shape}, got {tuple(output.shape)}")

    state_shape = (num_seqs, num_o_heads, head_size, head_size)
    if initial_state is not None:
        _check_cuda_tensor("initial_state", initial_state, device=device, ndim=4, dtype=_SUPPORTED_STATE_DTYPES)
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}")

    if output_state is not None and not output_final_state:
        raise ValueError("output_state requires output_final_state=True")
    if output_final_state and output_state is None:
        output_state = torch.empty(state_shape, dtype=torch.float32, device=device)
    elif output_state is not None:
        _check_cuda_tensor("output_state", output_state, device=device, ndim=4, dtype=_SUPPORTED_STATE_DTYPES)
        if tuple(output_state.shape) != state_shape:
            raise ValueError(f"output_state must have shape {state_shape}, got {tuple(output_state.shape)}")

    scale_value = 1.0 / math.sqrt(head_size) if scale is None or scale == 0.0 else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")

    return _GDNPrefillInputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output=output,
        cu_seqlens_i32=cu_seqlens_i32,
        initial_state=initial_state,
        output_state=output_state,
        scale=scale_value,
        num_seqs=num_seqs,
        num_o_heads=num_o_heads,
    )


def _launch_sm100_gdn_prefill(inputs: _GDNPrefillInputs) -> None:
    if _cuda_major_version() < 13:
        raise NotImplementedError(
            "Blackwell GDN prefill requires CUDA 13+ and nvidia-cutlass-dsl[cu13]. "
            f"Current torch CUDA version is {torch.version.cuda!r}."
        )

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from cula.ops.gdn_chunked_prefill import GatedDeltaNetChunkedKernel
    from cula.utils import assert_blackwell

    assert_blackwell(inputs.q.device)

    q = inputs.q
    k = inputs.k
    v = inputs.v
    gate = inputs.g
    beta = inputs.beta
    output = inputs.output
    cu_seqlens = inputs.cu_seqlens_i32
    initial_state = inputs.initial_state
    output_state = inputs.output_state

    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    is_gqa = num_q_heads >= num_v_heads
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    state_dtype = torch.float32
    if initial_state is not None:
        state_dtype = initial_state.dtype
    elif output_state is not None:
        state_dtype = output_state.dtype

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_dtype),
        num_q_heads,
        num_v_heads,
        is_gqa,
        use_initial_state,
        store_final_state,
    )
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    num_sm = _get_num_sm(device_index)

    if "compiled" not in cache:
        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=_cutlass_io_dtype(q.dtype),
            acc_dtype=cutlass.Float32,
            state_dtype=_cutlass_state_dtype(state_dtype),
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64),
            mma_tiler_kv=(128, 128, 64),
            max_active_clusters=num_sm,
            num_sm=num_sm,
            is_GQA=is_gqa,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            enable_checkpoints=False,
            is_persistent=True,
        )

        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        output_cute = from_dlpack(output, assumed_align=16)
        output_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        initial_state_cute = None
        if use_initial_state:
            initial_state_cute = from_dlpack(initial_state, assumed_align=16)
            initial_state_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=head_size
            )

        output_state_cute = None
        if store_final_state:
            output_state_cute = from_dlpack(output_state, assumed_align=16)
            output_state_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=head_size
            )

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, inputs.num_seqs, num_q_heads, num_v_heads, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)
        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        cache["compiled"] = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            gate_cute,
            beta_cute,
            output_cute,
            cu_seqlens_cute,
            initial_state_cute,
            output_state_cute,
            None,
            None,
            0,
            inputs.scale,
            workspace_cute,
            stream,
            options=_COMPILE_OPTIONS,
        )
        cache["num_sm"] = num_sm

    compiled = cache["compiled"]
    num_sm = cache["num_sm"]
    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, inputs.num_seqs, num_q_heads, num_v_heads, True
    )
    workspace_key = f"workspace_{device_index}"
    if workspace_key not in cache or cache[workspace_key].numel() < workspace_size:
        cache[workspace_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[workspace_key]

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        None,
        None,
        0,
        inputs.scale,
        workspace,
        stream,
    )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run packed-varlen GDN chunked prefill on Blackwell.

    Args:
        q: Query tensor with shape ``[total_tokens, num_q_heads, 128]``.
        k: Key tensor with shape ``[total_tokens, num_k_heads, 128]``.
        v: Value tensor with shape ``[total_tokens, num_v_heads, 128]``.
        g: Optional forget gate with shape ``[total_tokens, max(num_q_heads, num_v_heads)]``.
        beta: Optional update gate with shape ``[total_tokens, max(num_q_heads, num_v_heads)]``.
        scale: Optional output scale. ``None`` and ``0.0`` both mean ``1 / sqrt(128)``.
        initial_state: Optional initial state in ``[num_seqs, num_o_heads, V, K]`` layout.
        output_final_state: Whether to return final recurrent state.
        cu_seqlens: Cumulative sequence lengths. Public API accepts int32 or int64;
            the SM100 kernel receives int32.
        output: Optional preallocated output.
        output_state: Optional preallocated final state. Requires ``output_final_state=True``.

    Returns:
        ``output`` if ``output_final_state`` is false, otherwise ``(output, output_state)``.
    """
    inputs = _prepare_gdn_prefill_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        output=output,
        output_state=output_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    _launch_sm100_gdn_prefill(inputs)
    if output_final_state:
        assert inputs.output_state is not None
        return inputs.output, inputs.output_state
    return inputs.output

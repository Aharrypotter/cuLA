# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host orchestration for the private N4 Candidate M product skeleton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .config import (
    CHUNK_SIZE,
    HEAD_SIZE,
    RECURRENT_BACKEND_ID,
    VALUE_SIZE,
)
from .output_state import OutputAndState
from .recurrent import RecurrentOutputAndState
from .solve import UnitLowerSolve
from .vector import VectorTransform
from .wgmma import EPILOGUE_ERASE, EPILOGUE_QK, AsymmetricWgmma


@dataclass(frozen=True)
class FixedMHAIntermediates:
    """Consumer-visible global publications for N4 correctness auditing."""

    cumulative_log: torch.Tensor
    q_bar: torch.Tensor
    k_bar: torch.Tensor
    e_bar: torch.Tensor
    pseudo_value: torch.Tensor
    erase_lower: torch.Tensor
    causal_qk_scaled: torch.Tensor
    solved_u: torch.Tensor
    erase_raw_bf16: torch.Tensor
    qk_raw_bf16: torch.Tensor


@dataclass(frozen=True)
class RecurrentExecutionInfo:
    """Private N5 allocation and state-store receipt."""

    backend_id: str
    chunk_lengths: tuple[int, ...]
    state_buffer_count: int
    state_store_count: int
    final_state_materialized: bool
    disabled_last_chunk_sink_alias: bool


_compiled_vector: dict[tuple[int, int], object] = {}
_compiled_wgmma: dict[int, object] = {}
_compiled_solve: dict[int, object] = {}
_compiled_output_state: dict[tuple[int, int], object] = {}
_compiled_recurrent: dict[tuple[int, int], object] = {}


def _device_key(device: torch.device) -> int:
    if device.index is None:
        return torch.cuda.current_device()
    return device.index


def _current_stream(device: torch.device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _dynamic_tokens(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tensor.dim_order(),
    )


def _compile_vector(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    cumulative_log: torch.Tensor,
    q_bar: torch.Tensor,
    k_bar: torch.Tensor,
    e_bar: torch.Tensor,
    pseudo_value: torch.Tensor,
    stream: cuda.CUstream,
):
    key = (_device_key(q.device), q.shape[1])
    compiled = _compiled_vector.get(key)
    if compiled is None:
        compiled = cute.compile(
            VectorTransform(),
            *(_dynamic_tokens(tensor) for tensor in (q, k, v, g, b, w)),
            *(
                from_dlpack(tensor, assumed_align=16)
                for tensor in (
                    cumulative_log,
                    q_bar,
                    k_bar,
                    e_bar,
                    pseudo_value,
                )
            ),
            cutlass.Int32(q.shape[0]),
            cutlass.Int32(q.shape[1]),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_vector[key] = compiled
    return compiled


def _compile_wgmma(
    operand_a: torch.Tensor,
    operand_b: torch.Tensor,
    raw_matrix: torch.Tensor,
    epilogue_matrix: torch.Tensor,
    stream: cuda.CUstream,
):
    key = _device_key(operand_a.device)
    compiled = _compiled_wgmma.get(key)
    if compiled is None:
        compiled = cute.compile(
            AsymmetricWgmma(),
            from_dlpack(operand_a, assumed_align=16),
            from_dlpack(operand_b, assumed_align=16),
            from_dlpack(raw_matrix, assumed_align=16),
            from_dlpack(epilogue_matrix, assumed_align=16),
            cutlass.Int32(CHUNK_SIZE),
            cutlass.Int32(EPILOGUE_ERASE),
            cutlass.Float32(1.0),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_wgmma[key] = compiled
    return compiled


def _compile_solve(
    lower: torch.Tensor,
    pseudo_value: torch.Tensor,
    solved_u: torch.Tensor,
    stream: cuda.CUstream,
):
    key = _device_key(lower.device)
    compiled = _compiled_solve.get(key)
    if compiled is None:
        compiled = cute.compile(
            UnitLowerSolve(),
            from_dlpack(lower, assumed_align=16),
            from_dlpack(pseudo_value, assumed_align=16),
            from_dlpack(solved_u, assumed_align=16),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_solve[key] = compiled
    return compiled


def _compile_output_state(
    causal_qk_scaled: torch.Tensor,
    solved_u: torch.Tensor,
    k_bar: torch.Tensor,
    cumulative_log: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    valid_tokens: int,
    stream: cuda.CUstream,
):
    key = (_device_key(output.device), output.shape[1])
    compiled = _compiled_output_state.get(key)
    if compiled is None:
        compiled = cute.compile(
            OutputAndState(),
            *(
                from_dlpack(tensor, assumed_align=16)
                for tensor in (
                    causal_qk_scaled,
                    solved_u,
                    k_bar,
                    cumulative_log,
                    output,
                    final_state,
                )
            ),
            cutlass.Int32(valid_tokens),
            cutlass.Int32(output.shape[1]),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_output_state[key] = compiled
    return compiled


def _compile_recurrent(
    q_bar: torch.Tensor,
    causal_qk_scaled: torch.Tensor,
    solved_u: torch.Tensor,
    solved_y: torch.Tensor,
    k_bar: torch.Tensor,
    cumulative_log: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    valid_tokens: int,
    scale: float,
    initial_state_is_zero: bool,
    store_final_state: bool,
    stream: cuda.CUstream,
):
    key = (_device_key(output.device), output.shape[1])
    compiled = _compiled_recurrent.get(key)
    if compiled is None:
        compiled = cute.compile(
            RecurrentOutputAndState(),
            *(
                from_dlpack(tensor, assumed_align=16)
                for tensor in (
                    q_bar,
                    causal_qk_scaled,
                    solved_u,
                    solved_y,
                    k_bar,
                    cumulative_log,
                    initial_state,
                    output,
                    final_state,
                )
            ),
            cutlass.Int32(valid_tokens),
            cutlass.Int32(output.shape[1]),
            cutlass.Float32(scale),
            cutlass.Int32(int(initial_state_is_zero)),
            cutlass.Int32(int(store_final_state)),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_recurrent[key] = compiled
    return compiled


def _check_input(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")


def _tiled_operand(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(CHUNK_SIZE, HEAD_SIZE // 16, 16).permute(0, 2, 1)


def run_fixed_mha_single_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: float | None = None,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, FixedMHAIntermediates]:
    """Run the private N4 fixed-length, zero-state MHA backend.

    This is deliberately not exported through :mod:`cula.gdn2`. It accepts one
    sequence with ``1 <= T <= 64`` and returns BF16 output plus FP32 public
    ``[1,H,V,K]`` terminal state.
    """

    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must be a rank-3 [T,H,128] tensor")
    tokens, heads, width = q.shape
    if not 1 <= tokens <= CHUNK_SIZE:
        raise ValueError(f"T must be in [1,{CHUNK_SIZE}], got {tokens}")
    if heads <= 0 or width != HEAD_SIZE:
        raise ValueError(f"q must have shape [T,H,{HEAD_SIZE}] with H>0")
    device = q.device
    shape = (tokens, heads, HEAD_SIZE)
    for name, tensor, dtype in (
        ("q", q, torch.bfloat16),
        ("k", k, torch.bfloat16),
        ("v", v, torch.bfloat16),
        ("g", g, torch.float32),
        ("b", b, torch.bfloat16),
        ("w", w, torch.bfloat16),
    ):
        _check_input(name, tensor, shape=shape, dtype=dtype, device=device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError(
            f"GDN2 N4 requires compute capability 9.0, got {props.major}.{props.minor}",
        )
    scale_value = HEAD_SIZE**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")

    with torch.cuda.device(device):
        cumulative_log = torch.empty(
            (heads, CHUNK_SIZE, HEAD_SIZE),
            dtype=torch.float32,
            device=device,
        )
        transformed = [
            torch.empty(
                (heads, CHUNK_SIZE, HEAD_SIZE),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(4)
        ]
        q_bar, k_bar, e_bar, pseudo_value = transformed
        erase_raw = torch.empty(
            (heads, CHUNK_SIZE, CHUNK_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        qk_raw = torch.empty_like(erase_raw)
        erase_lower = torch.empty(
            (heads, CHUNK_SIZE, CHUNK_SIZE),
            dtype=torch.float32,
            device=device,
        )
        causal_qk_scaled = torch.empty_like(erase_lower)
        solved_u = torch.empty(
            (heads, CHUNK_SIZE, VALUE_SIZE),
            dtype=torch.float32,
            device=device,
        )
        output_padded = torch.empty(
            (CHUNK_SIZE, heads, VALUE_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        final_state = torch.empty(
            (1, heads, VALUE_SIZE, HEAD_SIZE),
            dtype=torch.float32,
            device=device,
        )
        stream = _current_stream(device)

        vector_kernel = _compile_vector(
            q,
            k,
            v,
            g,
            b,
            w,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            stream,
        )
        vector_kernel(
            q,
            k,
            v,
            g,
            b,
            w,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            tokens,
            heads,
            stream,
        )

        for head in range(heads):
            k_operand = _tiled_operand(k_bar[head])
            erase_operand = _tiled_operand(e_bar[head])
            q_operand = _tiled_operand(q_bar[head])
            wgmma_kernel = _compile_wgmma(
                erase_operand,
                k_operand,
                erase_raw[head],
                erase_lower[head],
                stream,
            )
            wgmma_kernel(
                erase_operand,
                k_operand,
                erase_raw[head],
                erase_lower[head],
                tokens,
                EPILOGUE_ERASE,
                1.0,
                stream,
            )
            solve_kernel = _compile_solve(
                erase_lower[head],
                pseudo_value[head],
                solved_u[head],
                stream,
            )
            solve_kernel(
                erase_lower[head],
                pseudo_value[head],
                solved_u[head],
                stream,
            )
            wgmma_kernel(
                q_operand,
                k_operand,
                qk_raw[head],
                causal_qk_scaled[head],
                tokens,
                EPILOGUE_QK,
                scale_value,
                stream,
            )

        output_state_kernel = _compile_output_state(
            causal_qk_scaled,
            solved_u,
            k_bar,
            cumulative_log,
            output_padded,
            final_state,
            tokens,
            stream,
        )
        output_state_kernel(
            causal_qk_scaled,
            solved_u,
            k_bar,
            cumulative_log,
            output_padded,
            final_state,
            tokens,
            heads,
            stream,
        )

    output = output_padded[:tokens]
    if not return_intermediates:
        return output, final_state
    intermediates = FixedMHAIntermediates(
        cumulative_log=cumulative_log,
        q_bar=q_bar,
        k_bar=k_bar,
        e_bar=e_bar,
        pseudo_value=pseudo_value,
        erase_lower=erase_lower,
        causal_qk_scaled=causal_qk_scaled,
        solved_u=solved_u,
        erase_raw_bf16=erase_raw,
        qk_raw_bf16=qk_raw,
    )
    return output, final_state, intermediates


def run_fixed_mha_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    return_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, RecurrentExecutionInfo]:
    """Run the private N5 fixed-length MHA recurrent backend.

    The caller state uses public ``[1,H,V,K]`` layout and remains read-only.
    Output-only final chunks execute no state store and allocate no dedicated
    final-state buffer; algorithmically necessary inter-chunk carry remains
    explicit in the debug receipt.
    """

    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must be a rank-3 [T,H,128] tensor")
    tokens, heads, width = q.shape
    if not 1 <= tokens <= 4096:
        raise ValueError(f"T must be in [1,4096], got {tokens}")
    if heads <= 0 or width != HEAD_SIZE:
        raise ValueError(f"q must have shape [T,H,{HEAD_SIZE}] with H>0")
    device = q.device
    shape = (tokens, heads, HEAD_SIZE)
    for name, tensor, dtype in (
        ("q", q, torch.bfloat16),
        ("k", k, torch.bfloat16),
        ("v", v, torch.bfloat16),
        ("g", g, torch.float32),
        ("b", b, torch.bfloat16),
        ("w", w, torch.bfloat16),
    ):
        _check_input(name, tensor, shape=shape, dtype=dtype, device=device)
    state_shape = (1, heads, VALUE_SIZE, HEAD_SIZE)
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor or None")
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(
                f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}",
            )
        if (
            initial_state.device != device
            or not initial_state.is_cuda
            or initial_state.dtype != torch.float32
            or not initial_state.is_contiguous()
        ):
            raise ValueError(
                "initial_state must be contiguous FP32 CUDA public [1,H,V,K]",
            )
        if initial_state.data_ptr() % 16:
            raise ValueError("initial_state data pointer must be 16-byte aligned")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        raise RuntimeError(
            f"GDN2 N5 requires compute capability 9.0, got {props.major}.{props.minor}",
        )
    scale_value = HEAD_SIZE**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")

    chunk_count = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    required_state_stores = chunk_count if output_final_state else chunk_count - 1
    state_buffer_count = min(2, required_state_stores)
    if initial_state is None and state_buffer_count == 0:
        state_buffer_count = 1

    with torch.cuda.device(device):
        padded_tokens = chunk_count * CHUNK_SIZE
        output_padded = torch.empty(
            (padded_tokens, heads, VALUE_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        cumulative_log = torch.empty(
            (heads, CHUNK_SIZE, HEAD_SIZE),
            dtype=torch.float32,
            device=device,
        )
        transformed = [
            torch.empty(
                (heads, CHUNK_SIZE, HEAD_SIZE),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(4)
        ]
        q_bar, k_bar, e_bar, pseudo_value = transformed
        erase_raw = torch.empty(
            (heads, CHUNK_SIZE, CHUNK_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        qk_raw = torch.empty_like(erase_raw)
        erase_lower = torch.empty(
            (heads, CHUNK_SIZE, CHUNK_SIZE),
            dtype=torch.float32,
            device=device,
        )
        causal_qk_scaled = torch.empty_like(erase_lower)
        solved_u = torch.empty(
            (heads, CHUNK_SIZE, VALUE_SIZE),
            dtype=torch.float32,
            device=device,
        )
        solved_y = torch.empty(
            (heads, CHUNK_SIZE, HEAD_SIZE),
            dtype=torch.float32,
            device=device,
        )
        state_buffers = [
            torch.empty(
                state_shape,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(state_buffer_count)
        ]
        stream = _current_stream(device)
        current_state = initial_state
        chunk_lengths: list[int] = []
        state_store_count = 0
        disabled_last_chunk_sink_alias = False

        for chunk_index in range(chunk_count):
            chunk_start = chunk_index * CHUNK_SIZE
            valid_tokens = min(CHUNK_SIZE, tokens - chunk_start)
            chunk_lengths.append(valid_tokens)
            chunk_end = chunk_start + valid_tokens
            q_chunk = q[chunk_start:chunk_end]
            k_chunk = k[chunk_start:chunk_end]
            v_chunk = v[chunk_start:chunk_end]
            g_chunk = g[chunk_start:chunk_end]
            b_chunk = b[chunk_start:chunk_end]
            w_chunk = w[chunk_start:chunk_end]

            vector_kernel = _compile_vector(
                q_chunk,
                k_chunk,
                v_chunk,
                g_chunk,
                b_chunk,
                w_chunk,
                cumulative_log,
                q_bar,
                k_bar,
                e_bar,
                pseudo_value,
                stream,
            )
            vector_kernel(
                q_chunk,
                k_chunk,
                v_chunk,
                g_chunk,
                b_chunk,
                w_chunk,
                cumulative_log,
                q_bar,
                k_bar,
                e_bar,
                pseudo_value,
                valid_tokens,
                heads,
                stream,
            )

            for head in range(heads):
                k_operand = _tiled_operand(k_bar[head])
                erase_operand = _tiled_operand(e_bar[head])
                q_operand = _tiled_operand(q_bar[head])
                wgmma_kernel = _compile_wgmma(
                    erase_operand,
                    k_operand,
                    erase_raw[head],
                    erase_lower[head],
                    stream,
                )
                wgmma_kernel(
                    erase_operand,
                    k_operand,
                    erase_raw[head],
                    erase_lower[head],
                    valid_tokens,
                    EPILOGUE_ERASE,
                    1.0,
                    stream,
                )
                solve_kernel = _compile_solve(
                    erase_lower[head],
                    pseudo_value[head],
                    solved_u[head],
                    stream,
                )
                solve_kernel(
                    erase_lower[head],
                    pseudo_value[head],
                    solved_u[head],
                    stream,
                )
                solve_kernel(
                    erase_lower[head],
                    e_bar[head],
                    solved_y[head],
                    stream,
                )
                wgmma_kernel(
                    q_operand,
                    k_operand,
                    qk_raw[head],
                    causal_qk_scaled[head],
                    valid_tokens,
                    EPILOGUE_QK,
                    scale_value,
                    stream,
                )

            initial_state_is_zero = current_state is None
            if current_state is None:
                state_in = state_buffers[0]
            else:
                state_in = current_state
            store_state = chunk_index < chunk_count - 1 or output_final_state
            if store_state:
                if initial_state_is_zero:
                    state_out = state_buffers[0]
                else:
                    state_out = next(buffer for buffer in state_buffers if buffer.data_ptr() != state_in.data_ptr())
            else:
                state_out = state_in
                disabled_last_chunk_sink_alias = state_out.data_ptr() == state_in.data_ptr()

            output_chunk = output_padded[chunk_start : chunk_start + CHUNK_SIZE]
            recurrent_kernel = _compile_recurrent(
                q_bar,
                causal_qk_scaled,
                solved_u,
                solved_y,
                k_bar,
                cumulative_log,
                state_in,
                output_chunk,
                state_out,
                valid_tokens,
                scale_value,
                initial_state_is_zero,
                store_state,
                stream,
            )
            recurrent_kernel(
                q_bar,
                causal_qk_scaled,
                solved_u,
                solved_y,
                k_bar,
                cumulative_log,
                state_in,
                output_chunk,
                state_out,
                valid_tokens,
                heads,
                scale_value,
                int(initial_state_is_zero),
                int(store_state),
                stream,
            )
            if store_state:
                current_state = state_out
                state_store_count += 1

    output = output_padded[:tokens]
    final_state = current_state if output_final_state else None
    if not return_debug:
        return output, final_state
    debug = RecurrentExecutionInfo(
        backend_id=RECURRENT_BACKEND_ID,
        chunk_lengths=tuple(chunk_lengths),
        state_buffer_count=state_buffer_count,
        state_store_count=state_store_count,
        final_state_materialized=output_final_state,
        disabled_last_chunk_sink_alias=disabled_last_chunk_sink_alias,
    )
    return output, final_state, debug

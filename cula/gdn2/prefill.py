# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public packed-varlen Gated DeltaNet-2 prefill API for Hopper SM90."""

from __future__ import annotations

import functools
import importlib.metadata
import math
from dataclasses import dataclass

import torch

from cula.ops.gdn2.sm90.config import (
    EXPECTED_CUTLASS_DSL_VERSION,
    HEAD_SIZE,
    I4_R23_BACKEND_ID,
    I4_R24_BACKEND_ID,
    I4_R24R1_BACKEND_ID,
    I4_R24R2_BACKEND_ID,
    I4_R30R1_BACKEND_ID,
    I4_R31_BACKEND_ID,
    I4_R35_BACKEND_ID,
    PUBLIC_BACKEND_ID,
    R36_IDENTITY_BACKEND_ID,
    R36_LPT32_BACKEND_ID,
    VALUE_SIZE,
    selected_public_backend_id,
)

__all__ = [
    "chunk_gdn2",
    "get_sm90_gdn2_backend",
    "get_sm90_gdn2_backend_identity",
    "is_sm90_gdn2_available",
]

_INT32_MAX = 2**31 - 1
_ALIGNMENT = 16


@dataclass(frozen=True)
class _GDN2Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    b: torch.Tensor
    w: torch.Tensor
    output: torch.Tensor
    initial_state: torch.Tensor | None
    output_state: torch.Tensor | None
    cu_seqlens: torch.Tensor
    total_tokens: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    output_final_state: bool
    scale: float


@functools.cache
def _installed_cutlass_dsl_version() -> str | None:
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return None


def get_sm90_gdn2_backend() -> str:
    """Return the only GDN2 v1 backend."""

    return "dsl"


def get_sm90_gdn2_backend_identity() -> str:
    """Return the exact source-owned public backend identity."""

    return selected_public_backend_id()


def is_sm90_gdn2_available(
    device: torch.device | int | str | None = None,
) -> bool:
    """Return whether the frozen SM90a GDN2 backend is available."""

    if device is not None and not isinstance(device, int):
        device = torch.device(device)
        if device.type != "cuda":
            return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (9, 0):
        return False
    return _installed_cutlass_dsl_version() == EXPECTED_CUTLASS_DSL_VERSION


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    ndim: int,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must be rank {ndim}, got shape {tuple(tensor.shape)}",
        )
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % _ALIGNMENT:
        raise ValueError(
            f"{name} data pointer must be {_ALIGNMENT}-byte aligned",
        )


def _storage_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _storage_interval(left)
    right_start, right_end = _storage_interval(right)
    return left_start < right_end and right_start < left_end


def _reject_writable_overlap(
    name: str,
    writable: torch.Tensor,
    read_only: dict[str, torch.Tensor],
) -> None:
    for other_name, other in read_only.items():
        if _overlaps(writable, other):
            raise ValueError(
                f"{name} must not overlap read-only tensor {other_name}",
            )


def _validate_device_contents(
    cu_seqlens: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    total_tokens: int,
) -> None:
    """Synchronously validate value preconditions for diagnostics only."""

    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if offsets[0] != 0:
        raise ValueError(f"cu_seqlens[0] must be 0, got {offsets[0]}")
    if offsets[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens[-1] must equal total_tokens={total_tokens}, got {offsets[-1]}",
        )
    if offsets[-1] > _INT32_MAX:
        raise ValueError(
            f"packed token count must not exceed {_INT32_MAX}",
        )
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError(
            "zero-length or decreasing sequences are unsupported",
        )
    if not bool(torch.isfinite(g).all()) or not bool((g <= 0).all()):
        raise ValueError("g must contain finite non-positive log decays")
    for name, gate in (("b", b), ("w", w)):
        if not bool(torch.isfinite(gate).all()) or not bool((gate >= 0).all()) or not bool((gate <= 1).all()):
            raise ValueError(
                f"{name} must contain finite values in [0,1]",
            )


def _prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    scale: float | None,
    output: torch.Tensor | None,
    output_state: torch.Tensor | None,
    validate_inputs: bool,
) -> _GDN2Inputs:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    device = q.device
    _check_tensor(
        "q",
        q,
        device=device,
        ndim=3,
        dtype=torch.bfloat16,
    )
    for name, tensor, dtype in (
        ("k", k, torch.bfloat16),
        ("v", v, torch.bfloat16),
        ("g", g, torch.float32),
        ("b", b, torch.bfloat16),
        ("w", w, torch.bfloat16),
    ):
        _check_tensor(
            name,
            tensor,
            device=device,
            ndim=3,
            dtype=dtype,
        )
    _check_tensor(
        "cu_seqlens",
        cu_seqlens,
        device=device,
        ndim=1,
        dtype=torch.int64,
    )

    total_tokens, num_q_heads, key_size = q.shape
    if not 1 <= total_tokens <= _INT32_MAX:
        raise ValueError(
            f"total_tokens must be in [1,{_INT32_MAX}]",
        )
    if key_size != HEAD_SIZE:
        raise ValueError(f"q key dimension must be {HEAD_SIZE}")
    expected_q_shape = (total_tokens, num_q_heads, HEAD_SIZE)
    for name, tensor in (("k", k), ("g", g), ("b", b)):
        if tuple(tensor.shape) != expected_q_shape:
            raise ValueError(
                f"{name} must have shape {expected_q_shape}, got {tuple(tensor.shape)}",
            )
    num_v_heads = v.shape[1]
    expected_v_shape = (total_tokens, num_v_heads, VALUE_SIZE)
    if tuple(v.shape) != expected_v_shape or tuple(w.shape) != expected_v_shape:
        raise ValueError(
            f"v and w must have shape {expected_v_shape}",
        )
    if num_q_heads <= 0 or num_v_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_v_heads < num_q_heads:
        raise NotImplementedError(
            "GQA is outside the GDN2 v1 contract",
        )
    if num_v_heads % num_q_heads:
        raise ValueError(
            "GVA requires Hv to be an integer multiple of Hq",
        )
    if cu_seqlens.numel() < 2:
        raise ValueError(
            "cu_seqlens must contain at least [0,total_tokens]",
        )
    num_sequences = cu_seqlens.numel() - 1

    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be a bool")
    if not isinstance(validate_inputs, bool):
        raise TypeError("validate_inputs must be a bool")
    if validate_inputs:
        _validate_device_contents(
            cu_seqlens,
            g,
            b,
            w,
            total_tokens=total_tokens,
        )

    state_shape = (
        num_sequences,
        num_v_heads,
        VALUE_SIZE,
        HEAD_SIZE,
    )
    if initial_state is not None:
        _check_tensor(
            "initial_state",
            initial_state,
            device=device,
            ndim=4,
            dtype=torch.float32,
        )
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(
                f"initial_state must have public [N,Hv,V,K] shape {state_shape}, got {tuple(initial_state.shape)}",
            )

    output_shape = (total_tokens, num_v_heads, VALUE_SIZE)
    with torch.cuda.device(device):
        if output is None:
            output = torch.empty(
                output_shape,
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            _check_tensor(
                "output",
                output,
                device=device,
                ndim=3,
                dtype=torch.bfloat16,
            )
            if tuple(output.shape) != output_shape:
                raise ValueError(
                    f"output must have shape {output_shape}, got {tuple(output.shape)}",
                )

        if output_state is not None and not output_final_state:
            raise ValueError(
                "output_state requires output_final_state=True",
            )
        if output_final_state and output_state is None:
            output_state = torch.empty(
                state_shape,
                dtype=torch.float32,
                device=device,
            )
        elif output_state is not None:
            _check_tensor(
                "output_state",
                output_state,
                device=device,
                ndim=4,
                dtype=torch.float32,
            )
            if tuple(output_state.shape) != state_shape:
                raise ValueError(
                    f"output_state must have public [N,Hv,V,K] shape {state_shape}, got {tuple(output_state.shape)}",
                )

    read_only = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": b,
        "w": w,
        "cu_seqlens": cu_seqlens,
    }
    if initial_state is not None:
        read_only["initial_state"] = initial_state
    _reject_writable_overlap("output", output, read_only)
    if output_state is not None:
        _reject_writable_overlap("output_state", output_state, read_only)
        if _overlaps(output, output_state):
            raise ValueError(
                "output and output_state must not overlap",
            )

    scale_value = HEAD_SIZE**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    return _GDN2Inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        output=output,
        initial_state=initial_state,
        output_state=output_state,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        num_sequences=num_sequences,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        output_final_state=output_final_state,
        scale=scale_value,
    )


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    validate_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run packed MHA/GVA GDN2 forward prefill on Hopper SM90.

    The default path validates tensor metadata only. CUDA-resident offset and
    gate values are caller preconditions unless ``validate_inputs=True`` is
    explicitly selected; that diagnostic mode synchronizes.
    """

    inputs = _prepare_inputs(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        output=output,
        output_state=output_state,
        validate_inputs=validate_inputs,
    )
    properties = torch.cuda.get_device_properties(inputs.q.device)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(
            f"GDN2 SM90 requires compute capability 9.0, got {properties.major}.{properties.minor}",
        )
    installed = _installed_cutlass_dsl_version()
    if installed != EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            f"GDN2 SM90 requires nvidia-cutlass-dsl=={EXPECTED_CUTLASS_DSL_VERSION}, found {installed}",
        )

    backend_id = selected_public_backend_id()
    if backend_id == I4_R23_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r23_exact_s3,
        )

        launch_i4_r23_exact_s3(inputs)
    elif backend_id == I4_R24_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r24_exact_s3,
        )

        launch_i4_r24_exact_s3(inputs)
    elif backend_id == I4_R24R1_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r24r1_exact_s3,
        )

        launch_i4_r24r1_exact_s3(inputs)
    elif backend_id == I4_R24R2_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r24r2_exact_s3,
        )

        launch_i4_r24r2_exact_s3(inputs)
    elif backend_id == I4_R30R1_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r30r1_exact_s3,
        )

        launch_i4_r30r1_exact_s3(inputs)
    elif backend_id == I4_R31_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r23_public import (
            launch_i4_r31_exact_s3,
        )

        launch_i4_r31_exact_s3(inputs)
    elif backend_id == I4_R35_BACKEND_ID:
        from cula.ops.gdn2.sm90.i4_r35_public import (
            launch_i4_r35_exact5,
        )

        launch_i4_r35_exact5(inputs)
    elif backend_id in (
        R36_IDENTITY_BACKEND_ID,
        R36_LPT32_BACKEND_ID,
    ):
        from cula.ops.gdn2.sm90.r36_bakeoff_public import (
            launch_r36_identity,
            launch_r36_lpt32,
        )

        if backend_id == R36_LPT32_BACKEND_ID:
            launch_r36_lpt32(inputs)
        else:
            launch_r36_identity(inputs)
    else:
        assert backend_id == PUBLIC_BACKEND_ID
        from cula.ops.gdn2.sm90.public import launch_sm90_gdn2

        launch_sm90_gdn2(inputs)
    if output_final_state:
        assert inputs.output_state is not None
        return inputs.output, inputs.output_state
    return inputs.output

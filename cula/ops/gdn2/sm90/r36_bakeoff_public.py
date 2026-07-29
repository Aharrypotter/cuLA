# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Development-only dynamic-shape public carrier for R36 LPT32."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .config import R36_IDENTITY_BACKEND_ID, R36_LPT32_BACKEND_ID
from .k1_i4_raw_g_prefix_state import (
    A2K1I4RawGFactorReadyLockstepOnlyState,
    A2K1I4RawGStableLpt32State,
)

if TYPE_CHECKING:
    from cula.gdn2.prefill import _GDN2Inputs

_MAX_SEQUENCES = 32
_SUPPORTED_Q_HEADS = 16
_SUPPORTED_V_HEADS = (16, 32, 64)
_compiled: dict[tuple[int, int, bool, bool, str], object] = {}


@dataclass(frozen=True)
class R36Lpt32ExecutionInfo:
    """Metadata receipt for one R36 LPT32 development launch."""

    backend_id: str
    total_tokens: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    has_initial_state: bool
    store_final_state: bool
    sequence_policy: str
    compile_cache_entries: int
    fallback: bool


def _device_key(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _dynamic_mode0(tensor: torch.Tensor):
    return from_dlpack(
        tensor,
        assumed_align=16,
    ).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tensor.dim_order(),
    )


def _resolve_support(inputs: _GDN2Inputs) -> None:
    if inputs.num_q_heads != _SUPPORTED_Q_HEADS:
        raise NotImplementedError(
            "R36 LPT32 requires Hq=16",
        )
    if inputs.num_v_heads not in _SUPPORTED_V_HEADS:
        raise NotImplementedError(
            "R36 LPT32 requires Hv in {16,32,64}",
        )
    if not 1 <= inputs.num_sequences <= _MAX_SEQUENCES:
        raise NotImplementedError(
            "R36 LPT32 requires 1 <= N <= 32",
        )


def _compile(
    inputs: _GDN2Inputs,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    stream: cuda.CUstream,
    *,
    schedule: str,
):
    has_initial_state = inputs.initial_state is not None
    store_final_state = inputs.output_final_state
    key = (
        _device_key(inputs.q.device),
        inputs.num_v_heads,
        has_initial_state,
        store_final_state,
        schedule,
    )
    compiled = _compiled.get(key)
    if compiled is not None:
        return compiled

    kernel_type = A2K1I4RawGStableLpt32State if schedule == "stable_lpt32" else A2K1I4RawGFactorReadyLockstepOnlyState
    kernel = kernel_type(
        has_initial_state=has_initial_state,
        store_final_state=store_final_state,
    )
    compiled = cute.compile(
        kernel,
        *(
            _dynamic_mode0(tensor)
            for tensor in (
                inputs.q,
                inputs.k,
                inputs.v,
                inputs.b,
                inputs.w,
                inputs.cu_seqlens,
                inputs.g,
                inputs.q,
                inputs.q,
                initial_state,
                inputs.output,
                final_state,
            )
        ),
        cutlass.Int32(inputs.num_sequences),
        cutlass.Int32(inputs.num_q_heads),
        cutlass.Int32(inputs.num_v_heads),
        cutlass.Int32(inputs.total_tokens),
        cutlass.Float32(inputs.scale),
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled


def _launch(
    inputs: _GDN2Inputs,
    *,
    schedule: str,
    return_debug: bool = False,
) -> R36Lpt32ExecutionInfo | None:
    """Launch one dynamic-shape R36 development schedule."""

    if schedule not in {"identity", "stable_lpt32"}:
        raise ValueError(f"unknown R36 schedule: {schedule}")
    _resolve_support(inputs)
    initial_state = inputs.initial_state if inputs.initial_state is not None else inputs.q
    final_state = inputs.output_state if inputs.output_state is not None else inputs.output

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            initial_state,
            final_state,
            stream,
            schedule=schedule,
        )
        compiled(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.b,
            inputs.w,
            inputs.cu_seqlens,
            inputs.g,
            inputs.q,
            inputs.q,
            initial_state,
            inputs.output,
            final_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return R36Lpt32ExecutionInfo(
        backend_id=(R36_LPT32_BACKEND_ID if schedule == "stable_lpt32" else R36_IDENTITY_BACKEND_ID),
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        has_initial_state=inputs.initial_state is not None,
        store_final_state=inputs.output_final_state,
        sequence_policy=schedule,
        compile_cache_entries=len(_compiled),
        fallback=False,
    )


def launch_r36_lpt32(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> R36Lpt32ExecutionInfo | None:
    """Launch the dynamic-shape R36 LPT32 development backend."""

    return _launch(
        inputs,
        schedule="stable_lpt32",
        return_debug=return_debug,
    )


def launch_r36_identity(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> R36Lpt32ExecutionInfo | None:
    """Launch the dynamic-shape R36 identity control backend."""

    return _launch(
        inputs,
        schedule="identity",
        return_debug=return_debug,
    )

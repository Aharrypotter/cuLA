# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed S1-S5 public dispatch for the I4-R35 matrix candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .config import I4_R35_BACKEND_ID
from .k1_i4_raw_g_prefix_state import (
    A2K1I4RawGSequenceWaveRotate12State,
)

if TYPE_CHECKING:
    from cula.gdn2.prefill import _GDN2Inputs

_ROW_CONTRACTS = {
    (64, 1, 16, 16, False, False): ("S1-MHA-T64-H16-H0NONE-HTOFF"),
    (1024, 1, 16, 16, True, True): ("S2-MHA-T1024-H16-H0-HTON"),
    (4096, 20, 16, 16, False, True): ("S3-MHA-PACKED-T4096-H16-H0NONE-HTON"),
    (1024, 1, 16, 64, True, True): ("S4-GVA4-T1024-H16-H64-H0-HTON"),
    (1024, 1, 16, 32, False, True): ("S5-GVA2-T1024-H16-H32-H0NONE-HTON"),
}
_compiled: dict[
    tuple[int, int, int, int, int, bool, bool],
    object,
] = {}


@dataclass(frozen=True)
class I4R35ExecutionInfo:
    """Metadata receipt for one exact-matrix I4-R35 public launch."""

    backend_id: str
    row_id: str
    total_tokens: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    has_initial_state: bool
    store_final_state: bool
    sequence_policy: str
    public_raw_g_consumed: bool
    global_g_cumsum_workspace: bool
    separate_prefix_launch: bool
    fallback: bool


def _device_key(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _contract_key(inputs: _GDN2Inputs) -> tuple[int, int, int, int, bool, bool]:
    return (
        inputs.total_tokens,
        inputs.num_sequences,
        inputs.num_q_heads,
        inputs.num_v_heads,
        inputs.initial_state is not None,
        inputs.output_final_state,
    )


def _resolve_row(inputs: _GDN2Inputs) -> str:
    key = _contract_key(inputs)
    try:
        return _ROW_CONTRACTS[key]
    except KeyError as exc:
        raise NotImplementedError(
            f"I4-R35 experimental dispatch is frozen to the exact S1-S5 matrix; observed contract key={key!r}",
        ) from exc


def _compile(
    inputs: _GDN2Inputs,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    stream: cuda.CUstream,
):
    has_initial_state = inputs.initial_state is not None
    store_final_state = inputs.output_final_state
    key = (
        _device_key(inputs.q.device),
        inputs.total_tokens,
        inputs.num_sequences,
        inputs.num_q_heads,
        inputs.num_v_heads,
        has_initial_state,
        store_final_state,
    )
    compiled = _compiled.get(key)
    if compiled is not None:
        return compiled

    kernel = A2K1I4RawGSequenceWaveRotate12State(
        has_initial_state=has_initial_state,
        store_final_state=store_final_state,
    )
    compiled = cute.compile(
        kernel,
        *(
            from_dlpack(tensor, assumed_align=16)
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


def launch_i4_r35_exact5(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R35ExecutionInfo | None:
    """Launch R35 only for one of the five frozen campaign rows."""

    row_id = _resolve_row(inputs)
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
    return I4R35ExecutionInfo(
        backend_id=I4_R35_BACKEND_ID,
        row_id=row_id,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        has_initial_state=inputs.initial_state is not None,
        store_final_state=inputs.output_final_state,
        sequence_policy=("rotate12" if inputs.num_sequences > 12 else "r31_identity"),
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )

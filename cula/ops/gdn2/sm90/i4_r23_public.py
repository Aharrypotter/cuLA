# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit exact-S3 public dispatch for the I4-r23 architecture gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .config import (
    I4_R23_BACKEND_ID,
    I4_R24_BACKEND_ID,
    I4_R24R1_BACKEND_ID,
    I4_R24R2_BACKEND_ID,
    I4_R30R1_BACKEND_ID,
    I4_R31_BACKEND_ID,
)
from .k1_i4_raw_g_prefix_state import (
    A2K1I4RawGFactorReadyLockstepOnlyState,
    A2K1I4RawGFullReadyMbarrierOnlyState,
    A2K1I4RawGPrefixState,
    A2K1I4RawGSubgroupExclusiveCarryState,
    A2K1I4RawGSubgroupPrefixState,
)

if TYPE_CHECKING:
    from cula.gdn2.prefill import _GDN2Inputs

_S3_TOTAL_TOKENS = 4096
_S3_NUM_SEQUENCES = 20
_S3_NUM_HEADS = 16
_compiled: dict[
    tuple[int, int, int, int, bool, bool, bool, bool],
    object,
] = {}


@dataclass(frozen=True)
class I4R23ExecutionInfo:
    """Metadata receipt for the explicit exact-S3 I4-r23 route."""

    backend_id: str
    total_tokens: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    public_raw_g_consumed: bool
    global_g_cumsum_workspace: bool
    separate_prefix_launch: bool
    fallback: bool


def _device_key(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _compile(
    inputs: _GDN2Inputs,
    stream: cuda.CUstream,
    *,
    subgroup_prefix: bool = False,
    subgroup_exclusive_carry: bool = False,
    full_ready_mbarrier_only: bool = False,
    factor_ready_lockstep_only: bool = False,
):
    key = (
        _device_key(inputs.q.device),
        inputs.total_tokens,
        inputs.num_sequences,
        inputs.num_q_heads,
        subgroup_prefix,
        subgroup_exclusive_carry,
        full_ready_mbarrier_only,
        factor_ready_lockstep_only,
    )
    compiled = _compiled.get(key)
    if compiled is not None:
        return compiled

    if factor_ready_lockstep_only:
        kernel_type = A2K1I4RawGFactorReadyLockstepOnlyState
    elif full_ready_mbarrier_only:
        kernel_type = A2K1I4RawGFullReadyMbarrierOnlyState
    elif subgroup_exclusive_carry:
        kernel_type = A2K1I4RawGSubgroupExclusiveCarryState
    elif subgroup_prefix:
        kernel_type = A2K1I4RawGSubgroupPrefixState
    else:
        kernel_type = A2K1I4RawGPrefixState
    kernel = kernel_type(
        has_initial_state=False,
        store_final_state=True,
    )
    assert inputs.output_state is not None
    unused_bf16 = inputs.q
    unused_initial_state = inputs.q
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
                unused_bf16,
                unused_bf16,
                unused_initial_state,
                inputs.output,
                inputs.output_state,
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


def launch_i4_r23_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch R23 only for its frozen exact-S3 public contract."""

    if (
        inputs.total_tokens != _S3_TOTAL_TOKENS
        or inputs.num_sequences != _S3_NUM_SEQUENCES
        or inputs.num_q_heads != _S3_NUM_HEADS
        or inputs.num_v_heads != _S3_NUM_HEADS
        or inputs.initial_state is not None
        or not inputs.output_final_state
        or inputs.output_state is None
    ):
        raise NotImplementedError(
            "I4-r23 experimental dispatch is frozen to exact S3: "
            "T=4096, N=20, Hq=Hv=16, no initial state, "
            "output_final_state=True",
        )

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(inputs, stream)
        unused_bf16 = inputs.q
        unused_initial_state = inputs.q
        compiled(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.b,
            inputs.w,
            inputs.cu_seqlens,
            inputs.g,
            unused_bf16,
            unused_bf16,
            unused_initial_state,
            inputs.output,
            inputs.output_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R23_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )


def launch_i4_r24_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch the R24 subgroup-prefix child on exact S3."""

    if (
        inputs.total_tokens != _S3_TOTAL_TOKENS
        or inputs.num_sequences != _S3_NUM_SEQUENCES
        or inputs.num_q_heads != _S3_NUM_HEADS
        or inputs.num_v_heads != _S3_NUM_HEADS
        or inputs.initial_state is not None
        or not inputs.output_final_state
        or inputs.output_state is None
    ):
        raise NotImplementedError(
            "I4-r24 experimental dispatch is frozen to exact S3: "
            "T=4096, N=20, Hq=Hv=16, no initial state, "
            "output_final_state=True",
        )

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            stream,
            subgroup_prefix=True,
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
            inputs.q,
            inputs.output,
            inputs.output_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R24_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )


def launch_i4_r24r1_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch the corrected R24R1 subgroup-prefix child on exact S3."""

    result = launch_i4_r24_exact_s3(
        inputs,
        return_debug=return_debug,
    )
    if result is None:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R24R1_BACKEND_ID,
        total_tokens=result.total_tokens,
        num_sequences=result.num_sequences,
        num_q_heads=result.num_q_heads,
        num_v_heads=result.num_v_heads,
        public_raw_g_consumed=result.public_raw_g_consumed,
        global_g_cumsum_workspace=result.global_g_cumsum_workspace,
        separate_prefix_launch=result.separate_prefix_launch,
        fallback=result.fallback,
    )


def launch_i4_r24r2_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch the R24R2 subtraction-derived carry child on exact S3."""

    if (
        inputs.total_tokens != _S3_TOTAL_TOKENS
        or inputs.num_sequences != _S3_NUM_SEQUENCES
        or inputs.num_q_heads != _S3_NUM_HEADS
        or inputs.num_v_heads != _S3_NUM_HEADS
        or inputs.initial_state is not None
        or not inputs.output_final_state
        or inputs.output_state is None
    ):
        raise NotImplementedError(
            "I4-r24r2 experimental dispatch is frozen to exact S3: "
            "T=4096, N=20, Hq=Hv=16, no initial state, "
            "output_final_state=True",
        )

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            stream,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
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
            inputs.q,
            inputs.output,
            inputs.output_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R24R2_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )


def launch_i4_r30r1_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch the R30R1 full-ready-mbarrier-only child on exact S3."""

    if (
        inputs.total_tokens != _S3_TOTAL_TOKENS
        or inputs.num_sequences != _S3_NUM_SEQUENCES
        or inputs.num_q_heads != _S3_NUM_HEADS
        or inputs.num_v_heads != _S3_NUM_HEADS
        or inputs.initial_state is not None
        or not inputs.output_final_state
        or inputs.output_state is None
    ):
        raise NotImplementedError(
            "I4-r30r1 experimental dispatch is frozen to exact S3: "
            "T=4096, N=20, Hq=Hv=16, no initial state, "
            "output_final_state=True",
        )

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            stream,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            full_ready_mbarrier_only=True,
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
            inputs.q,
            inputs.output,
            inputs.output_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R30R1_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )


def launch_i4_r31_exact_s3(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> I4R23ExecutionInfo | None:
    """Launch the R31 factor-ready-lockstep-only child on exact S3."""

    if (
        inputs.total_tokens != _S3_TOTAL_TOKENS
        or inputs.num_sequences != _S3_NUM_SEQUENCES
        or inputs.num_q_heads != _S3_NUM_HEADS
        or inputs.num_v_heads != _S3_NUM_HEADS
        or inputs.initial_state is not None
        or not inputs.output_final_state
        or inputs.output_state is None
    ):
        raise NotImplementedError(
            "I4-r31 experimental dispatch is frozen to exact S3: "
            "T=4096, N=20, Hq=Hv=16, no initial state, "
            "output_final_state=True",
        )

    device = inputs.q.device
    with torch.cuda.device(device):
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        compiled = _compile(
            inputs,
            stream,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            full_ready_mbarrier_only=True,
            factor_ready_lockstep_only=True,
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
            inputs.q,
            inputs.output,
            inputs.output_state,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.total_tokens,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return I4R23ExecutionInfo(
        backend_id=I4_R31_BACKEND_ID,
        total_tokens=inputs.total_tokens,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        public_raw_g_consumed=True,
        global_g_cumsum_workspace=False,
        separate_prefix_launch=False,
        fallback=False,
    )

# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Frozen host-metadata contract for TC-M1 + SC-P0R1-V64CAP.

This module owns only the exact route, value-tile, launch-count, and capsule
layout decisions selected in N6.6-C.  It performs no CUDA value read and is
kept separate from the product kernels so source and dispatcher audits can
distinguish the immutable policy from its native realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import CHUNK_SIZE, HEAD_SIZE, SC_P0_BACKEND_ID, VALUE_SIZE

H20_SM_COUNT = 78
A3_SCHEDULE_ID = "A3-S2-SINGLEWG-V32"
A2_K0_SCHEDULE_ID = "K0-S1-AUX256"
A2_K1_SCHEDULE_ID = "K1-S2R1-METADATA-V64CAP"
ALGORITHM_CANDIDATE_ID = "gdn2-sm90-n6-6-tc-m1-v2-v64cap"
SCHEDULE_CANDIDATE_ID = "gdn2-sm90-n6-6-sc-p0r1-v64cap-v1"
DISPATCH_POLICY_ID = "gdn2-sm90-n6-6-tc-m1-sc-p0r1-v64cap-dispatch-v1"


class TCm1Route(str, Enum):
    """The two immutable TC-M1 algorithm routes."""

    A3_SHORT = "tc-m1-a3-short"
    A2_GENERAL = "tc-m1-a2-general"


@dataclass(frozen=True)
class OwnerCapsuleLayout:
    """Logical A2 K0-to-K1 publication for one chunk and query head."""

    g_cumsum_shape: tuple[int, int] = (CHUNK_SIZE, HEAD_SIZE)
    g_cumsum_dtype: str = "float32"
    aqk_scaled_shape: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE)
    aqk_scaled_dtype: str = "bfloat16"
    akk_inverse_shape: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE)
    akk_inverse_dtype: str = "bfloat16"

    @property
    def bytes_per_owner_chunk(self) -> int:
        return CHUNK_SIZE * HEAD_SIZE * 4 + 2 * CHUNK_SIZE * CHUNK_SIZE * 2


OWNER_CAPSULE_LAYOUT = OwnerCapsuleLayout()


@dataclass(frozen=True)
class SCP0Plan:
    """Exact metadata-derived plan for one validated public call."""

    backend_id: str
    algorithm_candidate_id: str
    schedule_candidate_id: str
    dispatch_policy_id: str
    route: TCm1Route
    launch_count: int
    a3_schedule_id: str | None
    a2_k0_schedule_id: str | None
    a2_k1_schedule_id: str | None
    k1_vtile: int
    capsule_chunk_capacity: int
    capsule_bytes: int
    k0_cta_capacity: int
    k1_ctas: int
    native_gva: bool
    qkgb_physically_expanded: bool
    fallback: bool


def _validate_metadata(
    total_tokens: int,
    num_sequences: int,
    num_q_heads: int,
    num_v_heads: int,
) -> None:
    for name, value in (
        ("total_tokens", total_tokens),
        ("num_sequences", num_sequences),
        ("num_q_heads", num_q_heads),
        ("num_v_heads", num_v_heads),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if num_v_heads < num_q_heads or num_v_heads % num_q_heads:
        raise ValueError(
            "TC-M1 requires Hv to equal Hq or be an integer multiple of Hq",
        )


def select_tc_m1_route(total_tokens: int) -> TCm1Route:
    """Select A3 for ``T <= 64`` and A2 for the exact complement."""

    if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
        raise TypeError("total_tokens must be an integer")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if total_tokens <= CHUNK_SIZE:
        return TCm1Route.A3_SHORT
    return TCm1Route.A2_GENERAL


def select_k1_vtile(
    *,
    num_sequences: int,
    num_q_heads: int,
    num_v_heads: int,
    target_sm_count: int = H20_SM_COUNT,
) -> int:
    """Apply the V64-cap metadata-only K1 value-tile policy.

    ``target_sm_count`` remains part of the stable host interface, but the
    rejected packed-V128 priority no longer depends on it.
    """

    _validate_metadata(1, num_sequences, num_q_heads, num_v_heads)
    if not isinstance(target_sm_count, int) or isinstance(target_sm_count, bool) or target_sm_count <= 0:
        raise ValueError("target_sm_count must be a positive integer")
    if num_v_heads > num_q_heads:
        return 64
    return 32


def capsule_chunk_capacity(
    *,
    total_tokens: int,
    num_sequences: int,
) -> int:
    """Return a metadata-only upper bound for packed sequence-local chunks.

    For positive sequence lengths, splitting ``T`` tokens across ``N``
    sequences can add at most ``N-1`` extra partial chunks compared with
    ``ceil(T/64)``.  Device code rejects the unused tail work units after
    resolving exact sequence-local chunk counts from CUDA ``cu_seqlens``.
    """

    if total_tokens <= 0 or num_sequences <= 0:
        raise ValueError("total_tokens and num_sequences must be positive")
    return (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE + num_sequences - 1


def make_sc_p0_plan(
    *,
    total_tokens: int,
    num_sequences: int,
    num_q_heads: int,
    num_v_heads: int,
    target_sm_count: int = H20_SM_COUNT,
) -> SCP0Plan:
    """Construct the exact TC-M1 + SC-P0R1-V64CAP plan from metadata."""

    _validate_metadata(
        total_tokens,
        num_sequences,
        num_q_heads,
        num_v_heads,
    )
    route = select_tc_m1_route(total_tokens)
    native_gva = num_v_heads > num_q_heads
    if route is TCm1Route.A3_SHORT:
        vtile = 32
        k1_ctas = num_sequences * num_v_heads * (VALUE_SIZE // vtile)
        return SCP0Plan(
            backend_id=SC_P0_BACKEND_ID,
            algorithm_candidate_id=ALGORITHM_CANDIDATE_ID,
            schedule_candidate_id=SCHEDULE_CANDIDATE_ID,
            dispatch_policy_id=DISPATCH_POLICY_ID,
            route=route,
            launch_count=1,
            a3_schedule_id=A3_SCHEDULE_ID,
            a2_k0_schedule_id=None,
            a2_k1_schedule_id=None,
            k1_vtile=vtile,
            capsule_chunk_capacity=0,
            capsule_bytes=0,
            k0_cta_capacity=0,
            k1_ctas=k1_ctas,
            native_gva=native_gva,
            qkgb_physically_expanded=False,
            fallback=False,
        )

    vtile = select_k1_vtile(
        num_sequences=num_sequences,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        target_sm_count=target_sm_count,
    )
    capacity = capsule_chunk_capacity(
        total_tokens=total_tokens,
        num_sequences=num_sequences,
    )
    return SCP0Plan(
        backend_id=SC_P0_BACKEND_ID,
        algorithm_candidate_id=ALGORITHM_CANDIDATE_ID,
        schedule_candidate_id=SCHEDULE_CANDIDATE_ID,
        dispatch_policy_id=DISPATCH_POLICY_ID,
        route=route,
        launch_count=2,
        a3_schedule_id=None,
        a2_k0_schedule_id=A2_K0_SCHEDULE_ID,
        a2_k1_schedule_id=A2_K1_SCHEDULE_ID,
        k1_vtile=vtile,
        capsule_chunk_capacity=capacity,
        capsule_bytes=(capacity * num_q_heads * OWNER_CAPSULE_LAYOUT.bytes_per_owner_chunk),
        k0_cta_capacity=capacity * num_q_heads,
        k1_ctas=(num_sequences * num_v_heads * ((VALUE_SIZE + vtile - 1) // vtile)),
        native_gva=native_gva,
        qkgb_physically_expanded=False,
        fallback=False,
    )

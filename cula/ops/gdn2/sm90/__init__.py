# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Private Candidate M development implementations for Hopper SM90a."""

from .b1 import B1ExecutionInfo, launch_b1_gdn2
from .config import (
    B1_BACKEND_ID,
    I4_R23_BACKEND_ID,
    I4_R24_BACKEND_ID,
    I4_R24R1_BACKEND_ID,
    I4_R24R2_BACKEND_ID,
    I4_R30R1_BACKEND_ID,
    I4_R31_BACKEND_ID,
    I4_R35_BACKEND_ID,
    PRIVATE_BACKEND_ID,
    PUBLIC_BACKEND_ID,
    RECURRENT_BACKEND_ID,
    SC_P0_BACKEND_ID,
    selected_public_backend_id,
)
from .launch import (
    FixedMHAIntermediates,
    RecurrentExecutionInfo,
    run_fixed_mha_recurrent,
    run_fixed_mha_single_chunk,
)
from .sc_p0_contract import (
    SCP0Plan,
    TCm1Route,
    make_sc_p0_plan,
)

__all__ = [
    "FixedMHAIntermediates",
    "B1ExecutionInfo",
    "RecurrentExecutionInfo",
    "SCP0Plan",
    "TCm1Route",
    "get_b1_backend_identity",
    "get_private_backend_identity",
    "get_public_backend_identity",
    "get_recurrent_backend_identity",
    "get_sc_p0_backend_identity",
    "make_sc_p0_plan",
    "run_fixed_mha_recurrent",
    "run_fixed_mha_single_chunk",
    "launch_b1_gdn2",
]


def get_private_backend_identity() -> str:
    """Return the exact private N4 backend identity."""

    return PRIVATE_BACKEND_ID


def get_recurrent_backend_identity() -> str:
    """Return the exact private N5 recurrent backend identity."""

    return RECURRENT_BACKEND_ID


def get_public_backend_identity() -> str:
    """Return the exact N6 Candidate M public backend identity."""

    return selected_public_backend_id()


def get_b1_backend_identity() -> str:
    """Return the exact private N6.5-B backend identity."""

    return B1_BACKEND_ID


def get_sc_p0_backend_identity() -> str:
    """Return the exact selected N6.6 TC-M1 + SC-P0 identity."""

    return SC_P0_BACKEND_ID

# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Frozen constants for the private Hopper GDN2 development backends."""

import os

CHUNK_SIZE = 64
HEAD_SIZE = 128
VALUE_SIZE = 128
THREADS_PER_CTA = 128
EXPECTED_CUTLASS_DSL_VERSION = "4.5.1"
PRIVATE_BACKEND_ID = "sm90a_cutedsl_gdn2_n4_candidate_m"
RECURRENT_BACKEND_ID = "sm90a_cutedsl_gdn2_n5_candidate_m_recurrent"
PUBLIC_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_candidate_m_packed_gva"
B1_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_5_candidate_b1"
SC_P0_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_tc_m1_sc_p0r1_v64cap_v1"
I4_R23_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r23_raw_g_exact_s3"
I4_R24_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r24_subgroup_prefix_exact_s3"
I4_R24R1_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r24r1_subgroup_prefix_exact_s3"
I4_R24R2_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r24r2_exclusive_carry_exact_s3"
I4_R30R1_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r30r1_full_ready_mbarrier_only_exact_s3"
I4_R31_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r31_factor_ready_lockstep_only_exact_s3"
I4_R35_BACKEND_ID = "sm90a_cutedsl_gdn2_n6_6_i4_r35_sequence_wave_rotate12_exact5"
R36_IDENTITY_BACKEND_ID = "sm90a_cutedsl_gdn2_r36_identity"
R36_LPT32_BACKEND_ID = "sm90a_cutedsl_gdn2_r36_stable_lpt32r4"
EXPERIMENTAL_BACKEND_ENV = "CULA_GDN2_SM90_EXPERIMENTAL_BACKEND"


def selected_public_backend_id() -> str:
    """Return the explicit public backend selection for this process."""

    selected = os.environ.get(EXPERIMENTAL_BACKEND_ENV)
    if selected in (None, "", PUBLIC_BACKEND_ID):
        return PUBLIC_BACKEND_ID
    if selected == I4_R23_BACKEND_ID:
        return I4_R23_BACKEND_ID
    if selected == I4_R24_BACKEND_ID:
        return I4_R24_BACKEND_ID
    if selected == I4_R24R1_BACKEND_ID:
        return I4_R24R1_BACKEND_ID
    if selected == I4_R24R2_BACKEND_ID:
        return I4_R24R2_BACKEND_ID
    if selected == I4_R30R1_BACKEND_ID:
        return I4_R30R1_BACKEND_ID
    if selected == I4_R31_BACKEND_ID:
        return I4_R31_BACKEND_ID
    if selected == I4_R35_BACKEND_ID:
        return I4_R35_BACKEND_ID
    if selected == R36_IDENTITY_BACKEND_ID:
        return R36_IDENTITY_BACKEND_ID
    if selected == R36_LPT32_BACKEND_ID:
        return R36_LPT32_BACKEND_ID
    raise RuntimeError(
        f"unsupported {EXPERIMENTAL_BACKEND_ENV}={selected!r}",
    )

// Copyright 2025-2026 Ant Group Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace cula::gdn::sm90 {

void
launch_gdn_fwd_prefill_kernel(
    cudaStream_t stream,
    void* output,
    float* output_state,
    void const* q,
    void const* k,
    void const* v,
    float const* input_state,
    float const* alpha,
    float const* beta,
    int64_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_q_heads,
    int32_t num_k_heads,
    int32_t num_v_heads,
    int32_t num_o_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    int32_t sm_count);

}  // namespace cula::gdn::sm90

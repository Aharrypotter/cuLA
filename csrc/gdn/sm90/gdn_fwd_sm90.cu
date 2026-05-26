// Copyright 2025-2026 Ant Group Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>

#include <cutlass/arch/arch.h>
#include <flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh>

#include "gdn/sm90/prefill_kernel.hpp"

namespace cula::gdn::sm90 {

using BFloat16 = nv_bfloat16;

template <bool IsGVA, bool NeedsBeta, bool NeedsAlpha, bool InitStateFromInput>
void
launch_bf16_no_checkpoint(
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
    int32_t sm_count) {
  flat::launch_delta_rule_prefill_kernel_gbai<IsGVA, NeedsBeta, NeedsAlpha, InitStateFromInput, false,
                                              cutlass::arch::Sm90, BFloat16, BFloat16, float>(
      stream, static_cast<BFloat16*>(output), output_state, static_cast<BFloat16 const*>(q),
      static_cast<BFloat16 const*>(k), static_cast<BFloat16 const*>(v), input_state, alpha, beta,
      cu_seqlens, workspace_buffer, num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads,
      head_size, total_seqlen, scale, sm_count, nullptr, nullptr, 0);
}

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
    int32_t sm_count) {
  bool const is_gva = num_v_heads > num_q_heads;
  bool const needs_beta = beta != nullptr;
  bool const needs_alpha = alpha != nullptr;
  bool const init_state = input_state != nullptr;

#define LAUNCH(is_gva_, needs_beta_, needs_alpha_, init_state_)                                      \
  launch_bf16_no_checkpoint<is_gva_, needs_beta_, needs_alpha_, init_state_>(                        \
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens, workspace_buffer, \
      num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale,  \
      sm_count)

#define DISPATCH_INIT(is_gva_, needs_beta_, needs_alpha_) \
  do {                                                    \
    if (init_state) {                                     \
      LAUNCH(is_gva_, needs_beta_, needs_alpha_, true);   \
    } else {                                              \
      LAUNCH(is_gva_, needs_beta_, needs_alpha_, false);  \
    }                                                     \
  } while (false)

  if (is_gva && needs_beta && needs_alpha) {
    DISPATCH_INIT(true, true, true);
  } else if (is_gva && needs_beta && !needs_alpha) {
    DISPATCH_INIT(true, true, false);
  } else if (is_gva && !needs_beta && needs_alpha) {
    DISPATCH_INIT(true, false, true);
  } else if (is_gva && !needs_beta && !needs_alpha) {
    DISPATCH_INIT(true, false, false);
  } else if (!is_gva && needs_beta && needs_alpha) {
    DISPATCH_INIT(false, true, true);
  } else if (!is_gva && needs_beta && !needs_alpha) {
    DISPATCH_INIT(false, true, false);
  } else if (!is_gva && !needs_beta && needs_alpha) {
    DISPATCH_INIT(false, false, true);
  } else {
    DISPATCH_INIT(false, false, false);
  }

#undef DISPATCH_INIT
#undef LAUNCH
}

}  // namespace cula::gdn::sm90

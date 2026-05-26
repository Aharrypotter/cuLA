// Copyright 2025-2026 Ant Group Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "gdn/sm90/prefill_kernel.hpp"

using OptionalTensor = std::optional<torch::Tensor>;

void
gdn_fwd_prefill_sm90(
    torch::Tensor const& output,
    torch::Tensor const& output_state,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& cu_seqlens,
    OptionalTensor input_state_,
    OptionalTensor alpha_,
    OptionalTensor beta_,
    float scale,
    torch::Tensor workspace_buffer) {
  auto packed_seq = q.size(0);
  auto num_q_heads = q.size(1);
  auto num_k_heads = k.size(1);
  auto num_v_heads = v.size(1);
  auto num_o_heads = output.size(1);
  auto head_size = q.size(2);
  auto num_seqs = cu_seqlens.size(0) - 1;

  TORCH_CHECK(q.dtype() == torch::kBFloat16, "SM90 GDN q must be bfloat16");
  TORCH_CHECK(k.dtype() == torch::kBFloat16, "SM90 GDN k must be bfloat16");
  TORCH_CHECK(v.dtype() == torch::kBFloat16, "SM90 GDN v must be bfloat16");
  TORCH_CHECK(output.dtype() == torch::kBFloat16, "SM90 GDN output must be bfloat16");
  TORCH_CHECK(output_state.dtype() == torch::kFloat32, "SM90 GDN output_state must be float32");
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64, "SM90 GDN cu_seqlens must be int64");
  TORCH_CHECK(workspace_buffer.dtype() == torch::kUInt8, "SM90 GDN workspace_buffer must be uint8");

  TORCH_CHECK(q.is_contiguous(), "SM90 GDN q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "SM90 GDN k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "SM90 GDN v must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "SM90 GDN output must be contiguous");
  TORCH_CHECK(output_state.is_contiguous(), "SM90 GDN output_state must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "SM90 GDN cu_seqlens must be contiguous");
  TORCH_CHECK(workspace_buffer.is_contiguous(), "SM90 GDN workspace_buffer must be contiguous");

  TORCH_CHECK(k.size(0) == packed_seq && v.size(0) == packed_seq, "SM90 GDN q/k/v token dimensions must match");
  TORCH_CHECK(output.size(0) == packed_seq, "SM90 GDN output token dimension must match q");
  TORCH_CHECK(k.size(2) == head_size && v.size(2) == head_size && output.size(2) == head_size,
              "SM90 GDN q/k/v/output head dimensions must match");
  TORCH_CHECK(output_state.size(0) == num_seqs && output_state.size(1) == num_o_heads &&
                  output_state.size(2) == head_size && output_state.size(3) == head_size,
              "SM90 GDN output_state shape must be [num_seqs, num_o_heads, head_size, head_size]");

  if (num_q_heads >= num_v_heads) {
    TORCH_CHECK(num_k_heads == num_v_heads, "SM90 GDN GQA requires num_k_heads == num_v_heads");
    TORCH_CHECK(num_q_heads % num_k_heads == 0, "SM90 GDN GQA requires num_q_heads % num_k_heads == 0");
  } else {
    TORCH_CHECK(num_k_heads == num_q_heads, "SM90 GDN GVA requires num_k_heads == num_q_heads");
    TORCH_CHECK(num_v_heads % num_q_heads == 0, "SM90 GDN GVA requires num_v_heads % num_q_heads == 0");
  }

  float const* input_state_ptr = nullptr;
  if (input_state_.has_value()) {
    auto const& input_state = input_state_.value();
    TORCH_CHECK(input_state.dtype() == torch::kFloat32, "SM90 GDN input_state must be float32");
    TORCH_CHECK(input_state.is_contiguous(), "SM90 GDN input_state must be contiguous");
    TORCH_CHECK(input_state.sizes() == output_state.sizes(), "SM90 GDN input_state shape must match output_state");
    input_state_ptr = input_state.data_ptr<float>();
  }

  float const* alpha_ptr = nullptr;
  if (alpha_.has_value()) {
    auto const& alpha = alpha_.value();
    TORCH_CHECK(alpha.dtype() == torch::kFloat32, "SM90 GDN alpha must be float32");
    TORCH_CHECK(alpha.is_contiguous(), "SM90 GDN alpha must be contiguous");
    TORCH_CHECK(alpha.size(0) == packed_seq && alpha.size(1) == num_o_heads,
                "SM90 GDN alpha shape must be [packed_seq, num_o_heads]");
    alpha_ptr = alpha.data_ptr<float>();
  }

  float const* beta_ptr = nullptr;
  if (beta_.has_value()) {
    auto const& beta = beta_.value();
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "SM90 GDN beta must be float32");
    TORCH_CHECK(beta.is_contiguous(), "SM90 GDN beta must be contiguous");
    TORCH_CHECK(beta.size(0) == packed_seq && beta.size(1) == num_o_heads,
                "SM90 GDN beta shape must be [packed_seq, num_o_heads]");
    beta_ptr = beta.data_ptr<float>();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  cula::gdn::sm90::launch_gdn_fwd_prefill_kernel(
      stream, output.data_ptr(), output_state.data_ptr<float>(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      input_state_ptr, alpha_ptr, beta_ptr, cu_seqlens.data_ptr<int64_t>(), workspace_buffer.data_ptr<uint8_t>(),
      static_cast<int32_t>(num_seqs), static_cast<int32_t>(num_q_heads), static_cast<int32_t>(num_k_heads),
      static_cast<int32_t>(num_v_heads), static_cast<int32_t>(num_o_heads), static_cast<int32_t>(head_size),
      static_cast<int64_t>(packed_seq), scale, static_cast<int32_t>(sm_count));
}

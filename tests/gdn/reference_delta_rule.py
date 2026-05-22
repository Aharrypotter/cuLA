"""
Reference GDN delta-rule implementation adapted from FlashInfer tests.

Copyright (c) 2025 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import Optional

import torch


def exclusive_cumsum(values: list[int]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def seq_lens_to_cu_seqlens(seq_lens: list[int], *, device: torch.device, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.tensor(exclusive_cumsum(seq_lens), device=device, dtype=dtype)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype in (torch.float16, torch.bfloat16) or b.dtype in (torch.float16, torch.bfloat16):
        result = a.to(torch.float32) @ b.to(torch.float32)
        return result if a.dtype == torch.bfloat16 else result.to(torch.float16)
    return a @ b


def identity_add_strict_lower_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.size(-1)
    assert matrix.size(-2) == size
    matrix = matrix.clone()
    with torch.device(matrix.device):
        mask = torch.arange(size).unsqueeze(1) <= torch.arange(size)
        matrix[:, mask] = 0.0
        matrix = matrix + torch.eye(size).unsqueeze(0)
    return matrix


def to_logspace_gamma(alpha_hs: torch.Tensor, epsilon: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    gate = torch.log(alpha_hs + epsilon)
    cumsum_gate = torch.cumsum(gate, dim=-1)
    gamma_hss = cumsum_gate.unsqueeze(2) - cumsum_gate.unsqueeze(1)
    gamma_hs1 = cumsum_gate.unsqueeze(2)
    return gamma_hss, gamma_hs1


def _linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qk_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_heads = q.shape[1]
    q_hsq = q.transpose(0, 1)
    k_hsk = k.transpose(0, 1)
    v_hsv = v.transpose(0, 1)

    scores = matmul(q_hsq, k_hsk.transpose(-2, -1))
    seq_len = q_hsq.size(-2)
    mask = torch.tril(torch.ones(num_heads, seq_len, seq_len, dtype=q.dtype, device=q.device))
    if qk_weight is None:
        weights = mask
    else:
        weights = qk_weight.clone()
        weights[mask == 0.0] = 0.0

    output = matmul(scores * weights, v_hsv)
    return output.transpose(0, 1)


@torch.inference_mode()
def blockwise_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: list[int],
    *,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    block_size: int = 64,
    scale_factor: float = 1.0,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    num_o_heads = max(num_q_heads, num_v_heads)
    head_size = q.size(2)

    if sum(seq_lens) != total_tokens:
        raise ValueError(f"sum(seq_lens) must equal total tokens, got {sum(seq_lens)} and {total_tokens}")
    if alpha is None:
        alpha = torch.ones(total_tokens, num_o_heads, dtype=torch.float32, device=q.device)
    if beta is None:
        beta = torch.ones(total_tokens, num_o_heads, dtype=torch.float32, device=q.device)

    if num_q_heads >= num_v_heads:
        if num_k_heads != num_v_heads:
            raise ValueError("GQA reference requires num_k_heads == num_v_heads")
        qkv_heads = num_q_heads
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_v_heads, dim=1)
    else:
        if num_k_heads != num_q_heads:
            raise ValueError("GVA reference requires num_k_heads == num_q_heads")
        qkv_heads = num_v_heads
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    if initial_state is not None:
        expected_state_shape = (len(seq_lens), num_o_heads, head_size, head_size)
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}")

    output = torch.zeros(total_tokens, num_o_heads, head_size, dtype=q.dtype, device=q.device)
    final_state_hkv = torch.zeros(len(seq_lens), num_o_heads, head_size, head_size, dtype=state_dtype, device=q.device)

    offsets = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(offsets[:-1]):
        seq_end = offsets[seq_idx + 1]
        block_start = seq_start
        state_hkv = (
            initial_state[seq_idx].to(state_dtype).clone()
            if initial_state is not None
            else torch.zeros(num_o_heads, head_size, head_size, dtype=state_dtype, device=q.device)
        )
        while block_start < seq_end:
            valid_len = min(block_size, seq_end - block_start)
            block_end = block_start + valid_len

            q_blk = torch.zeros(block_size, qkv_heads, head_size, dtype=q.dtype, device=q.device)
            k_blk = torch.zeros(block_size, qkv_heads, head_size, dtype=k.dtype, device=k.device)
            v_blk = torch.zeros(block_size, qkv_heads, head_size, dtype=v.dtype, device=v.device)
            alpha_blk = torch.ones(block_size, num_o_heads, dtype=alpha.dtype, device=alpha.device)
            beta_blk = torch.zeros(block_size, num_o_heads, dtype=beta.dtype, device=beta.device)

            q_blk[:valid_len] = q[block_start:block_end]
            k_blk[:valid_len] = k[block_start:block_end]
            v_blk[:valid_len] = v[block_start:block_end]
            alpha_blk[:valid_len] = alpha[block_start:block_end]
            beta_blk[:valid_len] = beta[block_start:block_end]

            alpha_hs = alpha_blk.transpose(0, 1)
            beta_hs1 = beta_blk.transpose(0, 1).unsqueeze(2)
            gamma_hss, gamma_hs1 = to_logspace_gamma(alpha_hs)
            block_gamma = gamma_hs1[:, [valid_len - 1], :]

            q_hsq = q_blk.transpose(0, 1)
            k_hsk = k_blk.transpose(0, 1)
            v_hsv = v_blk.transpose(0, 1)

            ikk = identity_add_strict_lower_diagonal(
                beta_hs1 * torch.exp(gamma_hss) * matmul(k_hsk, k_hsk.transpose(-2, -1))
            )
            t_matrix = torch.inverse(ikk) * beta_hs1.transpose(1, 2)
            t_matrix = t_matrix.to(q.dtype)
            u_hsv = matmul(t_matrix, v_hsv)
            w_hsk = matmul(t_matrix, torch.exp(gamma_hs1) * k_hsk)
            new_v_hsv = u_hsv - matmul(w_hsk.to(torch.float32), state_hkv.to(torch.float32)).to(u_hsv.dtype)
            new_v = new_v_hsv.transpose(0, 1)

            o_inter = (
                matmul(torch.exp(gamma_hs1) * q_hsq.to(torch.float32), state_hkv.to(torch.float32))
                .transpose(0, 1)
                .to(q.dtype)
            )
            o_intra = _linear_attention(q_blk, k_blk, new_v, qk_weight=torch.exp(gamma_hss))
            output[block_start:block_end] = scale_factor * (o_inter + o_intra)[:valid_len]

            inc_hkv = matmul(
                (torch.exp(block_gamma - gamma_hs1) * k_hsk).transpose(-2, -1).to(torch.float32),
                new_v_hsv.to(torch.float32),
            )
            state_hkv = (torch.exp(block_gamma) * state_hkv.to(torch.float32) + inc_hkv).to(state_dtype)
            block_start += block_size

        final_state_hkv[seq_idx] = state_hkv

    return output, final_state_hkv

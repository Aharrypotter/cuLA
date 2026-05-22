import math

import pytest
import torch
import torch.nn.functional as F

from cula.gdn import chunk_gated_delta_rule
from reference_delta_rule import blockwise_delta_rule, seq_lens_to_cu_seqlens


def _cuda_major() -> int:
    if torch.version.cuda is None:
        return 0
    return int(torch.version.cuda.split(".")[0])


def test_reference_delta_rule_shape_and_continuation_cpu():
    torch.manual_seed(0)
    seq_lens = [5, 7]
    total_tokens = sum(seq_lens)
    head_size = 8
    q = torch.randn(total_tokens, 1, head_size, dtype=torch.float32)
    k = torch.randn(total_tokens, 1, head_size, dtype=torch.float32)
    v = torch.randn(total_tokens, 2, head_size, dtype=torch.float32)
    alpha = torch.rand(total_tokens, 2, dtype=torch.float32) * 0.2 + 0.75
    beta = torch.rand(total_tokens, 2, dtype=torch.float32)

    out, state = blockwise_delta_rule(
        q,
        k,
        v,
        seq_lens,
        alpha=alpha,
        beta=beta,
        block_size=4,
        scale_factor=1.0 / math.sqrt(head_size),
    )
    out_next, state_next = blockwise_delta_rule(
        q,
        k,
        v,
        seq_lens,
        alpha=alpha,
        beta=beta,
        initial_state=state,
        block_size=4,
        scale_factor=1.0 / math.sqrt(head_size),
    )

    assert out.shape == (total_tokens, 2, head_size)
    assert state.shape == (len(seq_lens), 2, head_size, head_size)
    assert out_next.shape == out.shape
    assert state_next.shape == state.shape
    assert not torch.equal(out_next, out)


def test_api_rejects_cpu_tensors_before_kernel_launch():
    q = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="q must be a CUDA tensor"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens)


@pytest.mark.sm100_only
def test_sm100_gdn_prefill_matches_reference_smoke():
    torch.manual_seed(1)
    device = torch.device("cuda")
    seq_lens = [64]
    total_tokens = sum(seq_lens)
    q = torch.randn(total_tokens, 1, 128, dtype=torch.bfloat16, device=device)
    k = F.normalize(torch.randn(total_tokens, 1, 128, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(total_tokens, 2, 128, dtype=torch.bfloat16, device=device) * 0.25
    g = torch.rand(total_tokens, 2, dtype=torch.float32, device=device) * 0.2 + 0.75
    beta = torch.rand(total_tokens, 2, dtype=torch.float32, device=device)
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(128)

    if _cuda_major() < 13:
        with pytest.raises(NotImplementedError, match="CUDA 13\\+"):
            chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens, scale=scale)
        return

    actual = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens, scale=scale)
    expected, _ = blockwise_delta_rule(
        q,
        k,
        v,
        seq_lens,
        alpha=g,
        beta=beta,
        block_size=64,
        scale_factor=scale,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)


@pytest.mark.sm100_only
def test_api_rejects_checkpointing_until_enabled():
    device = torch.device("cuda")
    q = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    k = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    v = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32, device=device)

    with pytest.raises(NotImplementedError, match="checkpointing is deferred"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens, checkpoint_every_n_tokens=64)

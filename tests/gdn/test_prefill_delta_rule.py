import math

import pytest
import torch
import torch.nn.functional as F

from cula.gdn import chunk_gated_delta_rule, get_sm90_gdn_prefill_backend
from reference_delta_rule import blockwise_delta_rule, seq_lens_to_cu_seqlens


def _cuda_major() -> int:
    if torch.version.cuda is None:
        return 0
    return int(torch.version.cuda.split(".")[0])


def _is_gdn_prefill_device() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major == 9 and minor == 0) or major >= 10


def _is_sm90_device() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 9 and minor == 0


def test_sm90_backend_env_selection(monkeypatch):
    monkeypatch.delenv("CULA_GDN_SM90_BACKEND", raising=False)
    assert get_sm90_gdn_prefill_backend() == "cutlass"

    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")
    assert get_sm90_gdn_prefill_backend() == "dsl"

    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "bad")
    with pytest.raises(ValueError, match="CULA_GDN_SM90_BACKEND"):
        get_sm90_gdn_prefill_backend()


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


@pytest.mark.skipif(not _is_gdn_prefill_device(), reason="GDN prefill smoke requires SM90 or SM100")
def test_gdn_prefill_matches_reference_smoke():
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

    major, minor = torch.cuda.get_device_capability()
    if major >= 10 and _cuda_major() < 13:
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


@pytest.mark.skipif(not _is_gdn_prefill_device(), reason="GDN prefill issue #76 smoke requires SM90 or SM100")
def test_gdn_prefill_issue76_mha64_varlen_state_smoke():
    torch.manual_seed(2)
    device = torch.device("cuda")
    seq_lens = [31, 33]
    total_tokens = sum(seq_lens)
    num_heads = 64
    head_size = 128
    q = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device)
    k = F.normalize(torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device), p=2, dim=-1)
    v = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.25
    g = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.2 + 0.75
    beta = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.01
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    major, minor = torch.cuda.get_device_capability()
    if major >= 10 and _cuda_major() < 13:
        with pytest.raises(NotImplementedError, match="CUDA 13\\+"):
            chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                scale=scale,
            )
        return

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )
    expected = expected.to(q.dtype)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=6e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL smoke requires Hopper")
@pytest.mark.parametrize(
    ("num_q_heads", "num_k_heads", "num_v_heads"),
    [
        pytest.param(2, 2, 2, id="mha"),
        pytest.param(4, 1, 1, id="gqa"),
        pytest.param(1, 1, 2, id="gva"),
    ],
)
def test_gdn_prefill_sm90_dsl_minimal_head_mapping_matches_reference(
    monkeypatch,
    num_q_heads,
    num_k_heads,
    num_v_heads,
):
    torch.manual_seed(3)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    head_size = 128
    q = torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.bfloat16, device=device) * 0.25
    k = F.normalize(
        torch.randn(total_tokens, num_k_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = torch.randn(
        len(seq_lens),
        num_o_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.005
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=6e-2, atol=6e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL no-initial-state smoke requires Hopper")
def test_gdn_prefill_sm90_dsl_no_initial_state_multi_chunk_matches_reference(monkeypatch):
    torch.manual_seed(8)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [70]
    total_tokens = sum(seq_lens)
    num_heads = 2
    head_size = 128
    q = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.2
    k = F.normalize(
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.5
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL multi-chunk smoke requires Hopper")
def test_gdn_prefill_sm90_dsl_multi_chunk_staged_output_matches_reference(monkeypatch):
    torch.manual_seed(4)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [70]
    total_tokens = sum(seq_lens)
    num_heads = 2
    head_size = 128
    q = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.2
    k = F.normalize(
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = torch.randn(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.005
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=6e-2, atol=6e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL varlen multi-chunk smoke requires Hopper")
def test_gdn_prefill_sm90_dsl_varlen_multi_chunk_gva_matches_reference(monkeypatch):
    torch.manual_seed(6)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [1, 65]
    total_tokens = sum(seq_lens)
    num_q_heads = 1
    num_k_heads = 1
    num_v_heads = 2
    num_o_heads = num_v_heads
    head_size = 128
    q = torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.bfloat16, device=device) * 0.2
    k = F.normalize(
        torch.randn(total_tokens, num_k_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = torch.randn(
        len(seq_lens),
        num_o_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.005
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL qwen3-next smoke requires Hopper")
def test_gdn_prefill_sm90_dsl_qwen3_next_head_mapping_matches_reference(monkeypatch):
    torch.manual_seed(5)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [4]
    total_tokens = sum(seq_lens)
    num_q_heads = 16
    num_k_heads = 16
    num_v_heads = 32
    num_o_heads = num_v_heads
    head_size = 128
    q = torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.bfloat16, device=device) * 0.2
    k = F.normalize(
        torch.randn(total_tokens, num_k_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = torch.randn(
        len(seq_lens),
        num_o_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.005
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(not _is_sm90_device(), reason="SM90 DSL issue #76 tiny smoke requires Hopper")
def test_gdn_prefill_sm90_dsl_issue76_mha64_tiny_matches_reference(monkeypatch):
    torch.manual_seed(7)
    monkeypatch.setenv("CULA_GDN_SM90_BACKEND", "dsl")

    device = torch.device("cuda")
    seq_lens = [16]
    total_tokens = sum(seq_lens)
    num_heads = 64
    head_size = 128
    q = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.2
    k = F.normalize(
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_tokens, num_heads, head_size, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    beta = torch.rand(total_tokens, num_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = torch.randn(
        len(seq_lens),
        num_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=device,
    ) * 0.005
    cu_seqlens = seq_lens_to_cu_seqlens(seq_lens, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_size)

    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    expected, expected_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        alpha=g,
        beta=beta,
        initial_state=initial_state,
        block_size=64,
        scale_factor=scale,
        state_dtype=torch.float32,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(actual_state.transpose(-1, -2), expected_state, rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_api_rejects_checkpointing_until_enabled():
    device = torch.device("cuda")
    q = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    k = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    v = torch.empty(64, 1, 128, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32, device=device)

    with pytest.raises(NotImplementedError, match="checkpointing is deferred"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens, checkpoint_every_n_tokens=64)

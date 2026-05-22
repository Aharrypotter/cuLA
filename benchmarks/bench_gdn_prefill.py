#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark cuLA GDN chunked prefill on Blackwell.

This benchmark assumes the SM100 GDN kernel is available, which currently means
CUDA 13+ and a Blackwell GPU. First-call CuTe DSL compilation is measured
separately from steady-state latency.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from cula.gdn import chunk_gated_delta_rule, is_sm100_gdn_prefill_available


HEAD_PRESETS = {
    "qwen3-next": (16, 16, 32),
    "gqa-small": (4, 1, 1),
    "gva-small": (1, 1, 2),
    "mha-small": (4, 4, 4),
}

DTYPES = {
    "bfloat16": torch.bfloat16,
}


def exclusive_cumsum(values: list[int]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gdn_tflops(total_tokens: int, num_heads: int, head_size: int, latency_ms: float) -> float:
    # Two dominant D x D interactions per token/head: state update and q @ state.
    flops = 4 * total_tokens * num_heads * head_size * head_size
    return flops / latency_ms / 1e9


def make_inputs(
    seq_lens: list[int],
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    use_initial_state: bool,
    output_final_state: bool,
) -> dict[str, torch.Tensor | None]:
    total_tokens = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    q = torch.randn(total_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(total_tokens, num_k_heads, head_size, dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(total_tokens, num_v_heads, head_size, dtype=dtype, device=device) * 0.25
    g = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.2 + 0.75
    beta = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(
            len(seq_lens),
            num_o_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        ) * 0.01
    output_state = None
    if output_final_state:
        output_state = torch.empty(
            len(seq_lens),
            num_o_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "output_state": output_state,
    }


def run_one_config(
    preset: str,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_initial_state: bool,
    output_final_state: bool,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    num_q_heads, num_k_heads, num_v_heads = HEAD_PRESETS[preset]
    head_size = 128
    seq_lens = [seq_len] * batch_size
    inputs = make_inputs(
        seq_lens,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        device,
        use_initial_state,
        output_final_state,
    )
    scale = 1.0 / math.sqrt(head_size)

    def fn():
        return chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            scale=scale,
            initial_state=inputs["initial_state"],
            output_final_state=output_final_state,
            cu_seqlens=inputs["cu_seqlens"],
            output_state=inputs["output_state"],
        )

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000
    latency_ms = time_cuda(fn, warmup, iters)
    total_tokens = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    return {
        "preset": preset,
        "batch": batch_size,
        "seq_len": seq_len,
        "tokens": total_tokens,
        "hq": num_q_heads,
        "hk": num_k_heads,
        "hv": num_v_heads,
        "compile_ms": compile_ms,
        "latency_ms": latency_ms,
        "tflops": gdn_tflops(total_tokens, num_o_heads, head_size, latency_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cuLA GDN chunked prefill")
    parser.add_argument("--preset", choices=sorted(HEAD_PRESETS), default="qwen3-next")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--initial-state", action="store_true")
    parser.add_argument("--output-final-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not is_sm100_gdn_prefill_available():
        raise SystemExit(
            "SM100 GDN prefill is unavailable. It requires a Blackwell GPU, "
            f"CUDA 13+, and nvidia-cutlass-dsl[cu13]. torch.version.cuda={torch.version.cuda!r}"
        )

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(
        f"{'preset':<12} {'B':>4} {'T':>6} {'tokens':>8} "
        f"{'heads(q/k/v)':>14} {'compile(ms)':>12} {'latency(ms)':>12} {'TFLOPS':>8}"
    )

    dtype = DTYPES[args.dtype]
    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            row = run_one_config(
                args.preset,
                batch_size,
                seq_len,
                dtype,
                args.warmup,
                args.iters,
                args.initial_state,
                args.output_final_state,
            )
            print(
                f"{row['preset']:<12} {row['batch']:>4} {row['seq_len']:>6} {row['tokens']:>8} "
                f"{row['hq']:>4}/{row['hk']:<4}/{row['hv']:<4} "
                f"{row['compile_ms']:>12.2f} {row['latency_ms']:>12.3f} {row['tflops']:>8.1f}"
            )


if __name__ == "__main__":
    main()

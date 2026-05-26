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

"""Benchmark cuLA GDN chunked prefill.

SM90 uses the cuLA CUDA extension path; SM100 uses the CuTe DSL Blackwell path.
First-call compilation/setup is measured separately from steady-state latency.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import CUDA_HOME

from cula.gdn import (
    chunk_gated_delta_rule as cula_chunk_gated_delta_rule,
    get_sm90_gdn_prefill_backend,
    is_sm90_gdn_prefill_available,
    is_sm100_gdn_prefill_available,
)


HEAD_PRESETS = {
    "issue76": (64, 64, 64),
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


def uniform_seq_lens(num_seqs: int, total_len: int) -> list[int]:
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    if total_len < num_seqs:
        raise ValueError(f"total_len must be >= num_seqs, got total_len={total_len}, num_seqs={num_seqs}")
    base = total_len // num_seqs
    remainder = total_len % num_seqs
    return [base + (1 if i < remainder else 0) for i in range(num_seqs)]


def random_seq_lens(num_seqs: int, total_len: int, seed: int) -> list[int]:
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    if total_len < num_seqs:
        raise ValueError(f"total_len must be >= num_seqs, got total_len={total_len}, num_seqs={num_seqs}")
    if num_seqs == 1:
        return [total_len]
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, total_len), num_seqs - 1))
    return [end - start for start, end in zip([0] + cuts, cuts + [total_len])]


def skewed_seq_lens(num_seqs: int, total_len: int) -> list[int]:
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    if total_len < num_seqs:
        raise ValueError(f"total_len must be >= num_seqs, got total_len={total_len}, num_seqs={num_seqs}")
    remaining = total_len - num_seqs
    weights = [(num_seqs - i) ** 2 for i in range(num_seqs)]
    weight_sum = sum(weights)
    lengths = [1 + remaining * weight // weight_sum for weight in weights]
    lengths[0] += total_len - sum(lengths)
    return lengths


def make_varlen_seq_lens(distribution: str, num_seqs: int, total_len: int, seed: int) -> list[int]:
    if distribution == "uniform":
        return uniform_seq_lens(num_seqs, total_len)
    if distribution == "random":
        return random_seq_lens(num_seqs, total_len, seed)
    if distribution == "skewed":
        return skewed_seq_lens(num_seqs, total_len)
    raise ValueError(f"unknown distribution {distribution!r}")


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
    chunk_gated_delta_rule,
    preset: str,
    mode: str,
    seq_lens: list[int],
    distribution: str,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_initial_state: bool,
    output_final_state: bool,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    num_q_heads, num_k_heads, num_v_heads = HEAD_PRESETS[preset]
    head_size = 128
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
        "mode": mode,
        "distribution": distribution,
        "preset": preset,
        "num_seqs": len(seq_lens),
        "max_seq_len": max(seq_lens),
        "tokens": total_tokens,
        "hq": num_q_heads,
        "hk": num_k_heads,
        "hv": num_v_heads,
        "compile_ms": compile_ms,
        "latency_ms": latency_ms,
        "tflops": gdn_tflops(total_tokens, num_o_heads, head_size, latency_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cuLA/FlashInfer GDN chunked prefill",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["cula", "flashinfer"],
        default="cula",
        help="Benchmark the migrated cuLA kernel or the upstream FlashInfer kernel.",
    )
    parser.add_argument(
        "--cula-sm90-backend",
        choices=["cutlass", "dsl"],
        default=None,
        help="When --backend=cula on SM90, select the migrated CUTLASS path or the experimental CuTe DSL path.",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "varlen"],
        default="fixed",
        help="Run a fixed-length batch sweep or a variable-length packed-sequence sweep.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(HEAD_PRESETS),
        default="qwen3-next",
        help="Select the head layout; issue76 means q/k/v = 64/64/64 and head size 128.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Fixed-mode batch sizes; each batch uses identical sequence lengths.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Fixed-mode sequence lengths to repeat across the batch.",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Varlen-mode sequence counts in the packed batch.",
    )
    parser.add_argument(
        "--total-len",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384],
        help="Varlen-mode total token counts across all packed sequences.",
    )
    parser.add_argument(
        "--distribution",
        choices=["uniform", "random", "skewed"],
        nargs="+",
        default=["uniform", "random", "skewed"],
        help="Varlen-mode sequence length distribution.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="Input dtype for q/k/v.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for varlen random layout.")
    parser.add_argument(
        "--initial-state",
        action="store_true",
        help="Materialize a non-empty initial state tensor before the kernel call.",
    )
    parser.add_argument(
        "--output-final-state",
        action="store_true",
        help="Request the final recurrent state from the kernel.",
    )
    return parser.parse_args()


def load_backend(args: argparse.Namespace):
    if args.backend == "cula":
        if args.cula_sm90_backend is not None:
            os.environ["CULA_GDN_SM90_BACKEND"] = args.cula_sm90_backend
        if not (is_sm90_gdn_prefill_available() or is_sm100_gdn_prefill_available()):
            raise SystemExit(
                "cuLA GDN prefill is unavailable. It requires an SM90 GPU or an SM100/SM103 GPU "
                f"with CUDA 13+. torch.version.cuda={torch.version.cuda!r}"
            )
        return cula_chunk_gated_delta_rule

    if args.cula_sm90_backend is not None:
        raise SystemExit("--cula-sm90-backend only applies when --backend=cula")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    flashinfer_root = repo_root / "3rdparty" / "flashinfer"
    cula_cutlass_root = repo_root / "csrc" / "cutlass"
    cuda_home = pathlib.Path(CUDA_HOME or "")
    cuda_cccl_root = cuda_home / "include" / "cccl"
    sys.path.insert(0, str(flashinfer_root))
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as flashinfer_chunk_gated_delta_rule
    from flashinfer.jit import env as flashinfer_jit_env
    from flashinfer.utils import get_compute_capability

    if not (cula_cutlass_root / "include" / "cute" / "tensor.hpp").exists():
        raise SystemExit(f"missing cuLA CUTLASS checkout: {cula_cutlass_root}")
    if not (cuda_cccl_root / "cub" / "cub.cuh").exists():
        raise SystemExit(f"missing CUDA CCCL headers: {cuda_cccl_root}")
    if (cuda_home / "lib").exists():
        extra_ldflags = os.environ.get("FLASHINFER_EXTRA_LDFLAGS", "")
        cuda_lib_flag = f"-L{cuda_home / 'lib'}"
        if cuda_lib_flag not in extra_ldflags.split():
            os.environ["FLASHINFER_EXTRA_LDFLAGS"] = " ".join(
                flag for flag in [extra_ldflags, cuda_lib_flag] if flag
            )

    # Source-tree FlashInfer defaults to package data paths. Override them so
    # PR #2276's SM90 JIT can run without a built wheel or initialized submodules.
    flashinfer_jit_env.FLASHINFER_CSRC_DIR = flashinfer_root / "csrc"
    flashinfer_jit_env.FLASHINFER_INCLUDE_DIR = flashinfer_root / "include"
    flashinfer_jit_env.CUTLASS_INCLUDE_DIRS = [
        cula_cutlass_root / "include",
        cula_cutlass_root / "tools" / "util" / "include",
    ]
    flashinfer_jit_env.CCCL_INCLUDE_DIRS = [cuda_cccl_root]

    major, minor = get_compute_capability(torch.device("cuda"))
    if major < 9:
        raise SystemExit(f"FlashInfer GDN prefill requires SM90+, got sm_{major}{minor}")
    return flashinfer_chunk_gated_delta_rule


def main() -> None:
    args = parse_args()
    chunk_gated_delta_rule = load_backend(args)

    device = torch.device("cuda")
    print(f"backend: {args.backend}")
    if args.backend == "cula" and is_sm90_gdn_prefill_available(device):
        print(f"cuLA SM90 backend: {get_sm90_gdn_prefill_backend()}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(
        f"{'mode':<7} {'dist':<8} {'preset':<12} {'N':>4} {'maxT':>6} {'tokens':>8} "
        f"{'heads(q/k/v)':>14} {'compile(ms)':>12} {'latency(ms)':>12} {'TFLOPS':>8}"
    )

    dtype = DTYPES[args.dtype]
    if args.mode == "fixed":
        configs = [
            ("fixed", [seq_len] * batch_size, "-")
            for batch_size in args.batch_size
            for seq_len in args.seq_len
        ]
    else:
        configs = [
            (
                "varlen",
                make_varlen_seq_lens(distribution, num_seqs, total_len, args.seed + num_seqs + total_len),
                distribution,
            )
            for distribution in args.distribution
            for num_seqs in args.num_seqs
            for total_len in args.total_len
        ]

    for mode, seq_lens, distribution in configs:
        row = run_one_config(
            chunk_gated_delta_rule,
            args.preset,
            mode,
            seq_lens,
            distribution,
            dtype,
            args.warmup,
            args.iters,
            args.initial_state,
            args.output_final_state,
        )
        print(
            f"{row['mode']:<7} {row['distribution']:<8} {row['preset']:<12} "
            f"{row['num_seqs']:>4} {row['max_seq_len']:>6} {row['tokens']:>8} "
            f"{row['hq']:>4}/{row['hk']:<4}/{row['hv']:<4} "
            f"{row['compile_ms']:>12.2f} {row['latency_ms']:>12.3f} {row['tflops']:>8.1f}"
        )


if __name__ == "__main__":
    main()

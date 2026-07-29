# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Candidate M output and public [V,K] final-state publication."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .config import CHUNK_SIZE, HEAD_SIZE, THREADS_PER_CTA

_INV_LN2 = 1.4426950408889634


class OutputAndState:
    """Consume QK/U/Kbar and publish the N4 output and terminal state."""

    @cute.jit
    def __call__(
        self,
        causal_qk_scaled: cute.Tensor,
        solved_u: cute.Tensor,
        k_bar: cute.Tensor,
        cumulative_log: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        valid_tokens: cutlass.Int32,
        num_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            causal_qk_scaled,
            solved_u,
            k_bar,
            cumulative_log,
            output,
            final_state,
            valid_tokens,
        ).launch(
            grid=(num_heads, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        causal_qk_scaled: cute.Tensor,
        solved_u: cute.Tensor,
        k_bar: cute.Tensor,
        cumulative_log: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        valid_tokens: cutlass.Int32,
    ) -> None:
        value_channel, _, _ = cute.arch.thread_idx()
        head, _, _ = cute.arch.block_idx()

        for token in cutlass.range_constexpr(CHUNK_SIZE):
            accumulator = cutlass.Float32(0.0)
            if token < valid_tokens:
                for previous in cutlass.range_constexpr(CHUNK_SIZE):
                    accumulator = accumulator + cutlass.Float32(causal_qk_scaled[head, token, previous]) * cutlass.Float32(
                        solved_u[head, previous, value_channel]
                    )
            output[token, head, value_channel] = cutlass.BFloat16(accumulator)

        for key_channel in cutlass.range_constexpr(HEAD_SIZE):
            accumulator = cutlass.Float32(0.0)
            for token in cutlass.range_constexpr(CHUNK_SIZE):
                accumulator = accumulator + cutlass.Float32(k_bar[head, token, key_channel]) * cutlass.Float32(
                    solved_u[head, token, value_channel]
                )
            end_log = cumulative_log[
                head,
                valid_tokens - cutlass.Int32(1),
                key_channel,
            ]
            gamma_end = cute.math.exp2(
                cutlass.Float32(end_log) * cutlass.Float32(_INV_LN2),
                fastmath=True,
            )
            final_state[0, head, value_channel, key_channel] = gamma_end * accumulator

# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Candidate M recurrent output and public [V,K] state publication."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .config import CHUNK_SIZE, HEAD_SIZE, THREADS_PER_CTA, VALUE_SIZE

_INV_LN2 = 1.4426950408889634


class RecurrentOutputAndState:
    """Consume the two triangular solves and one carried public state."""

    @cute.jit
    def __call__(
        self,
        q_bar: cute.Tensor,
        causal_qk_scaled: cute.Tensor,
        solved_u: cute.Tensor,
        solved_y: cute.Tensor,
        k_bar: cute.Tensor,
        cumulative_log: cute.Tensor,
        initial_state: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        valid_tokens: cutlass.Int32,
        num_heads: cutlass.Int32,
        scale: cutlass.Float32,
        initial_state_is_zero: cutlass.Int32,
        store_final_state: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        @cute.struct
        class SharedStorage:
            value_new: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    CHUNK_SIZE * VALUE_SIZE,
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            q_bar,
            causal_qk_scaled,
            solved_u,
            solved_y,
            k_bar,
            cumulative_log,
            initial_state,
            output,
            final_state,
            valid_tokens,
            scale,
            initial_state_is_zero,
            store_final_state,
        ).launch(
            grid=(num_heads, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_bar: cute.Tensor,
        causal_qk_scaled: cute.Tensor,
        solved_u: cute.Tensor,
        solved_y: cute.Tensor,
        k_bar: cute.Tensor,
        cumulative_log: cute.Tensor,
        initial_state: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        valid_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        initial_state_is_zero: cutlass.Int32,
        store_final_state: cutlass.Int32,
    ) -> None:
        value_channel, _, _ = cute.arch.thread_idx()
        head, _, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        value_new = storage.value_new.get_tensor(
            cute.make_layout(
                (CHUNK_SIZE, VALUE_SIZE),
                stride=(VALUE_SIZE, 1),
            ),
        )

        for token in cutlass.range_constexpr(CHUNK_SIZE):
            accumulator = cutlass.Float32(solved_u[head, token, value_channel])
            if initial_state_is_zero == cutlass.Int32(0):
                for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                    accumulator = accumulator - cutlass.Float32(solved_y[head, token, key_channel]) * cutlass.Float32(
                        initial_state[0, head, value_channel, key_channel],
                    )
            value_new[token, value_channel] = accumulator

        cute.arch.sync_threads()

        for token in cutlass.range_constexpr(CHUNK_SIZE):
            accumulator = cutlass.Float32(0.0)
            if token < valid_tokens:
                if initial_state_is_zero == cutlass.Int32(0):
                    for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                        accumulator = (
                            accumulator
                            + cutlass.Float32(q_bar[head, token, key_channel])
                            * cutlass.Float32(
                                initial_state[
                                    0,
                                    head,
                                    value_channel,
                                    key_channel,
                                ],
                            )
                            * scale
                        )
                for previous in cutlass.range_constexpr(CHUNK_SIZE):
                    accumulator = accumulator + cutlass.Float32(
                        causal_qk_scaled[head, token, previous],
                    ) * cutlass.Float32(value_new[previous, value_channel])
            output[token, head, value_channel] = cutlass.BFloat16(accumulator)

        if store_final_state != cutlass.Int32(0):
            for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                accumulator = cutlass.Float32(0.0)
                if initial_state_is_zero == cutlass.Int32(0):
                    accumulator = cutlass.Float32(
                        initial_state[0, head, value_channel, key_channel],
                    )
                for token in cutlass.range_constexpr(CHUNK_SIZE):
                    accumulator = accumulator + cutlass.Float32(k_bar[head, token, key_channel]) * cutlass.Float32(
                        value_new[token, value_channel]
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

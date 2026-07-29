# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Candidate M FP32 unit-lower solve stage."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .config import CHUNK_SIZE, THREADS_PER_CTA, VALUE_SIZE


class UnitLowerSolve:
    """Solve ``L @ U = Z`` with one value channel per thread."""

    @cute.jit
    def __call__(
        self,
        lower: cute.Tensor,
        pseudo_value: cute.Tensor,
        solved_u: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        history_layout = cute.make_layout(
            (CHUNK_SIZE, VALUE_SIZE),
            stride=(VALUE_SIZE, 1),
        )

        @cute.struct
        class SharedStorage:
            history: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(history_layout)],
                128,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            lower,
            pseudo_value,
            solved_u,
            history_layout,
        ).launch(
            grid=(1, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        lower: cute.Tensor,
        pseudo_value: cute.Tensor,
        solved_u: cute.Tensor,
        history_layout: cute.Layout,
    ) -> None:
        value_channel, _, _ = cute.arch.thread_idx()
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        history = storage.history.get_tensor(history_layout)

        for token in cutlass.range_constexpr(CHUNK_SIZE):
            accumulator = cutlass.Float32(pseudo_value[token, value_channel])
            for previous in range(token):
                accumulator = accumulator - cutlass.Float32(lower[token, previous]) * history[previous, value_channel]
            history[token, value_channel] = accumulator
            solved_u[token, value_channel] = accumulator

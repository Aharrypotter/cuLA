# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident packed adapters for the Candidate M component graph."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm

from .config import CHUNK_SIZE, HEAD_SIZE, THREADS_PER_CTA, VALUE_SIZE

_INV_LN2 = 1.4426950408889634


@cute.jit
def _device_fail_closed() -> None:
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class PackedQKTransform:
    """Transform Q/K/g/b in their native query-head space."""

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        g: cute.Tensor,
        b: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cumulative_log: cute.Tensor,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            q,
            k,
            g,
            b,
            cu_seqlens,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            chunk_index,
            num_sequences,
            num_q_heads,
            cutlass.Int32(cute.size(q, mode=[0])),
        ).launch(
            grid=(num_sequences * num_q_heads, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        g: cute.Tensor,
        b: cute.Tensor,
        cu_seqlens: cute.Tensor,
        cumulative_log: cute.Tensor,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
    ) -> None:
        channel, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        sequence = work_index // num_q_heads
        q_head = work_index - sequence * num_q_heads
        sequence_start_i64 = cutlass.Int64(cu_seqlens[sequence])
        sequence_end_i64 = cutlass.Int64(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        if q_head == cutlass.Int32(0):
            if sequence_end_i64 <= sequence_start_i64:
                _device_fail_closed()
            if sequence_start_i64 < cutlass.Int64(0):
                _device_fail_closed()
            if sequence_end_i64 > cutlass.Int64(total_tokens):
                _device_fail_closed()
            if sequence == cutlass.Int32(0):
                if sequence_start_i64 != cutlass.Int64(0):
                    _device_fail_closed()
            if sequence == num_sequences - cutlass.Int32(1):
                if sequence_end_i64 != cutlass.Int64(total_tokens):
                    _device_fail_closed()
        sequence_start = cutlass.Int32(sequence_start_i64)
        sequence_end = cutlass.Int32(sequence_end_i64)
        chunk_start = sequence_start + chunk_index * cutlass.Int32(CHUNK_SIZE)
        prefix = cutlass.Float32(0.0)

        for local_token in cutlass.range_constexpr(CHUNK_SIZE):
            token = chunk_start + cutlass.Int32(local_token)
            if token < sequence_end:
                prefix = prefix + cutlass.Float32(g[token, q_head, channel])
                gamma = cute.math.exp2(
                    prefix * cutlass.Float32(_INV_LN2),
                    fastmath=True,
                )
                gamma_inverse = cute.math.exp2(
                    -prefix * cutlass.Float32(_INV_LN2),
                    fastmath=True,
                )
                q_value = cutlass.Float32(q[token, q_head, channel])
                k_value = cutlass.Float32(k[token, q_head, channel])
                b_value = cutlass.Float32(b[token, q_head, channel])
                cumulative_log[sequence, q_head, local_token, channel] = prefix
                q_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(q_value * gamma)
                k_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(k_value * gamma_inverse)
                e_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(b_value * k_value * gamma)
            else:
                cumulative_log[sequence, q_head, local_token, channel] = cutlass.Float32(0.0)
                q_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(0.0)
                k_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(0.0)
                e_bar[sequence, q_head, local_token, channel] = cutlass.BFloat16(0.0)


class PackedValueTransform:
    """Transform V/w in native value/output-head space."""

    @cute.jit
    def __call__(
        self,
        v: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        pseudo_value: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            v,
            w,
            cu_seqlens,
            pseudo_value,
            chunk_index,
            num_v_heads,
        ).launch(
            grid=(num_sequences * num_v_heads, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        v: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        pseudo_value: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_v_heads: cutlass.Int32,
    ) -> None:
        channel, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        sequence = work_index // num_v_heads
        v_head = work_index - sequence * num_v_heads
        sequence_start = cutlass.Int32(cu_seqlens[sequence])
        sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])
        chunk_start = sequence_start + chunk_index * cutlass.Int32(CHUNK_SIZE)

        for local_token in cutlass.range_constexpr(CHUNK_SIZE):
            token = chunk_start + cutlass.Int32(local_token)
            if token < sequence_end:
                pseudo_value[sequence, v_head, local_token, channel] = cutlass.BFloat16(
                    cutlass.Float32(w[token, v_head, channel]) * cutlass.Float32(v[token, v_head, channel]),
                )
            else:
                pseudo_value[sequence, v_head, local_token, channel] = cutlass.BFloat16(0.0)


class PackedRecurrentOutputAndState:
    """Publish packed output and carry public ``[V,K]`` state on device."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        has_carry_state: bool,
        store_final_state: bool,
    ) -> None:
        self.has_initial_state = has_initial_state
        self.has_carry_state = has_carry_state
        self.store_final_state = store_final_state

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
        carry_state: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        scale: cutlass.Float32,
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
            carry_state,
            output,
            cu_seqlens,
            chunk_index,
            num_q_heads,
            num_v_heads,
            scale,
        ).launch(
            grid=(num_sequences * num_v_heads, 1, 1),
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
        carry_state: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_index: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        scale: cutlass.Float32,
    ) -> None:
        value_channel, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        sequence = work_index // num_v_heads
        value_head = work_index - sequence * num_v_heads
        group_size = num_v_heads // num_q_heads
        q_head = value_head // group_size
        sequence_start = cutlass.Int32(cu_seqlens[sequence])
        sequence_end = cutlass.Int32(cu_seqlens[sequence + cutlass.Int32(1)])
        chunk_start = sequence_start + chunk_index * cutlass.Int32(CHUNK_SIZE)
        remaining = sequence_end - chunk_start
        valid_tokens = remaining
        if valid_tokens > cutlass.Int32(CHUNK_SIZE):
            valid_tokens = cutlass.Int32(CHUNK_SIZE)
        active = valid_tokens > cutlass.Int32(0)

        if active:
            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            value_new = storage.value_new.get_tensor(
                cute.make_layout(
                    (CHUNK_SIZE, VALUE_SIZE),
                    stride=(VALUE_SIZE, 1),
                ),
            )

            for local_token in cutlass.range_constexpr(CHUNK_SIZE):
                accumulator = cutlass.Float32(
                    solved_u[
                        sequence,
                        value_head,
                        local_token,
                        value_channel,
                    ],
                )
                if chunk_index == cutlass.Int32(0):
                    if cutlass.const_expr(self.has_initial_state):
                        for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                            accumulator = accumulator - cutlass.Float32(
                                solved_y[
                                    sequence,
                                    q_head,
                                    local_token,
                                    key_channel,
                                ],
                            ) * cutlass.Float32(
                                initial_state[
                                    sequence,
                                    value_head,
                                    value_channel,
                                    key_channel,
                                ],
                            )
                else:
                    if cutlass.const_expr(self.has_carry_state):
                        for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                            accumulator = accumulator - cutlass.Float32(
                                solved_y[
                                    sequence,
                                    q_head,
                                    local_token,
                                    key_channel,
                                ],
                            ) * cutlass.Float32(
                                carry_state[
                                    sequence,
                                    value_head,
                                    value_channel,
                                    key_channel,
                                ],
                            )
                value_new[local_token, value_channel] = accumulator

            cute.arch.sync_threads()

            for local_token in cutlass.range_constexpr(CHUNK_SIZE):
                if local_token < valid_tokens:
                    accumulator = cutlass.Float32(0.0)
                    if chunk_index == cutlass.Int32(0):
                        if cutlass.const_expr(self.has_initial_state):
                            for key_channel in cutlass.range_constexpr(
                                HEAD_SIZE,
                            ):
                                accumulator = (
                                    accumulator
                                    + cutlass.Float32(
                                        q_bar[
                                            sequence,
                                            q_head,
                                            local_token,
                                            key_channel,
                                        ],
                                    )
                                    * cutlass.Float32(
                                        initial_state[
                                            sequence,
                                            value_head,
                                            value_channel,
                                            key_channel,
                                        ],
                                    )
                                    * scale
                                )
                    else:
                        if cutlass.const_expr(self.has_carry_state):
                            for key_channel in cutlass.range_constexpr(
                                HEAD_SIZE,
                            ):
                                accumulator = (
                                    accumulator
                                    + cutlass.Float32(
                                        q_bar[
                                            sequence,
                                            q_head,
                                            local_token,
                                            key_channel,
                                        ],
                                    )
                                    * cutlass.Float32(
                                        carry_state[
                                            sequence,
                                            value_head,
                                            value_channel,
                                            key_channel,
                                        ],
                                    )
                                    * scale
                                )
                    for previous in cutlass.range_constexpr(CHUNK_SIZE):
                        accumulator = accumulator + cutlass.Float32(
                            causal_qk_scaled[
                                sequence,
                                q_head,
                                local_token,
                                previous,
                            ],
                        ) * cutlass.Float32(
                            value_new[previous, value_channel],
                        )
                    output[
                        chunk_start + cutlass.Int32(local_token),
                        value_head,
                        value_channel,
                    ] = cutlass.BFloat16(accumulator)

            is_last_chunk = chunk_start + valid_tokens >= sequence_end
            should_store = not is_last_chunk
            if cutlass.const_expr(self.store_final_state):
                should_store = True
            if cutlass.const_expr(self.has_carry_state):
                if should_store:
                    for key_channel in cutlass.range_constexpr(HEAD_SIZE):
                        accumulator = cutlass.Float32(0.0)
                        if chunk_index == cutlass.Int32(0):
                            if cutlass.const_expr(self.has_initial_state):
                                accumulator = cutlass.Float32(
                                    initial_state[
                                        sequence,
                                        value_head,
                                        value_channel,
                                        key_channel,
                                    ],
                                )
                        else:
                            accumulator = cutlass.Float32(
                                carry_state[
                                    sequence,
                                    value_head,
                                    value_channel,
                                    key_channel,
                                ],
                            )
                        for local_token in cutlass.range_constexpr(CHUNK_SIZE):
                            accumulator = accumulator + cutlass.Float32(
                                k_bar[
                                    sequence,
                                    q_head,
                                    local_token,
                                    key_channel,
                                ],
                            ) * cutlass.Float32(
                                value_new[local_token, value_channel],
                            )
                        end_log = cumulative_log[
                            sequence,
                            q_head,
                            valid_tokens - cutlass.Int32(1),
                            key_channel,
                        ]
                        gamma_end = cute.math.exp2(
                            cutlass.Float32(end_log) * cutlass.Float32(_INV_LN2),
                            fastmath=True,
                        )
                        carry_state[
                            sequence,
                            value_head,
                            value_channel,
                            key_channel,
                        ] = gamma_end * accumulator

# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""K1-D10-B1 one-stage packed-V128 Producer/State composition.

One 384-thread CTA owns ``(sequence,value_head)``. WG0 synchronously prepares
one full-V128 shared input stage from real packed inputs and one query-owner
capsule. WG1 and WG2 independently keep the low/high natural-V64 FP32 state
slabs resident across all chunks, publish disjoint V64 output halves, and
optionally store disjoint halves of public ``[N,Hv,V,K]`` state.

This diagnostic source validates real-input ownership, three-WG barrier
accounting, packed tails and full-V output assembly. It intentionally excludes
TMA and timing; D10-B2 owns GTMA2 and the selected two-stage rings.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import T

from .config import CHUNK_SIZE, HEAD_SIZE, VALUE_SIZE

_INV_LN2 = 1.4426950408889634
_WARP_GROUP_SIZE = 128
_THREADS_PER_CTA = 384
_WGMMA_K = 16
_KEY_STAGES = HEAD_SIZE // _WGMMA_K
_TOKEN_STAGES = CHUNK_SIZE // _WGMMA_K
_PRODUCER_REGISTER_TARGET = 24
_STATE_REGISTER_TARGET = 240
_STATE_VALUE_TILE = 64


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


@cute.jit
def _convert_c_layout_to_a_layout(
    c_layout: cute.Layout,
    a_value_layout,
):
    """Convert a Hopper C-fragment layout to its matching RS-A layout."""

    return cute.make_layout(
        (
            a_value_layout,
            c_layout.shape[1],
            (
                c_layout.shape[2],
                cute.size(c_layout, mode=[0]) // cute.size(a_value_layout),
            ),
        ),
        stride=(
            c_layout.stride[0],
            c_layout.stride[1],
            (
                c_layout.stride[2],
                cute.size(a_value_layout, mode=[2]) * c_layout.stride[0][2],
            ),
        ),
    )


@cute.jit
def _make_acc_into_op(
    accumulator: cute.Tensor,
    tiled_mma: cute.TiledMma,
) -> cute.Tensor:
    operand = cute.make_rmem_tensor_like(
        _convert_c_layout_to_a_layout(
            accumulator.layout,
            tiled_mma.tv_layout_A.shape[1],
        ),
        cutlass.BFloat16,
    )
    operand_as_accumulator = cute.make_tensor(
        operand.iterator,
        accumulator.layout,
    )
    operand_as_accumulator.store(
        accumulator.load().to(cutlass.BFloat16),
    )
    return operand


@cute.jit
def _fence_f32_register(reg: cutlass.Float32) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [reg.ir_value()],
            "",
            "=f,0",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ),
    )


@cute.jit
def _fence_u32_register(reg: cutlass.Uint32) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [reg.ir_value()],
            "",
            "=r,0",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ),
    )


@cute.jit
def _fence_register_fragment(fragment: cute.Tensor) -> None:
    if cutlass.const_expr(fragment.element_type is cutlass.Float32):
        values = cute.recast_tensor(fragment, cutlass.Float32)
        for item in cutlass.range_constexpr(cute.size(values)):
            values[item] = _fence_f32_register(values[item])
    else:
        values = cute.recast_tensor(fragment, cutlass.Uint32)
        for item in cutlass.range_constexpr(cute.size(values)):
            values[item] = _fence_u32_register(values[item])


@cute.jit
def _wgmma_gemm(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    operand_a: cute.Tensor,
    operand_b: cute.Tensor,
    accumulate: bool,
) -> None:
    for k_block in cutlass.range(
        cute.size(operand_a, mode=[2]),
        unroll_full=True,
    ):
        tiled_mma.set(
            warpgroup.Field.ACCUMULATE,
            accumulate or k_block != 0,
        )
        cute.gemm(
            tiled_mma,
            accumulator,
            operand_a[(None, None, k_block)],
            operand_b[(None, None, k_block)],
            accumulator,
        )


class A2K1D10PackedV128OneStage:
    """D10-B1 384-thread full-V128 CTA with two natural-V64 State WGs."""

    value_tile = 128
    state_value_tile = _STATE_VALUE_TILE
    threads_per_cta = _THREADS_PER_CTA
    min_blocks_per_mp = 1

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        self.has_initial_state = has_initial_state
        self.store_final_state = store_final_state

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        capsule_g: cute.Tensor,
        capsule_aqk: cute.Tensor,
        capsule_akk: cute.Tensor,
        initial_state: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        state_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (self.state_value_tile, HEAD_SIZE, _WGMMA_K),
            warpgroup.OperandSource.RMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        token_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (self.state_value_tile, CHUNK_SIZE, _WGMMA_K),
            warpgroup.OperandSource.RMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        state_mma = cute.make_tiled_mma(
            cute.make_mma_atom(state_op),
            cute.make_layout((1, 1, 1)),
        )
        token_mma = cute.make_tiled_mma(
            cute.make_mma_atom(token_op),
            cute.make_layout((1, 1, 1)),
        )
        key_operand_layout = sm90_utils.make_smem_layout_b(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (self.state_value_tile, CHUNK_SIZE, _WGMMA_K),
            cutlass.BFloat16,
            _KEY_STAGES,
        )
        token_operand_layout = sm90_utils.make_smem_layout_b(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (self.state_value_tile, CHUNK_SIZE, _WGMMA_K),
            cutlass.BFloat16,
            _TOKEN_STAGES,
        )
        state_update_layout = sm90_utils.make_smem_layout_b(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (self.state_value_tile, HEAD_SIZE, _WGMMA_K),
            cutlass.BFloat16,
            _TOKEN_STAGES,
        )

        @cute.struct
        class SharedStorage:
            q_bar: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(key_operand_layout),
                ],
                128,
            ]
            erase_bar: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(key_operand_layout),
                ],
                128,
            ]
            key_tail: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(state_update_layout),
                ],
                128,
            ]
            aqk_scaled: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(token_operand_layout),
                ],
                128,
            ]
            akk_inverse: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(token_operand_layout),
                ],
                128,
            ]
            write_value: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    CHUNK_SIZE * self.value_tile,
                ],
                128,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    CHUNK_SIZE * self.value_tile,
                ],
                128,
            ]
            gamma_end: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    HEAD_SIZE,
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.dynamic_smem_bytes = SharedStorage.size_in_bytes()
        self.kernel(
            q,
            k,
            v,
            b,
            w,
            cu_seqlens,
            capsule_g,
            capsule_aqk,
            capsule_akk,
            initial_state,
            output,
            final_state,
            num_sequences,
            num_q_heads,
            num_v_heads,
            total_tokens,
            scale,
            state_mma,
            token_mma,
            key_operand_layout,
            token_operand_layout,
            state_update_layout,
        ).launch(
            grid=(
                num_sequences * num_v_heads * cutlass.Int32(VALUE_SIZE // self.value_tile),
                1,
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            smem=self.dynamic_smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        capsule_g: cute.Tensor,
        capsule_aqk: cute.Tensor,
        capsule_akk: cute.Tensor,
        initial_state: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        state_mma: cute.TiledMma,
        token_mma: cute.TiledMma,
        key_operand_layout: cute.ComposedLayout,
        token_operand_layout: cute.ComposedLayout,
        state_update_layout: cute.ComposedLayout,
    ) -> None:
        thread, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        warp_group = cute.arch.make_warp_uniform(thread // _WARP_GROUP_SIZE)
        thread_in_group = thread % _WARP_GROUP_SIZE

        value_tiles = cutlass.Int32(VALUE_SIZE // self.value_tile)
        sequence_stride = num_v_heads * value_tiles
        sequence = work_index // sequence_stride
        value_work = work_index - sequence * sequence_stride
        value_head = value_work // value_tiles
        value_tile_index = value_work - value_head * value_tiles
        value_start = value_tile_index * cutlass.Int32(self.value_tile)
        group_size = num_v_heads // num_q_heads
        q_head = value_head // group_size

        sequence_start_i64 = cutlass.Int64(cu_seqlens[sequence])
        sequence_end_i64 = cutlass.Int64(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        if (
            sequence_start_i64 < cutlass.Int64(0)
            or sequence_end_i64 <= sequence_start_i64
            or sequence_end_i64 > cutlass.Int64(total_tokens)
        ):
            _device_fail_closed()
        if sequence == cutlass.Int32(0) and sequence_start_i64 != cutlass.Int64(0):
            _device_fail_closed()
        if sequence == num_sequences - cutlass.Int32(1) and sequence_end_i64 != cutlass.Int64(total_tokens):
            _device_fail_closed()
        sequence_start = cutlass.Int32(sequence_start_i64)
        sequence_end = cutlass.Int32(sequence_end_i64)
        sequence_chunks = (sequence_end - sequence_start + cutlass.Int32(CHUNK_SIZE - 1)) // cutlass.Int32(CHUNK_SIZE)

        flat_chunk_base = cutlass.Int32(0)
        for previous_sequence in cutlass.range(sequence, unroll=0):
            previous_start = cutlass.Int32(cu_seqlens[previous_sequence])
            previous_end = cutlass.Int32(
                cu_seqlens[previous_sequence + cutlass.Int32(1)],
            )
            flat_chunk_base = flat_chunk_base + (
                previous_end - previous_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(CHUNK_SIZE)

        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        shared_q = storage.q_bar.get_tensor(
            key_operand_layout.outer,
            swizzle=key_operand_layout.inner,
        )
        shared_erase = storage.erase_bar.get_tensor(
            key_operand_layout.outer,
            swizzle=key_operand_layout.inner,
        )
        shared_key_tail = storage.key_tail.get_tensor(
            state_update_layout.outer,
            swizzle=state_update_layout.inner,
        )
        shared_aqk = storage.aqk_scaled.get_tensor(
            token_operand_layout.outer,
            swizzle=token_operand_layout.inner,
        )
        shared_akk = storage.akk_inverse.get_tensor(
            token_operand_layout.outer,
            swizzle=token_operand_layout.inner,
        )
        shared_write = storage.write_value.get_tensor(
            cute.make_layout(
                (CHUNK_SIZE, self.value_tile),
                stride=(self.value_tile, 1),
            ),
        )
        shared_output = storage.output.get_tensor(
            cute.make_layout(
                (CHUNK_SIZE, self.value_tile),
                stride=(self.value_tile, 1),
            ),
        )
        shared_gamma_end = storage.gamma_end.get_tensor(
            cute.make_layout((HEAD_SIZE,), stride=(1,)),
        )

        if warp_group == cutlass.Int32(0):
            cute.arch.warpgroup_reg_dealloc(_PRODUCER_REGISTER_TARGET)
            for local_chunk in cutlass.range(sequence_chunks, unroll=1):
                chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                valid_tokens = sequence_end - chunk_start
                if valid_tokens > cutlass.Int32(CHUNK_SIZE):
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                flat_chunk = flat_chunk_base + local_chunk
                key_channel = thread_in_group
                g_end = cutlass.Float32(
                    capsule_g[
                        flat_chunk,
                        q_head,
                        valid_tokens - cutlass.Int32(1),
                        key_channel,
                    ],
                )
                shared_gamma_end[key_channel] = cute.math.exp2(
                    g_end * cutlass.Float32(_INV_LN2),
                    fastmath=True,
                )
                for local_token in cutlass.range(
                    CHUNK_SIZE,
                    unroll=1,
                ):
                    q_value = cutlass.BFloat16(0.0)
                    erase_value = cutlass.BFloat16(0.0)
                    key_value = cutlass.BFloat16(0.0)
                    if cutlass.Int32(local_token) < valid_tokens:
                        token = chunk_start + cutlass.Int32(local_token)
                        g_value = cutlass.Float32(
                            capsule_g[
                                flat_chunk,
                                q_head,
                                local_token,
                                key_channel,
                            ],
                        )
                        gamma = cute.math.exp2(
                            g_value * cutlass.Float32(_INV_LN2),
                            fastmath=True,
                        )
                        tail_gamma = cute.math.exp2(
                            (g_end - g_value) * cutlass.Float32(_INV_LN2),
                            fastmath=True,
                        )
                        raw_k = cutlass.Float32(
                            k[token, q_head, key_channel],
                        )
                        q_value = cutlass.BFloat16(
                            cutlass.Float32(
                                q[token, q_head, key_channel],
                            )
                            * gamma,
                        )
                        erase_value = cutlass.BFloat16(
                            cutlass.Float32(
                                b[token, q_head, key_channel],
                            )
                            * raw_k
                            * gamma,
                        )
                        key_value = cutlass.BFloat16(raw_k * tail_gamma)
                    key_stage = key_channel // cutlass.Int32(_WGMMA_K)
                    key_column = key_channel % cutlass.Int32(_WGMMA_K)
                    shared_q[
                        local_token,
                        key_column,
                        key_stage,
                    ] = q_value
                    shared_erase[
                        local_token,
                        key_column,
                        key_stage,
                    ] = erase_value
                    shared_key_tail[
                        key_channel,
                        local_token % _WGMMA_K,
                        local_token // _WGMMA_K,
                    ] = key_value

                if thread_in_group < cutlass.Int32(self.value_tile):
                    value_index = thread_in_group
                    for local_token in cutlass.range(
                        CHUNK_SIZE,
                        unroll=1,
                    ):
                        write_value = cutlass.BFloat16(0.0)
                        if cutlass.Int32(local_token) < valid_tokens:
                            token = chunk_start + cutlass.Int32(local_token)
                            write_value = cutlass.BFloat16(
                                cutlass.Float32(
                                    w[
                                        token,
                                        value_head,
                                        value_start + value_index,
                                    ],
                                )
                                * cutlass.Float32(
                                    v[
                                        token,
                                        value_head,
                                        value_start + value_index,
                                    ],
                                ),
                            )
                        shared_write[local_token, value_index] = write_value

                for linear in cutlass.range(
                    thread_in_group,
                    CHUNK_SIZE * CHUNK_SIZE,
                    _WARP_GROUP_SIZE,
                    unroll=1,
                ):
                    row = linear // CHUNK_SIZE
                    column = linear % CHUNK_SIZE
                    shared_aqk[
                        row,
                        column % _WGMMA_K,
                        column // _WGMMA_K,
                    ] = capsule_aqk[
                        flat_chunk,
                        q_head,
                        row,
                        column,
                    ]
                    shared_akk[
                        row,
                        column % _WGMMA_K,
                        column // _WGMMA_K,
                    ] = capsule_akk[
                        flat_chunk,
                        q_head,
                        row,
                        column,
                    ]
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                cute.arch.sync_threads()
                for linear in cutlass.range(
                    thread_in_group,
                    CHUNK_SIZE * self.value_tile,
                    _WARP_GROUP_SIZE,
                    unroll=1,
                ):
                    local_token = linear // self.value_tile
                    value_index = linear % self.value_tile
                    if local_token < valid_tokens:
                        output[
                            chunk_start + local_token,
                            value_head,
                            value_start + value_index,
                        ] = shared_output[local_token, value_index]
                cute.arch.sync_threads()
        else:
            cute.arch.warpgroup_reg_alloc(_STATE_REGISTER_TARGET)
            state_slab = warp_group - cutlass.Int32(1)
            shared_value_start = state_slab * cutlass.Int32(self.state_value_tile)
            state_value_start = value_start + shared_value_start

            state_thread = state_mma.get_slice(thread_in_group)
            token_thread = token_mma.get_slice(thread_in_group)
            state_coordinates = state_thread.partition_C(
                cute.make_identity_tensor(
                    (self.state_value_tile, HEAD_SIZE),
                ),
            )
            token_coordinates = token_thread.partition_C(
                cute.make_identity_tensor(
                    (self.state_value_tile, CHUNK_SIZE),
                ),
            )
            state_accumulator = state_thread.make_fragment_C(
                state_thread.partition_shape_C(
                    (self.state_value_tile, HEAD_SIZE),
                ),
            )
            for element in cutlass.range_constexpr(
                cute.size(state_accumulator),
            ):
                value_index, key_index = state_coordinates[element]
                state_value = cutlass.Float32(0.0)
                if cutlass.const_expr(self.has_initial_state):
                    state_value = cutlass.Float32(
                        initial_state[
                            sequence,
                            value_head,
                            state_value_start + value_index,
                            key_index,
                        ],
                    )
                state_accumulator[element] = state_value

            q_operand = token_thread.make_fragment_B(
                token_thread.partition_B(shared_q),
            )
            q_stages = cute.group_modes(q_operand, 2, 4)
            erase_operand = token_thread.make_fragment_B(
                token_thread.partition_B(shared_erase),
            )
            erase_stages = cute.group_modes(erase_operand, 2, 4)
            aqk_operand = token_thread.make_fragment_B(
                token_thread.partition_B(shared_aqk),
            )
            aqk_stages = cute.group_modes(aqk_operand, 2, 4)
            akk_operand = token_thread.make_fragment_B(
                token_thread.partition_B(shared_akk),
            )
            akk_stages = cute.group_modes(akk_operand, 2, 4)
            key_operand = state_thread.make_fragment_B(
                state_thread.partition_B(shared_key_tail),
            )
            key_stages = cute.group_modes(key_operand, 2, 4)

            for local_chunk in cutlass.range(sequence_chunks, unroll=1):
                cute.arch.sync_threads()
                state_as_token_a = _make_acc_into_op(
                    state_accumulator,
                    token_mma,
                )

                output_accumulator = token_thread.make_fragment_C(
                    token_thread.partition_shape_C(
                        (self.state_value_tile, CHUNK_SIZE),
                    ),
                )
                _fence_register_fragment(state_as_token_a)
                _fence_register_fragment(output_accumulator)
                warpgroup.fence()
                _wgmma_gemm(
                    token_mma,
                    output_accumulator,
                    state_as_token_a,
                    q_stages,
                    False,
                )
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                for element in cutlass.range_constexpr(
                    cute.size(output_accumulator),
                ):
                    output_accumulator[element] = output_accumulator[element] * scale

                erase_projection = token_thread.make_fragment_C(
                    token_thread.partition_shape_C(
                        (self.state_value_tile, CHUNK_SIZE),
                    ),
                )
                _fence_register_fragment(state_as_token_a)
                _fence_register_fragment(erase_projection)
                warpgroup.fence()
                _wgmma_gemm(
                    token_mma,
                    erase_projection,
                    state_as_token_a,
                    erase_stages,
                    False,
                )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                for element in cutlass.range_constexpr(
                    cute.size(erase_projection),
                ):
                    value_index, token_index = token_coordinates[element]
                    erase_projection[element] = (
                        cutlass.Float32(
                            shared_write[
                                token_index,
                                shared_value_start + value_index,
                            ],
                        )
                        - erase_projection[element]
                    )

                residual_a = _make_acc_into_op(
                    erase_projection,
                    token_mma,
                )
                value_new = token_thread.make_fragment_C(
                    token_thread.partition_shape_C(
                        (self.state_value_tile, CHUNK_SIZE),
                    ),
                )
                _fence_register_fragment(residual_a)
                _fence_register_fragment(value_new)
                warpgroup.fence()
                _wgmma_gemm(
                    token_mma,
                    value_new,
                    residual_a,
                    akk_stages,
                    False,
                )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                value_new_a = _make_acc_into_op(
                    value_new,
                    token_mma,
                )
                _fence_register_fragment(value_new_a)
                _fence_register_fragment(output_accumulator)
                warpgroup.fence()
                _wgmma_gemm(
                    token_mma,
                    output_accumulator,
                    value_new_a,
                    aqk_stages,
                    True,
                )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                for element in cutlass.range_constexpr(
                    cute.size(output_accumulator),
                ):
                    value_index, token_index = token_coordinates[element]
                    shared_output[
                        token_index,
                        shared_value_start + value_index,
                    ] = cutlass.BFloat16(output_accumulator[element])

                for element in cutlass.range_constexpr(
                    cute.size(state_accumulator),
                ):
                    _, key_index = state_coordinates[element]
                    state_accumulator[element] = state_accumulator[element] * shared_gamma_end[key_index]

                value_new_as_state_a = _make_acc_into_op(
                    value_new,
                    state_mma,
                )
                _fence_register_fragment(value_new_as_state_a)
                _fence_register_fragment(state_accumulator)
                warpgroup.fence()
                _wgmma_gemm(
                    state_mma,
                    state_accumulator,
                    value_new_as_state_a,
                    key_stages,
                    True,
                )
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                cute.arch.sync_threads()

            if cutlass.const_expr(self.store_final_state):
                for element in cutlass.range_constexpr(
                    cute.size(state_accumulator),
                ):
                    value_index, key_index = state_coordinates[element]
                    final_state[
                        sequence,
                        value_head,
                        state_value_start + value_index,
                        key_index,
                    ] = state_accumulator[element]

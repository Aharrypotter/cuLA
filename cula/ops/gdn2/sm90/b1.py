# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Private N6.5-B product-shaped 384-thread GDN2 schedule candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.runtime import from_dlpack

from .config import B1_BACKEND_ID, CHUNK_SIZE, HEAD_SIZE, VALUE_SIZE
from .packed import _device_fail_closed

if TYPE_CHECKING:
    from cula.gdn2.prefill import _GDN2Inputs

_INV_LN2 = 1.4426950408889634
_WARP_GROUP_SIZE = 128
_B1_THREADS = 384
_K_TILE = 16
_INPUT_TILES = HEAD_SIZE // _K_TILE
_TMA_STAGES = 2
_TMA_TRANSACTION_BYTES = 2 * CHUNK_SIZE * _K_TILE * 2
_CONSUMER_SIGNAL_THREADS = _WARP_GROUP_SIZE // 32
_REGISTER_TARGETS = (80, 200, 200)


@dataclass(frozen=True)
class B1ExecutionInfo:
    """Metadata-only receipt for one private Candidate B1 launch."""

    backend_id: str
    launch_count: int
    max_chunks: int
    num_sequences: int
    num_q_heads: int
    num_v_heads: int
    group_size: int
    native_gva: bool
    qkgb_physically_expanded: bool
    combined_threads: int
    register_targets: tuple[int, int, int]
    fallback: bool
    skip: bool


class B1PackedQKTransform:
    """Materialize all query-head chunk transforms in one device launch."""

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
        max_chunks: cutlass.Int32,
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
            max_chunks,
            num_q_heads,
            cutlass.Int32(cute.size(q, mode=[0])),
        ).launch(
            grid=(num_sequences * max_chunks * num_q_heads, 1, 1),
            block=(_WARP_GROUP_SIZE, 1, 1),
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
        max_chunks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
    ) -> None:
        channel, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        q_head = work_index % num_q_heads
        chunk_owner = work_index // num_q_heads
        local_chunk = chunk_owner % max_chunks
        sequence = chunk_owner // max_chunks
        sequence_start_i64 = cutlass.Int64(cu_seqlens[sequence])
        sequence_end_i64 = cutlass.Int64(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        if q_head == cutlass.Int32(0) and local_chunk == cutlass.Int32(0):
            if sequence_end_i64 <= sequence_start_i64:
                _device_fail_closed()
            if sequence_start_i64 < cutlass.Int64(0):
                _device_fail_closed()
            if sequence_end_i64 > cutlass.Int64(total_tokens):
                _device_fail_closed()
            if sequence == cutlass.Int32(0):
                if sequence_start_i64 != cutlass.Int64(0):
                    _device_fail_closed()
        sequence_start = cutlass.Int32(sequence_start_i64)
        sequence_end = cutlass.Int32(sequence_end_i64)
        chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
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
                cumulative_log[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = prefix
                q_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(q_value * gamma)
                k_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(k_value * gamma_inverse)
                e_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(b_value * k_value * gamma)
            else:
                cumulative_log[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.Float32(0.0)
                q_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(0.0)
                k_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(0.0)
                e_bar[
                    sequence,
                    local_chunk,
                    q_head,
                    local_token,
                    channel,
                ] = cutlass.BFloat16(0.0)


class B1PackedValueTransform:
    """Materialize all value-head chunk transforms in one device launch."""

    @cute.jit
    def __call__(
        self,
        v: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        pseudo_value: cute.Tensor,
        max_chunks: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            v,
            w,
            cu_seqlens,
            pseudo_value,
            max_chunks,
            num_v_heads,
        ).launch(
            grid=(num_sequences * max_chunks * num_v_heads, 1, 1),
            block=(_WARP_GROUP_SIZE, 1, 1),
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
        max_chunks: cutlass.Int32,
        num_v_heads: cutlass.Int32,
    ) -> None:
        channel, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        value_head = work_index % num_v_heads
        chunk_owner = work_index // num_v_heads
        local_chunk = chunk_owner % max_chunks
        sequence = chunk_owner // max_chunks
        sequence_start = cutlass.Int32(cu_seqlens[sequence])
        sequence_end = cutlass.Int32(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)

        for local_token in cutlass.range_constexpr(CHUNK_SIZE):
            token = chunk_start + cutlass.Int32(local_token)
            value = cutlass.BFloat16(0.0)
            if token < sequence_end:
                value = cutlass.BFloat16(
                    cutlass.Float32(w[token, value_head, channel]) * cutlass.Float32(v[token, value_head, channel]),
                )
            pseudo_value[
                sequence,
                local_chunk,
                value_head,
                local_token,
                channel,
            ] = value


class B1CombinedIntra:
    """Compute real erase-WGMMA -> Y/U solve and causal QK in one CTA."""

    def __init__(self, group_size: int) -> None:
        if group_size not in {1, 2, 4}:
            raise ValueError(f"unsupported B1 GVA group_size={group_size}")
        self.group_size = group_size

    @cute.jit
    def __call__(
        self,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        pseudo_value: cute.Tensor,
        cu_seqlens: cute.Tensor,
        erase_lower: cute.Tensor,
        causal_qk_scaled: cute.Tensor,
        solved_y: cute.Tensor,
        solved_u: cute.Tensor,
        max_chunks: cutlass.Int32,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        mma_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (CHUNK_SIZE, CHUNK_SIZE, _K_TILE),
            warpgroup.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        tile_shape_mnk = (CHUNK_SIZE, CHUNK_SIZE, _K_TILE)
        a_smem_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            cutlass.BFloat16,
            _TMA_STAGES,
        )
        b_smem_layout = sm90_utils.make_smem_layout_b(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            cutlass.BFloat16,
            _TMA_STAGES,
        )
        output_smem_layout = sm90_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (CHUNK_SIZE, CHUNK_SIZE),
            1,
        )
        num_work = num_sequences * max_chunks * num_q_heads
        operand_layout = cute.make_layout(
            (CHUNK_SIZE, _K_TILE, _INPUT_TILES, num_work),
            stride=(
                HEAD_SIZE,
                1,
                _K_TILE,
                CHUNK_SIZE * HEAD_SIZE,
            ),
        )
        q_operand = cute.make_tensor(q_bar.iterator, operand_layout)
        k_operand = cute.make_tensor(k_bar.iterator, operand_layout)
        e_operand = cute.make_tensor(e_bar.iterator, operand_layout)
        tma_atom_e, tma_tensor_e = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            e_operand,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (CHUNK_SIZE, _K_TILE),
        )
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q_operand,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (CHUNK_SIZE, _K_TILE),
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_operand,
            cute.slice_(b_smem_layout, (None, None, 0)),
            (CHUNK_SIZE, _K_TILE),
        )
        lower_layout = cute.make_layout(
            (CHUNK_SIZE, CHUNK_SIZE),
            stride=(CHUNK_SIZE, 1),
        )
        solve_history_layout = cute.make_layout(
            (CHUNK_SIZE, VALUE_SIZE),
            stride=(VALUE_SIZE, 1),
        )

        @cute.struct
        class SharedStorage:
            pipeline_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _TMA_STAGES,
            ]
            operand_a: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(a_smem_layout),
                ],
                128,
            ]
            operand_b: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(b_smem_layout),
                ],
                128,
            ]
            publication: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(output_smem_layout),
                ],
                128,
            ]
            lower: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(lower_layout),
                ],
                128,
            ]
            solve_history: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(solve_history_layout),
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.dynamic_smem_bytes = SharedStorage.size_in_bytes()
        self.kernel(
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            cu_seqlens,
            erase_lower,
            causal_qk_scaled,
            solved_y,
            solved_u,
            tma_atom_e,
            tma_tensor_e,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
            output_smem_layout,
            lower_layout,
            solve_history_layout,
            max_chunks,
            num_q_heads,
            num_v_heads,
            scale,
        ).launch(
            grid=(num_work, 1, 1),
            block=(_B1_THREADS, 1, 1),
            cluster=(1, 1, 1),
            smem=self.dynamic_smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        pseudo_value: cute.Tensor,
        cu_seqlens: cute.Tensor,
        erase_lower: cute.Tensor,
        causal_qk_scaled: cute.Tensor,
        solved_y: cute.Tensor,
        solved_u: cute.Tensor,
        tma_atom_e: cute.CopyAtom,
        tma_tensor_e: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        output_smem_layout: cute.ComposedLayout,
        lower_layout: cute.Layout,
        solve_history_layout: cute.Layout,
        max_chunks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        scale: cutlass.Float32,
    ) -> None:
        thread_index, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        warp_group_index = cute.arch.make_warp_uniform(
            thread_index // _WARP_GROUP_SIZE,
        )
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_in_group = thread_index % _WARP_GROUP_SIZE
        chunk_owner = work_index // num_q_heads
        q_head = work_index - chunk_owner * num_q_heads
        local_chunk = chunk_owner % max_chunks
        sequence = chunk_owner // max_chunks
        sequence_start = cutlass.Int32(cu_seqlens[sequence])
        sequence_end = cutlass.Int32(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
        remaining = sequence_end - chunk_start
        valid_tokens = remaining
        if valid_tokens > cutlass.Int32(CHUNK_SIZE):
            valid_tokens = cutlass.Int32(CHUNK_SIZE)

        if valid_tokens > cutlass.Int32(0):
            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            shared_a = storage.operand_a.get_tensor(
                a_smem_layout.outer,
                swizzle=a_smem_layout.inner,
            )
            shared_b = storage.operand_b.get_tensor(
                b_smem_layout.outer,
                swizzle=b_smem_layout.inner,
            )
            publication = storage.publication.get_tensor(
                output_smem_layout.outer,
                swizzle=output_smem_layout.inner,
            )
            lower = storage.lower.get_tensor(lower_layout)
            solve_history = storage.solve_history.get_tensor(
                solve_history_layout,
            )
            transfer_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.pipeline_barriers.data_ptr(),
                num_stages=_TMA_STAGES,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    _CONSUMER_SIGNAL_THREADS,
                ),
                tx_count=_TMA_TRANSACTION_BYTES,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            grouped_shared_a = cute.group_modes(shared_a, 0, 2)
            grouped_shared_b = cute.group_modes(shared_b, 0, 2)
            grouped_global_e = cute.group_modes(tma_tensor_e, 0, 2)
            grouped_global_q = cute.group_modes(tma_tensor_q, 0, 2)
            grouped_global_k = cute.group_modes(tma_tensor_k, 0, 2)
            tma_shared_a, tma_global_e = cpasync.tma_partition(
                tma_atom_e,
                0,
                cute.make_layout(1),
                grouped_shared_a,
                grouped_global_e,
            )
            _, tma_global_q = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                grouped_shared_a,
                grouped_global_q,
            )
            tma_shared_b, tma_global_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                grouped_shared_b,
                grouped_global_k,
            )

            if warp_group_index == 0:
                cute.arch.warpgroup_reg_dealloc(_REGISTER_TARGETS[0])
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    _TMA_STAGES,
                )
                if warp_index == 0:
                    for _ in cutlass.range_constexpr(_INPUT_TILES):
                        transfer_pipeline.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_e,
                            tma_global_e[
                                (
                                    None,
                                    producer_state.count,
                                    work_index,
                                )
                            ],
                            tma_shared_a[(None, producer_state.index)],
                            tma_bar_ptr=(
                                transfer_pipeline.producer_get_barrier(
                                    producer_state,
                                )
                            ),
                        )
                        cute.copy(
                            tma_atom_k,
                            tma_global_k[
                                (
                                    None,
                                    producer_state.count,
                                    work_index,
                                )
                            ],
                            tma_shared_b[(None, producer_state.index)],
                            tma_bar_ptr=(
                                transfer_pipeline.producer_get_barrier(
                                    producer_state,
                                )
                            ),
                        )
                        transfer_pipeline.producer_commit(producer_state)
                        producer_state.advance()

                cute.arch.sync_threads()
                for linear_index in cutlass.range(
                    lane_in_group,
                    CHUNK_SIZE * CHUNK_SIZE,
                    _WARP_GROUP_SIZE,
                    unroll=1,
                ):
                    row = linear_index // CHUNK_SIZE
                    column = linear_index % CHUNK_SIZE
                    value = cutlass.Float32(0.0)
                    active = row < valid_tokens and column < valid_tokens
                    if active and row > column:
                        value = cutlass.Float32(publication[row, column, 0])
                    elif active and row == column:
                        value = cutlass.Float32(1.0)
                    lower[row, column] = value
                    erase_lower[
                        sequence,
                        local_chunk,
                        q_head,
                        row,
                        column,
                    ] = value
                cute.arch.sync_threads()

                if warp_index == 0:
                    for _ in cutlass.range_constexpr(_INPUT_TILES):
                        transfer_pipeline.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_q,
                            tma_global_q[
                                (
                                    None,
                                    producer_state.count - _INPUT_TILES,
                                    work_index,
                                )
                            ],
                            tma_shared_a[(None, producer_state.index)],
                            tma_bar_ptr=(
                                transfer_pipeline.producer_get_barrier(
                                    producer_state,
                                )
                            ),
                        )
                        cute.copy(
                            tma_atom_k,
                            tma_global_k[
                                (
                                    None,
                                    producer_state.count - _INPUT_TILES,
                                    work_index,
                                )
                            ],
                            tma_shared_b[(None, producer_state.index)],
                            tma_bar_ptr=(
                                transfer_pipeline.producer_get_barrier(
                                    producer_state,
                                )
                            ),
                        )
                        transfer_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                cute.arch.sync_threads()
                cute.arch.sync_threads()
            elif warp_group_index == 1:
                cute.arch.warpgroup_reg_alloc(_REGISTER_TARGETS[1])
                cute.arch.sync_threads()
                cute.arch.sync_threads()

                value_channel = lane_in_group
                for token in cutlass.range(
                    CHUNK_SIZE,
                    unroll=1,
                ):
                    accumulator = cutlass.Float32(
                        e_bar[
                            sequence,
                            local_chunk,
                            q_head,
                            token,
                            value_channel,
                        ],
                    )
                    for previous in cutlass.range(
                        CHUNK_SIZE,
                        unroll=1,
                    ):
                        if previous < token:
                            accumulator = accumulator - lower[token, previous] * solve_history[previous, value_channel]
                    solve_history[token, value_channel] = accumulator
                    solved_y[
                        sequence,
                        local_chunk,
                        q_head,
                        token,
                        value_channel,
                    ] = accumulator

                first_value_head = q_head * cutlass.Int32(self.group_size)
                for group_offset in cutlass.range_constexpr(
                    self.group_size,
                ):
                    value_head = first_value_head + cutlass.Int32(group_offset)
                    for token in cutlass.range(
                        CHUNK_SIZE,
                        unroll=1,
                    ):
                        accumulator = cutlass.Float32(
                            pseudo_value[
                                sequence,
                                local_chunk,
                                value_head,
                                token,
                                value_channel,
                            ],
                        )
                        for previous in cutlass.range(
                            CHUNK_SIZE,
                            unroll=1,
                        ):
                            if previous < token:
                                accumulator = accumulator - lower[token, previous] * solve_history[previous, value_channel]
                        solve_history[token, value_channel] = accumulator
                        solved_u[
                            sequence,
                            local_chunk,
                            value_head,
                            token,
                            value_channel,
                        ] = accumulator
                cute.arch.sync_threads()
                cute.arch.sync_threads()
            else:
                cute.arch.warpgroup_reg_alloc(_REGISTER_TARGETS[2])
                logical_matrix = cute.make_tensor(
                    causal_qk_scaled.iterator,
                    cute.make_layout(
                        (CHUNK_SIZE, CHUNK_SIZE),
                        stride=(CHUNK_SIZE, 1),
                    ),
                )
                warp_group_mma = tiled_mma.get_slice(0)
                logical_output_fragment = warp_group_mma.partition_C(
                    logical_matrix,
                )
                a_fragment = tiled_mma.make_fragment_A(
                    warp_group_mma.partition_A(shared_a),
                )
                b_fragment = tiled_mma.make_fragment_B(
                    warp_group_mma.partition_B(shared_b),
                )
                consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    _TMA_STAGES,
                )
                accumulator = cute.make_rmem_tensor(
                    logical_output_fragment.shape,
                    cutlass.Float32,
                )
                accumulator.fill(0.0)
                for tile_index in cutlass.range_constexpr(_INPUT_TILES):
                    ready = transfer_pipeline.consumer_try_wait(
                        consumer_state,
                    )
                    transfer_pipeline.consumer_wait(consumer_state, ready)
                    tiled_mma.set(
                        warpgroup.Field.ACCUMULATE,
                        tile_index != 0,
                    )
                    warpgroup.fence()
                    cute.gemm(
                        tiled_mma,
                        accumulator,
                        a_fragment[
                            (
                                None,
                                None,
                                None,
                                consumer_state.index,
                            )
                        ],
                        b_fragment[
                            (
                                None,
                                None,
                                None,
                                consumer_state.index,
                            )
                        ],
                        accumulator,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    transfer_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()
                self._publish_accumulator(
                    lane_in_group,
                    tiled_mma,
                    accumulator,
                    publication,
                )
                cute.arch.sync_threads()
                cute.arch.sync_threads()

                accumulator.fill(0.0)
                for tile_index in cutlass.range_constexpr(_INPUT_TILES):
                    ready = transfer_pipeline.consumer_try_wait(
                        consumer_state,
                    )
                    transfer_pipeline.consumer_wait(consumer_state, ready)
                    tiled_mma.set(
                        warpgroup.Field.ACCUMULATE,
                        tile_index != 0,
                    )
                    warpgroup.fence()
                    cute.gemm(
                        tiled_mma,
                        accumulator,
                        a_fragment[
                            (
                                None,
                                None,
                                None,
                                consumer_state.index,
                            )
                        ],
                        b_fragment[
                            (
                                None,
                                None,
                                None,
                                consumer_state.index,
                            )
                        ],
                        accumulator,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    transfer_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()
                self._publish_accumulator(
                    lane_in_group,
                    tiled_mma,
                    accumulator,
                    publication,
                )
                cute.arch.sync_threads()
                for linear_index in cutlass.range(
                    lane_in_group,
                    CHUNK_SIZE * CHUNK_SIZE,
                    _WARP_GROUP_SIZE,
                    unroll=1,
                ):
                    row = linear_index // CHUNK_SIZE
                    column = linear_index % CHUNK_SIZE
                    value = cutlass.Float32(0.0)
                    if row < valid_tokens and column < valid_tokens and row >= column:
                        value = cutlass.Float32(publication[row, column, 0]) * scale
                    causal_qk_scaled[
                        sequence,
                        local_chunk,
                        q_head,
                        row,
                        column,
                    ] = value
                cute.arch.sync_threads()

    @staticmethod
    @cute.jit
    def _publish_accumulator(
        lane_in_group: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        publication: cute.Tensor,
    ) -> None:
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=cutlass.Float32,
        )
        stmatrix_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4),
            cutlass.BFloat16,
        )
        tiled_stmatrix = cute.make_tiled_copy_C_atom(
            stmatrix_atom,
            tiled_mma,
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_stmatrix,
        )
        thread_copy_r2s = tiled_copy_r2s.get_slice(lane_in_group)
        shared_output_fragment = thread_copy_r2s.partition_D(publication)
        accumulator_for_store = tiled_copy_r2s.retile(accumulator)
        register_shape = cute.shape(
            thread_copy_r2s.partition_S(publication),
        )
        register_layout = cute.make_layout(register_shape[:3])
        store_accumulator = cute.make_rmem_tensor_like(
            register_layout,
            cutlass.Float32,
        )
        for element_index in cutlass.range_constexpr(
            cute.size(store_accumulator),
        ):
            store_accumulator[element_index] = accumulator_for_store[element_index]
        store_output = cute.make_rmem_tensor_like(
            register_layout,
            cutlass.BFloat16,
        )
        store_output.store(
            store_accumulator.load().to(cutlass.BFloat16),
        )
        cute.copy(
            tiled_copy_r2s,
            store_output,
            shared_output_fragment[(None, None, None, 0)],
        )
        cute.arch.fence_proxy("async.shared", space="cta")


class B1PackedRecurrent:
    """Consume every local chunk in sequence order and publish output/state."""

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
        max_chunks: cutlass.Int32,
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
            max_chunks,
            num_q_heads,
            num_v_heads,
            scale,
        ).launch(
            grid=(num_sequences * num_v_heads, 1, 1),
            block=(_WARP_GROUP_SIZE, 1, 1),
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
        max_chunks: cutlass.Int32,
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
        sequence_end = cutlass.Int32(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        sequence_length = sequence_end - sequence_start
        num_chunks = (sequence_length + cutlass.Int32(CHUNK_SIZE - 1)) // cutlass.Int32(CHUNK_SIZE)
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        value_new = storage.value_new.get_tensor(
            cute.make_layout(
                (CHUNK_SIZE, VALUE_SIZE),
                stride=(VALUE_SIZE, 1),
            ),
        )

        for local_chunk in cutlass.range(max_chunks, unroll=1):
            if local_chunk < num_chunks:
                chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                remaining = sequence_end - chunk_start
                valid_tokens = remaining
                if valid_tokens > cutlass.Int32(CHUNK_SIZE):
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)

                for local_token in cutlass.range_constexpr(CHUNK_SIZE):
                    accumulator = cutlass.Float32(
                        solved_u[
                            sequence,
                            local_chunk,
                            value_head,
                            local_token,
                            value_channel,
                        ],
                    )
                    if local_chunk == cutlass.Int32(0):
                        if cutlass.const_expr(self.has_initial_state):
                            for key_channel in cutlass.range(
                                HEAD_SIZE,
                                unroll=1,
                            ):
                                accumulator = accumulator - cutlass.Float32(
                                    solved_y[
                                        sequence,
                                        local_chunk,
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
                        for key_channel in cutlass.range(
                            HEAD_SIZE,
                            unroll=1,
                        ):
                            accumulator = accumulator - cutlass.Float32(
                                solved_y[
                                    sequence,
                                    local_chunk,
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
                        if local_chunk == cutlass.Int32(0):
                            if cutlass.const_expr(self.has_initial_state):
                                for key_channel in cutlass.range(
                                    HEAD_SIZE,
                                    unroll=1,
                                ):
                                    accumulator = (
                                        accumulator
                                        + cutlass.Float32(
                                            q_bar[
                                                sequence,
                                                local_chunk,
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
                            for key_channel in cutlass.range(
                                HEAD_SIZE,
                                unroll=1,
                            ):
                                accumulator = (
                                    accumulator
                                    + cutlass.Float32(
                                        q_bar[
                                            sequence,
                                            local_chunk,
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
                        for previous in cutlass.range(
                            CHUNK_SIZE,
                            unroll=1,
                        ):
                            accumulator = (
                                accumulator
                                + cutlass.Float32(
                                    causal_qk_scaled[
                                        sequence,
                                        local_chunk,
                                        q_head,
                                        local_token,
                                        previous,
                                    ],
                                )
                                * value_new[previous, value_channel]
                            )
                        output[
                            chunk_start + cutlass.Int32(local_token),
                            value_head,
                            value_channel,
                        ] = cutlass.BFloat16(accumulator)

                for key_channel in cutlass.range(
                    HEAD_SIZE,
                    unroll=1,
                ):
                    accumulator = cutlass.Float32(0.0)
                    if local_chunk == cutlass.Int32(0):
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
                    for local_token in cutlass.range(
                        CHUNK_SIZE,
                        unroll=1,
                    ):
                        accumulator = (
                            accumulator
                            + cutlass.Float32(
                                k_bar[
                                    sequence,
                                    local_chunk,
                                    q_head,
                                    local_token,
                                    key_channel,
                                ],
                            )
                            * value_new[local_token, value_channel]
                        )
                    end_log = cumulative_log[
                        sequence,
                        local_chunk,
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
                cute.arch.sync_threads()


_compiled_qk_transform: dict[tuple[int, int, int, int], object] = {}
_compiled_value_transform: dict[tuple[int, int, int, int], object] = {}
_compiled_combined: dict[tuple[int, int, int, int, int], object] = {}
_compiled_recurrent: dict[
    tuple[int, int, int, int, int, bool, bool],
    object,
] = {}


def _device_key(device: torch.device) -> int:
    if device.index is None:
        return torch.cuda.current_device()
    return device.index


def _dynamic_first(tensor: torch.Tensor):
    return from_dlpack(
        tensor,
        assumed_align=16,
    ).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tensor.dim_order(),
    )


def _compile_qk_transform(
    inputs: _GDN2Inputs,
    max_chunks: int,
    cumulative_log: torch.Tensor,
    q_bar: torch.Tensor,
    k_bar: torch.Tensor,
    e_bar: torch.Tensor,
    stream: cuda.CUstream,
):
    key = (
        _device_key(inputs.q.device),
        inputs.num_sequences,
        max_chunks,
        inputs.num_q_heads,
    )
    compiled = _compiled_qk_transform.get(key)
    if compiled is None:
        compiled = cute.compile(
            B1PackedQKTransform(),
            *(
                _dynamic_first(tensor)
                for tensor in (
                    inputs.q,
                    inputs.k,
                    inputs.g,
                    inputs.b,
                    inputs.cu_seqlens,
                    cumulative_log,
                    q_bar,
                    k_bar,
                    e_bar,
                )
            ),
            cutlass.Int32(max_chunks),
            cutlass.Int32(inputs.num_sequences),
            cutlass.Int32(inputs.num_q_heads),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_qk_transform[key] = compiled
    return compiled


def _compile_value_transform(
    inputs: _GDN2Inputs,
    max_chunks: int,
    pseudo_value: torch.Tensor,
    stream: cuda.CUstream,
):
    key = (
        _device_key(inputs.q.device),
        inputs.num_sequences,
        max_chunks,
        inputs.num_v_heads,
    )
    compiled = _compiled_value_transform.get(key)
    if compiled is None:
        compiled = cute.compile(
            B1PackedValueTransform(),
            *(
                _dynamic_first(tensor)
                for tensor in (
                    inputs.v,
                    inputs.w,
                    inputs.cu_seqlens,
                    pseudo_value,
                )
            ),
            cutlass.Int32(max_chunks),
            cutlass.Int32(inputs.num_sequences),
            cutlass.Int32(inputs.num_v_heads),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_value_transform[key] = compiled
    return compiled


def _compile_combined(
    inputs: _GDN2Inputs,
    max_chunks: int,
    q_bar: torch.Tensor,
    k_bar: torch.Tensor,
    e_bar: torch.Tensor,
    pseudo_value: torch.Tensor,
    erase_lower: torch.Tensor,
    causal_qk_scaled: torch.Tensor,
    solved_y: torch.Tensor,
    solved_u: torch.Tensor,
    stream: cuda.CUstream,
):
    group_size = inputs.num_v_heads // inputs.num_q_heads
    key = (
        _device_key(inputs.q.device),
        inputs.num_sequences,
        max_chunks,
        inputs.num_q_heads,
        group_size,
    )
    compiled = _compiled_combined.get(key)
    if compiled is None:
        compiled = cute.compile(
            B1CombinedIntra(group_size),
            *(
                _dynamic_first(tensor)
                for tensor in (
                    q_bar,
                    k_bar,
                    e_bar,
                    pseudo_value,
                    inputs.cu_seqlens,
                    erase_lower,
                    causal_qk_scaled,
                    solved_y,
                    solved_u,
                )
            ),
            cutlass.Int32(max_chunks),
            cutlass.Int32(inputs.num_sequences),
            cutlass.Int32(inputs.num_q_heads),
            cutlass.Int32(inputs.num_v_heads),
            cutlass.Float32(inputs.scale),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_combined[key] = compiled
    return compiled


def _compile_recurrent(
    inputs: _GDN2Inputs,
    max_chunks: int,
    q_bar: torch.Tensor,
    causal_qk_scaled: torch.Tensor,
    solved_u: torch.Tensor,
    solved_y: torch.Tensor,
    k_bar: torch.Tensor,
    cumulative_log: torch.Tensor,
    initial_state: torch.Tensor,
    carry_state: torch.Tensor,
    stream: cuda.CUstream,
):
    key = (
        _device_key(inputs.q.device),
        inputs.num_sequences,
        max_chunks,
        inputs.num_q_heads,
        inputs.num_v_heads,
        inputs.initial_state is not None,
        inputs.output_final_state,
    )
    compiled = _compiled_recurrent.get(key)
    if compiled is None:
        compiled = cute.compile(
            B1PackedRecurrent(
                has_initial_state=inputs.initial_state is not None,
                store_final_state=inputs.output_final_state,
            ),
            *(
                _dynamic_first(tensor)
                for tensor in (
                    q_bar,
                    causal_qk_scaled,
                    solved_u,
                    solved_y,
                    k_bar,
                    cumulative_log,
                    initial_state,
                    carry_state,
                    inputs.output,
                    inputs.cu_seqlens,
                )
            ),
            cutlass.Int32(max_chunks),
            cutlass.Int32(inputs.num_sequences),
            cutlass.Int32(inputs.num_q_heads),
            cutlass.Int32(inputs.num_v_heads),
            cutlass.Float32(inputs.scale),
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _compiled_recurrent[key] = compiled
    return compiled


def launch_b1_gdn2(
    inputs: _GDN2Inputs,
    *,
    return_debug: bool = False,
) -> B1ExecutionInfo | None:
    """Launch private B1 without changing the N6 public dispatch."""

    group_size = inputs.num_v_heads // inputs.num_q_heads
    if group_size not in {1, 2, 4}:
        raise NotImplementedError(
            f"B1 supports GVA group sizes 1, 2 and 4, got {group_size}",
        )
    max_chunks = (inputs.total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    qk_shape = (
        inputs.num_sequences,
        max_chunks,
        inputs.num_q_heads,
        CHUNK_SIZE,
        HEAD_SIZE,
    )
    value_shape = (
        inputs.num_sequences,
        max_chunks,
        inputs.num_v_heads,
        CHUNK_SIZE,
        VALUE_SIZE,
    )
    matrix_shape = (
        inputs.num_sequences,
        max_chunks,
        inputs.num_q_heads,
        CHUNK_SIZE,
        CHUNK_SIZE,
    )
    state_shape = (
        inputs.num_sequences,
        inputs.num_v_heads,
        VALUE_SIZE,
        HEAD_SIZE,
    )
    device = inputs.q.device

    with torch.cuda.device(device):
        cumulative_log = torch.empty(
            qk_shape,
            dtype=torch.float32,
            device=device,
        )
        q_bar = torch.empty(qk_shape, dtype=torch.bfloat16, device=device)
        k_bar = torch.empty_like(q_bar)
        e_bar = torch.empty_like(q_bar)
        pseudo_value = torch.empty(
            value_shape,
            dtype=torch.bfloat16,
            device=device,
        )
        erase_lower = torch.empty(
            matrix_shape,
            dtype=torch.float32,
            device=device,
        )
        causal_qk_scaled = torch.empty_like(erase_lower)
        solved_y = torch.empty(
            qk_shape,
            dtype=torch.float32,
            device=device,
        )
        solved_u = torch.empty(
            value_shape,
            dtype=torch.float32,
            device=device,
        )
        carry_state = (
            inputs.output_state
            if inputs.output_state is not None
            else torch.empty(
                state_shape,
                dtype=torch.float32,
                device=device,
            )
        )
        initial_state = inputs.initial_state if inputs.initial_state is not None else inputs.q
        stream = cuda.CUstream(
            torch.cuda.current_stream(device).cuda_stream,
        )
        qk_transform = _compile_qk_transform(
            inputs,
            max_chunks,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            stream,
        )
        value_transform = _compile_value_transform(
            inputs,
            max_chunks,
            pseudo_value,
            stream,
        )
        combined = _compile_combined(
            inputs,
            max_chunks,
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            erase_lower,
            causal_qk_scaled,
            solved_y,
            solved_u,
            stream,
        )
        recurrent = _compile_recurrent(
            inputs,
            max_chunks,
            q_bar,
            causal_qk_scaled,
            solved_u,
            solved_y,
            k_bar,
            cumulative_log,
            initial_state,
            carry_state,
            stream,
        )

        qk_transform(
            inputs.q,
            inputs.k,
            inputs.g,
            inputs.b,
            inputs.cu_seqlens,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            max_chunks,
            inputs.num_sequences,
            inputs.num_q_heads,
            stream,
        )
        value_transform(
            inputs.v,
            inputs.w,
            inputs.cu_seqlens,
            pseudo_value,
            max_chunks,
            inputs.num_sequences,
            inputs.num_v_heads,
            stream,
        )
        combined(
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            inputs.cu_seqlens,
            erase_lower,
            causal_qk_scaled,
            solved_y,
            solved_u,
            max_chunks,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.scale,
            stream,
        )
        recurrent(
            q_bar,
            causal_qk_scaled,
            solved_u,
            solved_y,
            k_bar,
            cumulative_log,
            initial_state,
            carry_state,
            inputs.output,
            inputs.cu_seqlens,
            max_chunks,
            inputs.num_sequences,
            inputs.num_q_heads,
            inputs.num_v_heads,
            inputs.scale,
            stream,
        )

    if not return_debug:
        return None
    return B1ExecutionInfo(
        backend_id=B1_BACKEND_ID,
        launch_count=4,
        max_chunks=max_chunks,
        num_sequences=inputs.num_sequences,
        num_q_heads=inputs.num_q_heads,
        num_v_heads=inputs.num_v_heads,
        group_size=group_size,
        native_gva=group_size > 1,
        qkgb_physically_expanded=False,
        combined_threads=_B1_THREADS,
        register_targets=_REGISTER_TARGETS,
        fallback=False,
        skip=False,
    )

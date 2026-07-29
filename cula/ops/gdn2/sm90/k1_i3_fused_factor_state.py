# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""I3 two-generation in-CTA factor/state lookahead candidate.

WG1/WG2 retain D12 distributed common transforms and resident FP32 state.
They prepare Q/K/B/G-derived operands for generation ``n + 1`` before
executing recurrence generation ``n``. WG0 computes causal QK, erase, and
collective inverse for ``n + 1`` concurrently with that recurrence. V/W
materialization remains single-buffered. WG0 produces next-generation QKB
before draining current-generation V/W, matching State's prepare-next-first
order without the r22 circular wait.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T

from cula.ops.gdn.sm90.collective_inverse_hmma import CollectiveInverse

from .config import CHUNK_SIZE, HEAD_SIZE, VALUE_SIZE
from .k1_d10_p384_v128_one_stage import (
    _INV_LN2,
    _device_fail_closed,
    _fence_register_fragment,
    _make_acc_into_op,
    _wgmma_gemm,
)

_WARP_GROUP_SIZE = 128
_THREADS_PER_CTA = 384
_WGMMA_K = 16
_RAW_KEY_TILES = HEAD_SIZE // _WGMMA_K
_KEY_STAGES = HEAD_SIZE // _WGMMA_K
_VALUE_TILES = VALUE_SIZE // _WGMMA_K
_TOKEN_STAGES = CHUNK_SIZE // _WGMMA_K
_RAW_STAGES = 2
_VW_PRIVATE_STAGES = 1
_INPUT_STAGES = 2
_FACTOR_WORKSPACE_STAGES = 1
_WRITE_STAGES = 1
_OUTPUT_STAGES = 2
_PRODUCER_SIGNAL_WARPS = _WARP_GROUP_SIZE // 32
_PRODUCER_REGISTER_TARGET = 72
_STATE_REGISTER_TARGET = 216
_STATE_VALUE_TILE = 64
_QKBG_TRANSACTION_BYTES = 3 * CHUNK_SIZE * _WGMMA_K * 2 + CHUNK_SIZE * _WGMMA_K * 4
_VW_TRANSACTION_BYTES = 2 * CHUNK_SIZE * _WGMMA_K * 2
_STATE0_WRITE_BARRIER = 2
_STATE1_WRITE_BARRIER = 3
_STATE_COMMON_BARRIER = 4
_STATE_ITERATION_DONE_BARRIER = 5
_STORE_WG_BARRIER = 1
_INVERSE_BARRIER = 13
_QKB_STREAM_TILES = _RAW_KEY_TILES // 2


@cute.jit
def _post_wgmma_thread_index(
    thread_in_group: cutlass.Int32,
    accumulator_dependency: cutlass.Float32,
) -> cutlass.Int32:
    """Keep mask-coordinate construction data-dependent on WGMMA completion."""

    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                thread_in_group.ir_value(),
                accumulator_dependency.ir_value(),
            ],
            "",
            "=r,0,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ),
    )


@cute.jit
def _publish_factor_qk(
    factor_mma: cute.TiledMma,
    thread_in_group: cutlass.Int32,
    q_bar: cute.Tensor,
    k_bar: cute.Tensor,
    aqk: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
) -> None:
    """Compute exact causal scaled QK and publish BF16 with StMatrix."""

    thread_mma = factor_mma.get_slice(thread_in_group)
    q_fragment = thread_mma.make_fragment_A(
        thread_mma.partition_A(q_bar),
    )
    k_fragment = thread_mma.make_fragment_B(
        thread_mma.partition_B(k_bar),
    )
    accumulator = thread_mma.make_fragment_C(
        thread_mma.partition_shape_C((CHUNK_SIZE, CHUNK_SIZE)),
    )
    accumulator.fill(0.0)
    warpgroup.fence()
    for key_tile in cutlass.range_constexpr(_KEY_STAGES):
        factor_mma.set(warpgroup.Field.ACCUMULATE, key_tile != 0)
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, key_tile)],
            k_fragment[(None, None, key_tile)],
            accumulator,
        )
    warpgroup.commit_group()
    warpgroup.wait_group(0)
    mask_thread_mma = factor_mma.get_slice(
        _post_wgmma_thread_index(
            thread_in_group,
            accumulator[cutlass.Int32(0)],
        ),
    )
    coordinates = mask_thread_mma.partition_C(
        cute.make_identity_tensor((CHUNK_SIZE, CHUNK_SIZE)),
    )
    for element in cutlass.range_constexpr(cute.size(accumulator)):
        row, column = coordinates[element]
        if not (row < valid_tokens and column < valid_tokens and row >= column):
            accumulator[element] = cutlass.Float32(0.0)
        else:
            accumulator[element] = accumulator[element] * scale

    store_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4,
        ),
        cutlass.BFloat16,
    )
    tiled_store = cute.make_tiled_copy_C(store_atom, factor_mma)
    thread_store = tiled_store.get_slice(thread_in_group)
    destination = thread_store.partition_D(aqk)
    source = thread_store.retile(accumulator)
    converted = cute.make_fragment_like(
        source,
        cutlass.BFloat16,
    )
    for element in cutlass.range_constexpr(cute.size(converted)):
        converted[element] = cutlass.BFloat16(source[element])
    cute.copy(tiled_store, converted, destination)
    cute.arch.fence_proxy("async.shared", space="cta")


@cute.jit
def _publish_factor_erase(
    factor_mma: cute.TiledMma,
    thread_in_group: cutlass.Int32,
    erase_bar: cute.Tensor,
    k_bar: cute.Tensor,
    inverse: cute.Tensor,
    valid_tokens: cutlass.Int32,
) -> None:
    """Compute strict-lower erase Gram into the inverse workspace."""

    thread_mma = factor_mma.get_slice(thread_in_group)
    erase_fragment = thread_mma.make_fragment_A(
        thread_mma.partition_A(erase_bar),
    )
    k_fragment = thread_mma.make_fragment_B(
        thread_mma.partition_B(k_bar),
    )
    accumulator = thread_mma.make_fragment_C(
        thread_mma.partition_shape_C((CHUNK_SIZE, CHUNK_SIZE)),
    )
    accumulator.fill(0.0)
    warpgroup.fence()
    for key_tile in cutlass.range_constexpr(_KEY_STAGES):
        factor_mma.set(warpgroup.Field.ACCUMULATE, key_tile != 0)
        cute.gemm(
            factor_mma,
            accumulator,
            erase_fragment[(None, None, key_tile)],
            k_fragment[(None, None, key_tile)],
            accumulator,
        )
    warpgroup.commit_group()
    warpgroup.wait_group(0)
    mask_thread_mma = factor_mma.get_slice(
        _post_wgmma_thread_index(
            thread_in_group,
            accumulator[cutlass.Int32(0)],
        ),
    )
    coordinates = mask_thread_mma.partition_C(
        cute.make_identity_tensor((CHUNK_SIZE, CHUNK_SIZE)),
    )
    for element in cutlass.range_constexpr(cute.size(accumulator)):
        row, column = coordinates[element]
        if not (row < valid_tokens and column < valid_tokens and row > column):
            accumulator[element] = cutlass.Float32(0.0)

    store_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4,
        ),
        cutlass.Float16,
    )
    tiled_store = cute.make_tiled_copy_C(store_atom, factor_mma)
    thread_store = tiled_store.get_slice(thread_in_group)
    destination = thread_store.partition_D(inverse)
    source = thread_store.retile(accumulator)
    converted = cute.make_fragment_like(source, cutlass.Float16)
    for element in cutlass.range_constexpr(cute.size(converted)):
        converted[element] = cutlass.Float16(source[element])
    cute.copy(tiled_store, converted, destination)
    cute.arch.fence_proxy("async.shared", space="cta")


class A2K1I3FusedFactorState:
    """I3-r22r1 producer-pipelined next-QKB-first lookahead."""

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
        factor_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (CHUNK_SIZE, CHUNK_SIZE, _WGMMA_K),
            warpgroup.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        factor_mma = cute.make_tiled_mma(factor_op)
        raw_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (CHUNK_SIZE, CHUNK_SIZE, _WGMMA_K),
            cutlass.BFloat16,
            _RAW_STAGES,
        )
        g_staging_layout = cute.make_layout(
            (CHUNK_SIZE, _WGMMA_K, _RAW_STAGES),
            stride=(
                _WGMMA_K,
                1,
                CHUNK_SIZE * _WGMMA_K,
            ),
        )
        operand_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW32,
            cutlass.BFloat16,
        )
        key_operand_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, HEAD_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        factor_workspace_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, HEAD_SIZE, _FACTOR_WORKSPACE_STAGES),
            (0, 1, 2),
        )
        token_operand_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        factor_inverse_layout = cute.make_layout(
            (CHUNK_SIZE, CHUNK_SIZE, _FACTOR_WORKSPACE_STAGES),
            stride=(
                CHUNK_SIZE,
                1,
                CHUNK_SIZE * HEAD_SIZE,
            ),
        )
        state_update_layout = cute.tile_to_shape(
            operand_layout_atom,
            (HEAD_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        write_layout = cute.make_layout(
            (CHUNK_SIZE, self.value_tile, _WRITE_STAGES),
            stride=(
                self.value_tile,
                1,
                CHUNK_SIZE * self.value_tile,
            ),
        )
        gamma_layout = cute.make_layout(
            (HEAD_SIZE, _INPUT_STAGES),
            stride=(1, HEAD_SIZE),
        )
        output_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW64,
            cutlass.BFloat16,
        )
        output_layout = cute.tile_to_shape(
            output_layout_atom,
            (CHUNK_SIZE, self.value_tile, _OUTPUT_STAGES),
            (0, 1, 2),
        )

        q_heads = cute.size(q, mode=[1])
        v_heads = cute.size(v, mode=[1])
        token_extent = cute.size(q, mode=[0])
        raw_q_layout = cute.make_layout(
            (token_extent, HEAD_SIZE, q_heads),
            stride=(q_heads * HEAD_SIZE, 1, HEAD_SIZE),
        )
        raw_v_layout = cute.make_layout(
            (token_extent, VALUE_SIZE, v_heads),
            stride=(v_heads * VALUE_SIZE, 1, VALUE_SIZE),
        )
        q_global = cute.make_tensor(q.iterator, raw_q_layout)
        k_global = cute.make_tensor(k.iterator, raw_q_layout)
        b_global = cute.make_tensor(b.iterator, raw_q_layout)
        v_global = cute.make_tensor(v.iterator, raw_v_layout)
        w_global = cute.make_tensor(w.iterator, raw_v_layout)

        capsule_chunks = cute.size(capsule_g, mode=[0])
        capsule_heads = cute.size(capsule_g, mode=[1])
        capsule_g_layout = cute.make_layout(
            (
                CHUNK_SIZE,
                HEAD_SIZE,
                capsule_chunks,
                capsule_heads,
            ),
            stride=(
                HEAD_SIZE,
                1,
                capsule_heads * CHUNK_SIZE * HEAD_SIZE,
                CHUNK_SIZE * HEAD_SIZE,
            ),
        )
        capsule_g_global = cute.make_tensor(
            capsule_g.iterator,
            capsule_g_layout,
        )

        output_global_layout = cute.make_layout(
            (token_extent, VALUE_SIZE, v_heads),
            stride=(v_heads * VALUE_SIZE, 1, VALUE_SIZE),
        )
        tma_output_global = cute.make_tensor(
            output.iterator,
            output_global_layout,
        )

        q_atom, q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        k_atom, k_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        b_atom, b_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            b_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        v_atom, v_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            v_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        w_atom, w_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        g_atom, g_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            capsule_g_global,
            cute.slice_(g_staging_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        output_atom, output_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tma_output_global,
            cute.slice_(output_layout, (None, None, 0)),
            (CHUNK_SIZE, self.value_tile),
        )

        @cute.struct
        class SharedStorage:
            qkb0_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            qkb1_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            vw0_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _VW_PRIVATE_STAGES,
            ]
            vw1_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _VW_PRIVATE_STAGES,
            ]
            raw_handoff_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            factor_ready_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _INPUT_STAGES,
            ]
            factor_done_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _INPUT_STAGES,
            ]
            output_handoff_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _OUTPUT_STAGES,
            ]
            producer_value_work_by_warp: cute.struct.MemRange[
                cutlass.Int32,
                _PRODUCER_SIGNAL_WARPS,
            ]
            raw_q: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_k: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_b: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_g: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(g_staging_layout),
                ],
                128,
            ]
            raw_v: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_w: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
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
                    cute.cosize(write_layout),
                ],
                128,
            ]
            factor_workspace: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(factor_workspace_layout),
                ],
                128,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(output_layout),
                ],
                128,
            ]
            gamma_end: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(gamma_layout),
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.dynamic_smem_bytes = SharedStorage.size_in_bytes()
        self.kernel(
            q_atom,
            q_tma,
            k_atom,
            k_tma,
            b_atom,
            b_tma,
            g_atom,
            g_tma,
            v_atom,
            v_tma,
            w_atom,
            w_tma,
            output_atom,
            output_tma,
            output,
            cu_seqlens,
            capsule_g,
            initial_state,
            final_state,
            num_sequences,
            num_q_heads,
            num_v_heads,
            total_tokens,
            scale,
            state_mma,
            token_mma,
            factor_mma,
            raw_layout,
            g_staging_layout,
            key_operand_layout,
            factor_workspace_layout,
            token_operand_layout,
            factor_inverse_layout,
            state_update_layout,
            write_layout,
            gamma_layout,
            output_layout,
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
        q_atom: cute.CopyAtom,
        q_tma: cute.Tensor,
        k_atom: cute.CopyAtom,
        k_tma: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tma: cute.Tensor,
        g_atom: cute.CopyAtom,
        g_tma: cute.Tensor,
        v_atom: cute.CopyAtom,
        v_tma: cute.Tensor,
        w_atom: cute.CopyAtom,
        w_tma: cute.Tensor,
        output_atom: cute.CopyAtom,
        output_tma: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        capsule_g: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        state_mma: cute.TiledMma,
        token_mma: cute.TiledMma,
        factor_mma: cute.TiledMma,
        raw_layout: cute.ComposedLayout,
        g_staging_layout: cute.Layout,
        key_operand_layout: cute.ComposedLayout,
        factor_workspace_layout: cute.ComposedLayout,
        token_operand_layout: cute.ComposedLayout,
        factor_inverse_layout: cute.Layout,
        state_update_layout: cute.ComposedLayout,
        write_layout: cute.Layout,
        gamma_layout: cute.Layout,
        output_layout: cute.ComposedLayout,
    ) -> None:
        thread, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        warp_group = cute.arch.make_warp_uniform(
            thread // _WARP_GROUP_SIZE,
        )
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())
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
        flat_chunk_base = cutlass.Int32(0)
        for previous_sequence in cutlass.range(sequence, unroll=0):
            previous_start = cutlass.Int32(
                cu_seqlens[previous_sequence],
            )
            previous_end = cutlass.Int32(
                cu_seqlens[previous_sequence + cutlass.Int32(1)],
            )
            flat_chunk_base = flat_chunk_base + (
                previous_end - previous_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(CHUNK_SIZE)

        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        raw_q = storage.raw_q.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_k = storage.raw_k.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_b = storage.raw_b.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_g = storage.raw_g.get_tensor(g_staging_layout)
        raw_v = storage.raw_v.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_w = storage.raw_w.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
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
        shared_write = storage.write_value.get_tensor(write_layout)
        shared_output = storage.output.get_tensor(
            output_layout.outer,
            swizzle=output_layout.inner,
        )
        shared_factor_k = storage.factor_workspace.get_tensor(
            factor_workspace_layout.outer,
            swizzle=factor_workspace_layout.inner,
        )
        factor_workspace_address = storage.factor_workspace.data_ptr().toint()
        shared_inverse = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float16,
                factor_workspace_address,
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            factor_inverse_layout,
        )
        shared_gamma_end = storage.gamma_end.get_tensor(gamma_layout)
        producer_value_work_by_warp = storage.producer_value_work_by_warp.get_tensor(
            cute.make_layout(
                (_PRODUCER_SIGNAL_WARPS,),
                stride=(1,),
            ),
        )
        qkb0_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.qkb0_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_QKBG_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        qkb1_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.qkb1_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_QKBG_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        vw0_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.vw0_barriers.data_ptr(),
            num_stages=_VW_PRIVATE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_VW_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        vw1_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.vw1_barriers.data_ptr(),
            num_stages=_VW_PRIVATE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_VW_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        raw_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.raw_handoff_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
        )
        factor_ready_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.factor_ready_barriers.data_ptr(),
            num_stages=_INPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )
        factor_done_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.factor_done_barriers.data_ptr(),
            num_stages=_INPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
        )
        output_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.output_handoff_barriers.data_ptr(),
            num_stages=_OUTPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=_OUTPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )

        if warp_index == cutlass.Int32(0):
            cpasync.prefetch_descriptor(q_atom)
            cpasync.prefetch_descriptor(k_atom)
            cpasync.prefetch_descriptor(b_atom)
            cpasync.prefetch_descriptor(g_atom)
            cpasync.prefetch_descriptor(v_atom)
            cpasync.prefetch_descriptor(w_atom)
            cpasync.prefetch_descriptor(output_atom)

        if warp_group == cutlass.Int32(0):
            cute.arch.warpgroup_reg_dealloc(
                _PRODUCER_REGISTER_TARGET,
            )
            producer_work_index, _, _ = cute.arch.block_idx()
            producer_sequence = producer_work_index // sequence_stride
            producer_value_work = producer_work_index - producer_sequence * sequence_stride
            if thread_in_group % cutlass.Int32(32) == cutlass.Int32(0):
                producer_value_work_by_warp[warp_index] = producer_value_work
            cute.arch.sync_warp()
            producer_sequence_start = cutlass.Int32(
                cu_seqlens[producer_sequence],
            )
            producer_sequence_end = cutlass.Int32(
                cu_seqlens[producer_sequence + cutlass.Int32(1)],
            )
            producer_sequence_chunks = (
                producer_sequence_end - producer_sequence_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(
                CHUNK_SIZE,
            )
            output_wait = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _OUTPUT_STAGES,
            )
            output_release = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _OUTPUT_STAGES,
            )
            for pipeline_step in cutlass.range(
                producer_sequence_chunks + cutlass.Int32(1),
                unroll=1,
            ):
                factor_stage = cutlass.Int32(0)
                factor_valid_tokens = cutlass.Int32(0)
                if pipeline_step < producer_sequence_chunks:
                    qkb0_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        1,
                    )
                    qkb1_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        1,
                    )
                    local_chunk = pipeline_step
                    chunk_start = producer_sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        valid_tokens = producer_sequence_end - chunk_start
                    flat_chunk = flat_chunk_base + local_chunk

                    factor_stage = pipeline_step % cutlass.Int32(_INPUT_STAGES)
                    factor_valid_tokens = valid_tokens
                    raw_handoff.producer_acquire(
                        pipeline.PipelineState(
                            1,
                            pipeline_step,
                            cutlass.Int32(0),
                            cutlass.Int32(1) - pipeline_step % cutlass.Int32(2),
                        ),
                    )

                    q_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        q_tma,
                    )
                    k_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        k_tma,
                    )
                    b_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        b_tma,
                    )
                    q_tiles = cute.local_tile(
                        q_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    k_tiles = cute.local_tile(
                        k_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    b_tiles = cute.local_tile(
                        b_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    g_tiles = cute.local_tile(
                        g_tma[None, None, flat_chunk, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    q_smem, q_gmem = cpasync.tma_partition(
                        q_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_q, 0, 2),
                        cute.group_modes(q_tiles, 0, 2),
                    )
                    k_smem, k_gmem = cpasync.tma_partition(
                        k_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_k, 0, 2),
                        cute.group_modes(k_tiles, 0, 2),
                    )
                    b_smem, b_gmem = cpasync.tma_partition(
                        b_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_b, 0, 2),
                        cute.group_modes(b_tiles, 0, 2),
                    )
                    g_smem, g_gmem = cpasync.tma_partition(
                        g_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_g, 0, 2),
                        cute.group_modes(g_tiles, 0, 2),
                    )

                    if warp_index == cutlass.Int32(0):
                        qkb0_pipeline.producer_acquire(qkb0_producer)
                        qkb0_barrier = qkb0_pipeline.producer_get_barrier(
                            qkb0_producer,
                        )
                        cute.copy(
                            q_atom,
                            q_gmem[(None, 0, cutlass.Int32(0))],
                            q_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            k_atom,
                            k_gmem[(None, 0, cutlass.Int32(0))],
                            k_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            b_atom,
                            b_gmem[(None, 0, cutlass.Int32(0))],
                            b_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            g_atom,
                            g_gmem[(None, 0, cutlass.Int32(0))],
                            g_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        qkb0_pipeline.producer_commit(qkb0_producer)
                        qkb0_producer.advance()

                        qkb1_pipeline.producer_acquire(qkb1_producer)
                        qkb1_barrier = qkb1_pipeline.producer_get_barrier(
                            qkb1_producer,
                        )
                        cute.copy(
                            q_atom,
                            q_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            q_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            k_atom,
                            k_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            k_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            b_atom,
                            b_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            b_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            g_atom,
                            g_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            g_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        qkb1_pipeline.producer_commit(qkb1_producer)
                        qkb1_producer.advance()

                    # State may begin consuming Q/K/B/G after each private
                    # stream has its first tile in flight. V/W is deliberately
                    # issued only after this generation's factor is complete,
                    # so its single shared write buffer cannot obstruct the
                    # next factor generation.
                    raw_handoff.producer_commit(
                        pipeline.PipelineState(
                            1,
                            pipeline_step,
                            cutlass.Int32(0),
                            cutlass.Int32(1) - pipeline_step % cutlass.Int32(2),
                        ),
                    )

                    for local_key_tile in cutlass.range(
                        1,
                        _QKB_STREAM_TILES,
                        unroll=1,
                    ):
                        if warp_index == cutlass.Int32(0):
                            qkb0_pipeline.producer_acquire(qkb0_producer)
                            qkb0_barrier = qkb0_pipeline.producer_get_barrier(
                                qkb0_producer,
                            )
                            cute.copy(
                                q_atom,
                                q_gmem[(None, 0, local_key_tile)],
                                q_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                k_atom,
                                k_gmem[(None, 0, local_key_tile)],
                                k_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                b_atom,
                                b_gmem[(None, 0, local_key_tile)],
                                b_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                g_atom,
                                g_gmem[(None, 0, local_key_tile)],
                                g_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            qkb0_pipeline.producer_commit(qkb0_producer)
                            qkb0_producer.advance()

                            high_key_tile = local_key_tile + cutlass.Int32(_QKB_STREAM_TILES)
                            qkb1_pipeline.producer_acquire(qkb1_producer)
                            qkb1_barrier = qkb1_pipeline.producer_get_barrier(
                                qkb1_producer,
                            )
                            cute.copy(
                                q_atom,
                                q_gmem[(None, 0, high_key_tile)],
                                q_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                k_atom,
                                k_gmem[(None, 0, high_key_tile)],
                                k_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                b_atom,
                                b_gmem[(None, 0, high_key_tile)],
                                b_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                g_atom,
                                g_gmem[(None, 0, high_key_tile)],
                                g_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            qkb1_pipeline.producer_commit(qkb1_producer)
                            qkb1_producer.advance()

                if pipeline_step > cutlass.Int32(0):
                    # Produce next-generation QKB above before draining the
                    # current V/W generation. This matches State's
                    # prepare-next-before-materialize-current order and
                    # removes the single-stage V/W/QKB circular wait in r22.
                    vw_chunk = pipeline_step - cutlass.Int32(1)
                    vw0_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        _VW_PRIVATE_STAGES,
                    )
                    vw1_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        _VW_PRIVATE_STAGES,
                    )
                    vw_chunk_start = producer_sequence_start + vw_chunk * cutlass.Int32(CHUNK_SIZE)
                    v_use = cute.domain_offset(
                        (
                            vw_chunk_start,
                            value_start,
                            cutlass.Int32(0),
                        ),
                        v_tma,
                    )
                    w_use = cute.domain_offset(
                        (
                            vw_chunk_start,
                            value_start,
                            cutlass.Int32(0),
                        ),
                        w_tma,
                    )
                    v_tiles = cute.local_tile(
                        v_use[None, None, value_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    w_tiles = cute.local_tile(
                        w_use[None, None, value_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    v_smem, v_gmem = cpasync.tma_partition(
                        v_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_v, 0, 2),
                        cute.group_modes(v_tiles, 0, 2),
                    )
                    w_smem, w_gmem = cpasync.tma_partition(
                        w_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_w, 0, 2),
                        cute.group_modes(w_tiles, 0, 2),
                    )
                    for local_value_tile in cutlass.range(
                        _VALUE_TILES // 2,
                        unroll=1,
                    ):
                        if warp_index == cutlass.Int32(0):
                            vw0_pipeline.producer_acquire(vw0_producer)
                            vw0_barrier = vw0_pipeline.producer_get_barrier(
                                vw0_producer,
                            )
                            cute.copy(
                                v_atom,
                                v_gmem[(None, 0, local_value_tile)],
                                v_smem[(None, 0)],
                                tma_bar_ptr=vw0_barrier,
                            )
                            cute.copy(
                                w_atom,
                                w_gmem[(None, 0, local_value_tile)],
                                w_smem[(None, 0)],
                                tma_bar_ptr=vw0_barrier,
                            )
                            vw0_pipeline.producer_commit(vw0_producer)
                            vw0_producer.advance()

                            high_value_tile = local_value_tile + cutlass.Int32(_VALUE_TILES // 2)
                            vw1_pipeline.producer_acquire(vw1_producer)
                            vw1_barrier = vw1_pipeline.producer_get_barrier(
                                vw1_producer,
                            )
                            cute.copy(
                                v_atom,
                                v_gmem[(None, 0, high_value_tile)],
                                v_smem[(None, 1)],
                                tma_bar_ptr=vw1_barrier,
                            )
                            cute.copy(
                                w_atom,
                                w_gmem[(None, 0, high_value_tile)],
                                w_smem[(None, 1)],
                                tma_bar_ptr=vw1_barrier,
                            )
                            vw1_pipeline.producer_commit(vw1_producer)
                            vw1_producer.advance()

                if pipeline_step < producer_sequence_chunks:
                    factor_consumer_state = pipeline.PipelineState(
                        _INPUT_STAGES,
                        pipeline_step,
                        pipeline_step % cutlass.Int32(_INPUT_STAGES),
                        (pipeline_step // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2),
                    )
                    factor_producer_state = pipeline.PipelineState(
                        _INPUT_STAGES,
                        pipeline_step,
                        pipeline_step % cutlass.Int32(_INPUT_STAGES),
                        cutlass.Int32(1) - ((pipeline_step // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                    )
                    factor_ready_handoff.consumer_wait(
                        factor_consumer_state,
                    )
                    factor_done_handoff.producer_acquire(
                        factor_producer_state,
                    )

                    factor_q = shared_q[None, None, factor_stage]
                    factor_erase = shared_erase[None, None, factor_stage]
                    factor_k = shared_factor_k[
                        None,
                        None,
                        cutlass.Int32(0),
                    ]
                    factor_aqk = shared_aqk[None, None, factor_stage]
                    factor_akk = shared_akk[None, None, factor_stage]
                    factor_inverse = shared_inverse[
                        None,
                        None,
                        cutlass.Int32(0),
                    ]
                    _publish_factor_qk(
                        factor_mma,
                        thread_in_group,
                        factor_q,
                        factor_k,
                        factor_aqk,
                        factor_valid_tokens,
                        scale,
                    )
                    _publish_factor_erase(
                        factor_mma,
                        thread_in_group,
                        factor_erase,
                        factor_k,
                        factor_inverse,
                        factor_valid_tokens,
                    )
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    CollectiveInverse().run(
                        factor_inverse,
                        _INVERSE_BARRIER,
                    )
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    for linear in cutlass.range(
                        thread_in_group,
                        CHUNK_SIZE * CHUNK_SIZE,
                        _WARP_GROUP_SIZE,
                        unroll=1,
                    ):
                        row = linear // CHUNK_SIZE
                        column = linear % CHUNK_SIZE
                        factor_akk[row, column] = cutlass.BFloat16(
                            factor_inverse[row, column],
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    factor_done_handoff.producer_commit(
                        factor_producer_state,
                    )
                    factor_ready_handoff.consumer_release(
                        factor_consumer_state,
                    )
                if pipeline_step > cutlass.Int32(0):
                    local_chunk = pipeline_step - cutlass.Int32(1)
                    chunk_start = producer_sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        valid_tokens = producer_sequence_end - chunk_start

                    output_handoff.consumer_wait(output_wait)
                    if valid_tokens == cutlass.Int32(CHUNK_SIZE):
                        output_view = cute.domain_offset(
                            (
                                chunk_start,
                                value_start,
                                cutlass.Int32(0),
                            ),
                            output_tma,
                        )
                        output_tile = cute.zipped_divide(
                            output_view[None, None, value_head],
                            (CHUNK_SIZE, self.value_tile),
                        )[
                            (
                                (None, None),
                                (cutlass.Int32(0), cutlass.Int32(0)),
                            )
                        ]
                        output_stage = shared_output[
                            None,
                            None,
                            output_wait.index,
                        ]
                        output_smem, output_gmem = cpasync.tma_partition(
                            output_atom,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(output_stage, 0, 2),
                            cute.group_modes(output_tile, 0, 2),
                        )
                        if warp_index == cutlass.Int32(0):
                            cute.arch.fence_view_async_shared()
                            cute.copy(
                                output_atom,
                                output_smem,
                                output_gmem,
                            )
                            store_pipeline.producer_commit()
                            if local_chunk > cutlass.Int32(0):
                                store_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        if local_chunk > cutlass.Int32(0):
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                        if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                            if warp_index == cutlass.Int32(0):
                                store_pipeline.producer_tail()
                            cute.arch.barrier(
                                barrier_id=_STORE_WG_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                    else:
                        if warp_index == cutlass.Int32(0):
                            store_pipeline.producer_tail()
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        if local_chunk > cutlass.Int32(0):
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                        tail_value_work = producer_value_work_by_warp[warp_index]
                        tail_value_head = tail_value_work // value_tiles
                        tail_value_tile_index = tail_value_work - tail_value_head * value_tiles
                        tail_value_start = tail_value_tile_index * cutlass.Int32(self.value_tile)
                        for linear in cutlass.range(
                            thread_in_group,
                            valid_tokens * cutlass.Int32(self.value_tile),
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // self.value_tile
                            value_index = linear % self.value_tile
                            output[
                                chunk_start + local_token,
                                tail_value_head,
                                tail_value_start + value_index,
                            ] = shared_output[
                                local_token,
                                value_index,
                                output_wait.index,
                            ]
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        output_handoff.consumer_release(output_release)
                        output_release.advance()
                    output_wait.advance()

        else:
            cute.arch.warpgroup_reg_alloc(_STATE_REGISTER_TARGET)
            state_sequence_start = cutlass.Int32(cu_seqlens[sequence])
            state_sequence_end = cutlass.Int32(
                cu_seqlens[sequence + cutlass.Int32(1)],
            )
            state_sequence_chunks = (
                state_sequence_end - state_sequence_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(
                CHUNK_SIZE,
            )
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

            output_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _OUTPUT_STAGES,
            )
            for state_step in cutlass.range(
                state_sequence_chunks + cutlass.Int32(1),
                unroll=1,
            ):
                current_stage = cutlass.Int32(0)
                current_valid_tokens = cutlass.Int32(0)
                if state_step > cutlass.Int32(0):
                    current_chunk = state_step - cutlass.Int32(1)
                    current_stage = current_chunk % cutlass.Int32(_INPUT_STAGES)
                    current_chunk_start = state_sequence_start + current_chunk * cutlass.Int32(CHUNK_SIZE)
                    current_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if current_chunk + cutlass.Int32(1) == state_sequence_chunks:
                        current_valid_tokens = state_sequence_end - current_chunk_start

                    # The next factor may not reuse this input stage until the
                    # current factor has completed.
                    factor_done_consumer_state = pipeline.PipelineState(
                        _INPUT_STAGES,
                        current_chunk,
                        current_chunk % cutlass.Int32(_INPUT_STAGES),
                        (current_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2),
                    )
                    factor_done_handoff.consumer_wait(
                        factor_done_consumer_state,
                    )
                    factor_done_handoff.consumer_release(
                        factor_done_consumer_state,
                    )

                if state_step < state_sequence_chunks:
                    prepare_chunk = state_step
                    prepare_stage = prepare_chunk % cutlass.Int32(_INPUT_STAGES)
                    prepare_chunk_start = state_sequence_start + prepare_chunk * cutlass.Int32(CHUNK_SIZE)
                    prepare_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if prepare_chunk + cutlass.Int32(1) == state_sequence_chunks:
                        prepare_valid_tokens = state_sequence_end - prepare_chunk_start

                    qkb_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        1,
                    )
                    raw_handoff.consumer_wait(
                        pipeline.PipelineState(
                            1,
                            prepare_chunk,
                            cutlass.Int32(0),
                            prepare_chunk % cutlass.Int32(2),
                        ),
                    )
                    factor_ready_handoff.producer_acquire(
                        pipeline.PipelineState(
                            _INPUT_STAGES,
                            prepare_chunk,
                            (prepare_chunk % cutlass.Int32(_INPUT_STAGES)),
                            cutlass.Int32(1) - ((prepare_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                        ),
                    )

                    for local_key_tile in cutlass.range(
                        _QKB_STREAM_TILES,
                        unroll=1,
                    ):
                        if state_slab == cutlass.Int32(0):
                            qkb_ready = qkb0_pipeline.consumer_try_wait(
                                qkb_consumer,
                            )
                            qkb0_pipeline.consumer_wait(
                                qkb_consumer,
                                qkb_ready,
                            )
                        else:
                            qkb_ready = qkb1_pipeline.consumer_try_wait(
                                qkb_consumer,
                            )
                            qkb1_pipeline.consumer_wait(
                                qkb_consumer,
                                qkb_ready,
                            )

                        raw_stage = state_slab
                        key_tile = state_slab * cutlass.Int32(_QKB_STREAM_TILES) + local_key_tile
                        for linear in cutlass.range(
                            thread_in_group,
                            CHUNK_SIZE * _WGMMA_K,
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // _WGMMA_K
                            tile_channel = linear % _WGMMA_K
                            key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                            q_value = cutlass.BFloat16(0.0)
                            erase_value = cutlass.BFloat16(0.0)
                            if local_token < prepare_valid_tokens:
                                g_value = cutlass.Float32(
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                gamma = cute.math.exp2(
                                    g_value * cutlass.Float32(_INV_LN2),
                                    fastmath=True,
                                )
                                raw_k_value = cutlass.Float32(
                                    raw_k[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                q_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_q[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * gamma,
                                )
                                erase_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_b[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * raw_k_value
                                    * gamma,
                                )
                            shared_q[
                                local_token,
                                key_channel,
                                prepare_stage,
                            ] = q_value
                            shared_erase[
                                local_token,
                                key_channel,
                                prepare_stage,
                            ] = erase_value

                        for linear in cutlass.range(
                            thread_in_group,
                            CHUNK_SIZE * _WGMMA_K,
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // _WGMMA_K
                            tile_channel = linear % _WGMMA_K
                            key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                            key_value = cutlass.BFloat16(0.0)
                            factor_key_value = cutlass.BFloat16(0.0)
                            if local_token < prepare_valid_tokens:
                                g_value = cutlass.Float32(
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                g_end = cutlass.Float32(
                                    raw_g[
                                        prepare_valid_tokens - cutlass.Int32(1),
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                tail_gamma = cute.math.exp2(
                                    (g_end - g_value) * cutlass.Float32(_INV_LN2),
                                    fastmath=True,
                                )
                                key_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_k[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * tail_gamma,
                                )
                                factor_key_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_k[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * cute.math.exp2(
                                        -g_value * cutlass.Float32(_INV_LN2),
                                        fastmath=True,
                                    ),
                                )
                            shared_key_tail[
                                key_channel,
                                local_token,
                                prepare_stage,
                            ] = key_value
                            shared_factor_k[
                                local_token,
                                key_channel,
                                cutlass.Int32(0),
                            ] = factor_key_value
                            if local_token == cutlass.Int32(0):
                                shared_gamma_end[
                                    key_channel,
                                    prepare_stage,
                                ] = cute.math.exp2(
                                    cutlass.Float32(
                                        raw_g[
                                            prepare_valid_tokens - cutlass.Int32(1),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * cutlass.Float32(_INV_LN2),
                                    fastmath=True,
                                )

                        if state_slab == cutlass.Int32(0):
                            qkb0_pipeline.consumer_release(qkb_consumer)
                        else:
                            qkb1_pipeline.consumer_release(qkb_consumer)
                        qkb_consumer.advance()

                    raw_handoff.consumer_release(
                        pipeline.PipelineState(
                            1,
                            prepare_chunk,
                            cutlass.Int32(0),
                            prepare_chunk % cutlass.Int32(2),
                        ),
                    )

                    # Both State WGs publish disjoint halves of the common
                    # factor operands. Order all 256 stores before WG0 starts
                    # the next-generation factor.
                    cute.arch.barrier(
                        barrier_id=_STATE_COMMON_BARRIER,
                        number_of_threads=2 * _WARP_GROUP_SIZE,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    factor_ready_handoff.producer_commit(
                        pipeline.PipelineState(
                            _INPUT_STAGES,
                            prepare_chunk,
                            (prepare_chunk % cutlass.Int32(_INPUT_STAGES)),
                            cutlass.Int32(1) - ((prepare_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                        ),
                    )

                if state_step > cutlass.Int32(0):
                    # Prioritize the next factor-ready publication above. V/W
                    # for the current generation was issued after its factor
                    # completed and can now materialize while WG0 starts the
                    # next factor.
                    vw_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        _VW_PRIVATE_STAGES,
                    )
                    for local_value_tile in cutlass.range(
                        _VALUE_TILES // 2,
                        unroll=1,
                    ):
                        if state_slab == cutlass.Int32(0):
                            vw_ready = vw0_pipeline.consumer_try_wait(
                                vw_consumer,
                            )
                            vw0_pipeline.consumer_wait(
                                vw_consumer,
                                vw_ready,
                            )
                        else:
                            vw_ready = vw1_pipeline.consumer_try_wait(
                                vw_consumer,
                            )
                            vw1_pipeline.consumer_wait(
                                vw_consumer,
                                vw_ready,
                            )

                        for linear in cutlass.range(
                            thread_in_group,
                            CHUNK_SIZE * _WGMMA_K,
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // _WGMMA_K
                            tile_value = linear % _WGMMA_K
                            value_index = (
                                cutlass.Int32(
                                    local_value_tile * _WGMMA_K,
                                )
                                + tile_value
                            )
                            write_value = cutlass.BFloat16(0.0)
                            if local_token < current_valid_tokens:
                                write_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_v[
                                            local_token,
                                            tile_value,
                                            state_slab,
                                        ],
                                    )
                                    * cutlass.Float32(
                                        raw_w[
                                            local_token,
                                            tile_value,
                                            state_slab,
                                        ],
                                    ),
                                )
                            shared_write[
                                local_token,
                                shared_value_start + value_index,
                                cutlass.Int32(0),
                            ] = write_value

                        if state_slab == cutlass.Int32(0):
                            vw0_pipeline.consumer_release(vw_consumer)
                        else:
                            vw1_pipeline.consumer_release(vw_consumer)
                        vw_consumer.advance()

                    if state_slab == cutlass.Int32(0):
                        cute.arch.barrier(
                            barrier_id=_STATE0_WRITE_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=_STATE1_WRITE_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )

                    input_stage = current_stage

                    q_stage = shared_q[None, None, input_stage]
                    erase_stage = shared_erase[None, None, input_stage]
                    key_stage = shared_key_tail[None, None, input_stage]
                    aqk_stage = shared_aqk[None, None, input_stage]
                    akk_stage = shared_akk[None, None, input_stage]

                    q_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(q_stage),
                    )
                    q_stages = q_operand
                    erase_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(erase_stage),
                    )
                    erase_stages = erase_operand
                    aqk_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(aqk_stage),
                    )
                    aqk_stages = aqk_operand
                    akk_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(akk_stage),
                    )
                    akk_stages = akk_operand
                    key_operand = state_thread.make_fragment_B(
                        state_thread.partition_B(key_stage),
                    )
                    key_stages = key_operand

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
                                    cutlass.Int32(0),
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
                    output_handoff.producer_acquire(output_producer)
                    for element in cutlass.range_constexpr(
                        cute.size(output_accumulator),
                    ):
                        value_index, token_index = token_coordinates[element]
                        shared_output[
                            token_index,
                            shared_value_start + value_index,
                            output_producer.index,
                        ] = cutlass.BFloat16(output_accumulator[element])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    output_handoff.producer_commit(output_producer)
                    output_producer.advance()

                    for element in cutlass.range_constexpr(
                        cute.size(state_accumulator),
                    ):
                        _, key_index = state_coordinates[element]
                        state_accumulator[element] = state_accumulator[element] * shared_gamma_end[key_index, input_stage]

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
                    cute.arch.barrier(
                        barrier_id=_STATE_ITERATION_DONE_BARRIER,
                        number_of_threads=2 * _WARP_GROUP_SIZE,
                    )

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

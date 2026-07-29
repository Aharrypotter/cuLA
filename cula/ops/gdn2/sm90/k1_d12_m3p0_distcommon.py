# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""K1-D12 M3-P0 distributed-common exact-schedule candidate.

One 384-thread CTA owns ``(sequence,value_head)``. WG0 issues two independent
one-stage raw-Q/K/B/G streams plus private raw-V/W streams. WG1 prepares
common key tiles 0-3 and the low natural-V64 write slab; WG2 prepares common
key tiles 4-7 and the high natural-V64 write slab. The two State WGs then
synchronize before consuming the full common factors and execute unchanged
resident-state math.

The two raw-Q/K/B/G streams partition the existing two-stage storage, so the
candidate targets zero shared-memory growth. Barrier 4 orders distributed
common stores across all 256 State threads; barriers 2 and 3 preserve the
per-State-WG write-slab ordering. P40/S232/S232 register targets are inherited
from D11R1. This candidate is not a correctness, timing, or product-selection
result.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup

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
_AQK_STAGES = 2
_INPUT_STAGES = 2
_OUTPUT_STAGES = 2
_PRODUCER_SIGNAL_WARPS = _WARP_GROUP_SIZE // 32
_STATE_SIGNAL_WARPS = 2 * _WARP_GROUP_SIZE // 32
_PRODUCER_REGISTER_TARGET = 40
_STATE_REGISTER_TARGET = 232
_STATE_VALUE_TILE = 64
_QKBG_TRANSACTION_BYTES = 3 * CHUNK_SIZE * _WGMMA_K * 2 + CHUNK_SIZE * _WGMMA_K * 4
_VW_TRANSACTION_BYTES = 2 * CHUNK_SIZE * _WGMMA_K * 2
_AQK_TRANSACTION_BYTES = 2 * CHUNK_SIZE * CHUNK_SIZE * 2
_STORE_WG_BARRIER = 1
_STATE0_WRITE_BARRIER = 2
_STATE1_WRITE_BARRIER = 3
_STATE_COMMON_BARRIER = 4
_QKB_STREAM_TILES = _RAW_KEY_TILES // 2


class A2K1D12M3P0DistCommon:
    """M3-P0 full-V128 CTA with distributed common-factor preparation."""

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
        token_operand_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        state_update_layout = cute.tile_to_shape(
            operand_layout_atom,
            (HEAD_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        write_layout = cute.make_layout(
            (CHUNK_SIZE, self.value_tile, _INPUT_STAGES),
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

        capsule_chunks = cute.size(capsule_aqk, mode=[0])
        capsule_heads = cute.size(capsule_aqk, mode=[1])
        capsule_layout = cute.make_layout(
            (
                CHUNK_SIZE,
                CHUNK_SIZE,
                capsule_chunks,
                capsule_heads,
            ),
            stride=(
                CHUNK_SIZE,
                1,
                capsule_heads * CHUNK_SIZE * CHUNK_SIZE,
                CHUNK_SIZE * CHUNK_SIZE,
            ),
        )
        aqk_global = cute.make_tensor(capsule_aqk.iterator, capsule_layout)
        akk_global = cute.make_tensor(capsule_akk.iterator, capsule_layout)
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
        matrix_tma_layout = cute.slice_(
            token_operand_layout,
            (None, None, 0),
        )
        aqk_atom, aqk_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            aqk_global,
            matrix_tma_layout,
            (CHUNK_SIZE, CHUNK_SIZE),
        )
        akk_atom, akk_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            akk_global,
            matrix_tma_layout,
            (CHUNK_SIZE, CHUNK_SIZE),
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
            aqk_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _AQK_STAGES,
            ]
            input_handoff_barriers: cute.struct.MemRange[
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
            aqk_atom,
            aqk_tma,
            akk_atom,
            akk_tma,
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
            raw_layout,
            g_staging_layout,
            key_operand_layout,
            token_operand_layout,
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
        aqk_atom: cute.CopyAtom,
        aqk_tma: cute.Tensor,
        akk_atom: cute.CopyAtom,
        akk_tma: cute.Tensor,
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
        raw_layout: cute.ComposedLayout,
        g_staging_layout: cute.Layout,
        key_operand_layout: cute.ComposedLayout,
        token_operand_layout: cute.ComposedLayout,
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
        sequence_start = cutlass.Int32(sequence_start_i64)
        sequence_end = cutlass.Int32(sequence_end_i64)

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
        aqk_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.aqk_barriers.data_ptr(),
            num_stages=_AQK_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _STATE_SIGNAL_WARPS,
            ),
            tx_count=_AQK_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        input_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.input_handoff_barriers.data_ptr(),
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
            cpasync.prefetch_descriptor(aqk_atom)
            cpasync.prefetch_descriptor(akk_atom)
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
            producer_sequence_chunks = (sequence_end - sequence_start + cutlass.Int32(CHUNK_SIZE - 1)) // cutlass.Int32(
                CHUNK_SIZE
            )
            qkb0_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            qkb1_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            vw0_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _VW_PRIVATE_STAGES,
            )
            vw1_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _VW_PRIVATE_STAGES,
            )
            aqk_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _AQK_STAGES,
            )
            input_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _INPUT_STAGES,
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
                if pipeline_step < producer_sequence_chunks:
                    local_chunk = pipeline_step
                    chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        producer_sequence_end = cutlass.Int32(
                            cu_seqlens[sequence + cutlass.Int32(1)],
                        )
                        valid_tokens = producer_sequence_end - chunk_start
                    flat_chunk = flat_chunk_base + local_chunk

                    input_handoff.producer_acquire(input_producer)

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

                    v_use = cute.domain_offset(
                        (chunk_start, value_start, cutlass.Int32(0)),
                        v_tma,
                    )
                    w_use = cute.domain_offset(
                        (chunk_start, value_start, cutlass.Int32(0)),
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

                    aqk_tile = cute.zipped_divide(
                        aqk_tma[None, None, flat_chunk, q_head],
                        (CHUNK_SIZE, CHUNK_SIZE),
                    )[
                        (
                            (None, None),
                            (cutlass.Int32(0), cutlass.Int32(0)),
                        )
                    ]
                    akk_tile = cute.zipped_divide(
                        akk_tma[None, None, flat_chunk, q_head],
                        (CHUNK_SIZE, CHUNK_SIZE),
                    )[
                        (
                            (None, None),
                            (cutlass.Int32(0), cutlass.Int32(0)),
                        )
                    ]
                    aqk_stage = shared_aqk[
                        None,
                        None,
                        aqk_producer.index,
                    ]
                    akk_stage = shared_akk[
                        None,
                        None,
                        aqk_producer.index,
                    ]
                    aqk_smem, aqk_gmem = cpasync.tma_partition(
                        aqk_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(aqk_stage, 0, 2),
                        cute.group_modes(aqk_tile, 0, 2),
                    )
                    akk_smem, akk_gmem = cpasync.tma_partition(
                        akk_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(akk_stage, 0, 2),
                        cute.group_modes(akk_tile, 0, 2),
                    )
                    if warp_index == cutlass.Int32(0):
                        aqk_pipeline.producer_acquire(aqk_producer)
                        aqk_barrier = aqk_pipeline.producer_get_barrier(
                            aqk_producer,
                        )
                        cute.copy(
                            aqk_atom,
                            aqk_gmem,
                            aqk_smem,
                            tma_bar_ptr=aqk_barrier,
                        )
                        cute.copy(
                            akk_atom,
                            akk_gmem,
                            akk_smem,
                            tma_bar_ptr=aqk_barrier,
                        )
                        aqk_pipeline.producer_commit(aqk_producer)
                        aqk_producer.advance()

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

                    if warp_index == cutlass.Int32(0):
                        vw0_pipeline.producer_acquire(vw0_producer)
                        vw0_barrier = vw0_pipeline.producer_get_barrier(
                            vw0_producer,
                        )
                        cute.copy(
                            v_atom,
                            v_gmem[(None, 0, cutlass.Int32(0))],
                            v_smem[(None, 0)],
                            tma_bar_ptr=vw0_barrier,
                        )
                        cute.copy(
                            w_atom,
                            w_gmem[(None, 0, cutlass.Int32(0))],
                            w_smem[(None, 0)],
                            tma_bar_ptr=vw0_barrier,
                        )
                        vw0_pipeline.producer_commit(vw0_producer)
                        vw0_producer.advance()

                        vw1_pipeline.producer_acquire(vw1_producer)
                        vw1_barrier = vw1_pipeline.producer_get_barrier(
                            vw1_producer,
                        )
                        cute.copy(
                            v_atom,
                            v_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_VALUE_TILES // 2),
                                )
                            ],
                            v_smem[(None, 1)],
                            tma_bar_ptr=vw1_barrier,
                        )
                        cute.copy(
                            w_atom,
                            w_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_VALUE_TILES // 2),
                                )
                            ],
                            w_smem[(None, 1)],
                            tma_bar_ptr=vw1_barrier,
                        )
                        vw1_pipeline.producer_commit(vw1_producer)
                        vw1_producer.advance()

                    # Publish raw work after both common streams and both
                    # private V/W streams have their first tile in flight.
                    input_handoff.producer_commit(input_producer)
                    input_producer.advance()

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

                    for local_value_tile in cutlass.range(
                        1,
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
                if pipeline_step > cutlass.Int32(0):
                    local_chunk = pipeline_step - cutlass.Int32(1)
                    chunk_start = sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        producer_sequence_end = cutlass.Int32(
                            cu_seqlens[sequence + cutlass.Int32(1)],
                        )
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
                            output_handoff.consumer_release(
                                output_release,
                            )
                            output_release.advance()
                        if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                            if warp_index == cutlass.Int32(0):
                                store_pipeline.producer_tail()
                            cute.arch.barrier(
                                barrier_id=_STORE_WG_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                            output_handoff.consumer_release(
                                output_release,
                            )
                            output_release.advance()
                    else:
                        if warp_index == cutlass.Int32(0):
                            store_pipeline.producer_tail()
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        if local_chunk > cutlass.Int32(0):
                            output_handoff.consumer_release(
                                output_release,
                            )
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

            input_handoff.producer_tail(input_producer)

        else:
            cute.arch.warpgroup_reg_alloc(_STATE_REGISTER_TARGET)
            state_sequence_chunks = (sequence_end - sequence_start + cutlass.Int32(CHUNK_SIZE - 1)) // cutlass.Int32(
                CHUNK_SIZE
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

            input_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _INPUT_STAGES,
            )
            qkb_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                1,
            )
            vw_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _VW_PRIVATE_STAGES,
            )
            aqk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _AQK_STAGES,
            )
            output_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _OUTPUT_STAGES,
            )

            for state_chunk in cutlass.range(
                state_sequence_chunks,
                unroll=1,
            ):
                input_handoff.consumer_wait(input_consumer)
                input_stage = input_consumer.index
                state_chunk_start = sequence_start + state_chunk * cutlass.Int32(CHUNK_SIZE)
                state_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                if state_chunk + cutlass.Int32(1) == state_sequence_chunks:
                    state_valid_tokens = sequence_end - state_chunk_start

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
                        if local_token < state_valid_tokens:
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
                            input_stage,
                        ] = q_value
                        shared_erase[
                            local_token,
                            key_channel,
                            input_stage,
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
                        if local_token < state_valid_tokens:
                            g_value = cutlass.Float32(
                                raw_g[
                                    local_token,
                                    tile_channel,
                                    raw_stage,
                                ],
                            )
                            g_end = cutlass.Float32(
                                raw_g[
                                    state_valid_tokens - cutlass.Int32(1),
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
                        shared_key_tail[
                            key_channel,
                            local_token,
                            input_stage,
                        ] = key_value
                        if local_token == cutlass.Int32(0):
                            shared_gamma_end[
                                key_channel,
                                input_stage,
                            ] = cute.math.exp2(
                                cutlass.Float32(
                                    raw_g[
                                        state_valid_tokens - cutlass.Int32(1),
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
                        if local_token < state_valid_tokens:
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
                            input_stage,
                        ] = write_value

                    if state_slab == cutlass.Int32(0):
                        vw0_pipeline.consumer_release(vw_consumer)
                    else:
                        vw1_pipeline.consumer_release(vw_consumer)
                    vw_consumer.advance()

                # WG1 and WG2 own disjoint common key tiles, but both consume
                # the full common domain. Order all 256 distributed stores
                # before either State WG enters resident-state math.
                cute.arch.barrier(
                    barrier_id=_STATE_COMMON_BARRIER,
                    number_of_threads=2 * _WARP_GROUP_SIZE,
                )

                # Preserve the source-bound per-State-WG ordering for each
                # private write slab. The common barrier also orders these
                # stores, but keeping the distinct edge makes the inherited
                # racecheck contract explicit.
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

                aqk_ready = aqk_pipeline.consumer_try_wait(aqk_consumer)
                aqk_pipeline.consumer_wait(aqk_consumer, aqk_ready)
                output_handoff.producer_acquire(output_producer)

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
                                input_stage,
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
                        output_producer.index,
                    ] = cutlass.BFloat16(output_accumulator[element])
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

                aqk_pipeline.consumer_release(aqk_consumer)
                aqk_consumer.advance()
                input_handoff.consumer_release(input_consumer)
                input_consumer.advance()

            output_handoff.producer_tail(output_producer)
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

# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Candidate M asymmetric BF16 WGMMA stage."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync

from .config import CHUNK_SIZE, HEAD_SIZE, THREADS_PER_CTA

K_TILE = 16
INPUT_TILES = HEAD_SIZE // K_TILE
PIPELINE_STAGES = 2
CONSUMER_SIGNAL_THREADS = THREADS_PER_CTA // 32
TMA_TRANSACTION_BYTES = 2 * CHUNK_SIZE * K_TILE * 2
EPILOGUE_ERASE = 0
EPILOGUE_QK = 1


class AsymmetricWgmma:
    """Compute one 64x64 product and publish its GDN2 epilogue."""

    @cute.jit
    def __call__(
        self,
        operand_a: cute.Tensor,
        operand_b: cute.Tensor,
        raw_matrix: cute.Tensor,
        epilogue_matrix: cute.Tensor,
        valid_tokens: cutlass.Int32,
        epilogue_mode: cutlass.Int32,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        mma_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (CHUNK_SIZE, CHUNK_SIZE, K_TILE),
            warpgroup.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        tile_shape_mnk = (CHUNK_SIZE, CHUNK_SIZE, K_TILE)
        a_smem_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            cutlass.BFloat16,
            PIPELINE_STAGES,
        )
        b_smem_layout = sm90_utils.make_smem_layout_b(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            cutlass.BFloat16,
            PIPELINE_STAGES,
        )
        output_smem_layout = sm90_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (CHUNK_SIZE, CHUNK_SIZE),
            1,
        )
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            operand_a,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (CHUNK_SIZE, K_TILE),
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            operand_b,
            cute.slice_(b_smem_layout, (None, None, 0)),
            (CHUNK_SIZE, K_TILE),
        )

        @cute.struct
        class SharedStorage:
            pipeline_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * PIPELINE_STAGES,
            ]
            operand_a: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_smem_layout)],
                128,
            ]
            operand_b: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_smem_layout)],
                128,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(output_smem_layout)],
                128,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            raw_matrix,
            epilogue_matrix,
            valid_tokens,
            epilogue_mode,
            scale,
            tiled_mma,
            a_smem_layout,
            b_smem_layout,
            output_smem_layout,
        ).launch(
            grid=(1, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        operand_a: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        operand_b: cute.Tensor,
        raw_matrix: cute.Tensor,
        epilogue_matrix: cute.Tensor,
        valid_tokens: cutlass.Int32,
        epilogue_mode: cutlass.Int32,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        output_smem_layout: cute.ComposedLayout,
    ) -> None:
        thread_index, _, _ = cute.arch.thread_idx()
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())
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
        shared_output = storage.output.get_tensor(
            output_smem_layout.outer,
            swizzle=output_smem_layout.inner,
        )
        transfer_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.pipeline_barriers.data_ptr(),
            num_stages=PIPELINE_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                CONSUMER_SIGNAL_THREADS,
            ),
            tx_count=TMA_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        grouped_shared_a = cute.group_modes(shared_a, 0, 2)
        grouped_shared_b = cute.group_modes(shared_b, 0, 2)
        grouped_global_a = cute.group_modes(operand_a, 0, 2)
        grouped_global_b = cute.group_modes(operand_b, 0, 2)
        tma_shared_a, tma_global_a = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            grouped_shared_a,
            grouped_global_a,
        )
        tma_shared_b, tma_global_b = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            grouped_shared_b,
            grouped_global_b,
        )
        if warp_index == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        warp_group_mma = tiled_mma.get_slice(0)
        logical_output_fragment = warp_group_mma.partition_C(raw_matrix)
        a_fragment = tiled_mma.make_fragment_A(warp_group_mma.partition_A(shared_a))
        b_fragment = tiled_mma.make_fragment_B(warp_group_mma.partition_B(shared_b))
        accumulator = cute.make_rmem_tensor(logical_output_fragment.shape, cutlass.Float32)
        accumulator.fill(0.0)
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            PIPELINE_STAGES,
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            PIPELINE_STAGES,
        )
        for tile_index in cutlass.range_constexpr(INPUT_TILES):
            if warp_index == 0:
                transfer_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tma_global_a[(None, producer_state.count)],
                    tma_shared_a[(None, producer_state.index)],
                    tma_bar_ptr=transfer_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tma_global_b[(None, producer_state.count)],
                    tma_shared_b[(None, producer_state.index)],
                    tma_bar_ptr=transfer_pipeline.producer_get_barrier(producer_state),
                )
                transfer_pipeline.producer_commit(producer_state)
                producer_state.advance()
            ready = transfer_pipeline.consumer_try_wait(consumer_state)
            transfer_pipeline.consumer_wait(consumer_state, ready)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, tile_index != 0)
            warpgroup.fence()
            cute.gemm(
                tiled_mma,
                accumulator,
                a_fragment[(None, None, None, consumer_state.index)],
                b_fragment[(None, None, None, consumer_state.index)],
                accumulator,
            )
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            transfer_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=cutlass.Float32,
        )
        stmatrix_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4),
            cutlass.BFloat16,
        )
        tiled_stmatrix = cute.make_tiled_copy_C_atom(stmatrix_atom, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_stmatrix)
        thread_copy_r2s = tiled_copy_r2s.get_slice(thread_index)
        shared_output_fragment = thread_copy_r2s.partition_D(shared_output)
        accumulator_for_store = tiled_copy_r2s.retile(accumulator)
        register_shape = cute.shape(thread_copy_r2s.partition_S(shared_output))
        register_layout = cute.make_layout(register_shape[:3])
        store_accumulator = cute.make_rmem_tensor_like(register_layout, cutlass.Float32)
        for element_index in cutlass.range_constexpr(cute.size(store_accumulator)):
            store_accumulator[element_index] = accumulator_for_store[element_index]
        store_output = cute.make_rmem_tensor_like(register_layout, cutlass.BFloat16)
        store_output.store(store_accumulator.load().to(cutlass.BFloat16))
        cute.copy(
            tiled_copy_r2s,
            store_output,
            shared_output_fragment[(None, None, None, 0)],
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        for linear_index in cutlass.range(
            thread_index,
            CHUNK_SIZE * CHUNK_SIZE,
            THREADS_PER_CTA,
            unroll=1,
        ):
            row = linear_index // CHUNK_SIZE
            column = linear_index % CHUNK_SIZE
            raw = shared_output[row, column, 0]
            raw_matrix[row, column] = raw
            value = cutlass.Float32(0.0)
            active = row < valid_tokens and column < valid_tokens
            if epilogue_mode == cutlass.Int32(EPILOGUE_ERASE):
                if active and row > column:
                    value = cutlass.Float32(raw)
                elif active and row == column:
                    value = cutlass.Float32(1.0)
            else:
                if active and row >= column:
                    value = cutlass.Float32(raw) * scale
            epilogue_matrix[row, column] = value

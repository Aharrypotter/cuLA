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

"""Experimental SM90 CuTe DSL GDN prefill entry point.

The migrated SM90 CUTLASS kernel remains the default production baseline. This
module is the staging boundary for the issue #76 CuTe DSL rewrite so the new
kernel can be developed and benchmarked without changing the public API.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_HEAD_SIZE = 128
_CHUNK_SIZE = 64
_THREADS_PER_CTA = 128


def is_sm90_gdn_prefill_dsl_available(device: torch.device | int | str | None = None) -> bool:
    """Return whether the current environment can attempt the SM90 DSL path."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major == 9 and props.minor == 0


def _check_minimal_dsl_support(inputs) -> None:
    q = inputs.q
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"SM90 GDN DSL currently supports bf16 only, got {q.dtype}.")
    if q.size(2) != _HEAD_SIZE:
        raise NotImplementedError(f"SM90 GDN DSL currently supports head_size={_HEAD_SIZE}, got {q.size(2)}.")


@functools.cache
def _get_sm90_dsl_compile_cache(
    total_tokens: int,
    num_seqs: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    num_o_heads: int,
    use_initial_state: bool,
    store_final_state: bool,
    scale: float,
):
    return {}


class _SM90GDNPrefillReferenceKernel:
    """Correctness-first SM90 DSL scaffold used to validate the recurrence.

    One CTA owns one ``(sequence, output_head)`` pair. The CTA keeps the full
    128x128 fp32 recurrent state in shared memory, stages Q/K/V/gate/beta in
    64-token chunks, emits output through the chunked inter+intra decomposition,
    and advances state with the chunked ``Phi*S_prev + K^T@decay(new_v)``
    update. This is intentionally not the final high-performance structure; it
    gives the CuTe DSL path a numerical anchor before the FlashInfer SM90
    CUTLASS stage graph is rewritten in CuTe DSL.
    """

    def __init__(
        self,
        *,
        total_tokens: int,
        num_seqs: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        num_o_heads: int,
        use_initial_state: bool,
        store_final_state: bool,
        scale: float,
    ):
        self.total_tokens = total_tokens
        self.num_seqs = num_seqs
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_o_heads = num_o_heads
        self.is_gqa = num_q_heads >= num_v_heads
        self.q_per_kv = num_q_heads // num_k_heads if self.is_gqa else 1
        self.v_per_q = num_v_heads // num_q_heads if not self.is_gqa else 1
        self.use_initial_state = use_initial_state
        self.store_final_state = store_final_state
        self.scale = scale
        self.head_size = _HEAD_SIZE
        self.chunk_size = _CHUNK_SIZE
        self.threads_per_cta = _THREADS_PER_CTA

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state: cute.Tensor,
        output_state: cute.Tensor,
        stream,
    ):
        state_layout = cute.make_layout(
            (self.head_size, self.head_size),
            stride=(self.head_size, 1),
        )
        qkv_tile_layout = cute.make_layout(
            (self.chunk_size, self.head_size),
            stride=(self.head_size, 1),
        )
        gate_tile_layout = cute.make_layout((self.chunk_size,), stride=(1,))
        score_tile_layout = cute.make_layout(
            (self.chunk_size, self.chunk_size),
            stride=(self.chunk_size, 1),
        )
        self.kernel(
            q,
            k,
            v,
            gate,
            beta,
            output,
            cu_seqlens,
            initial_state,
            output_state,
            state_layout,
            qkv_tile_layout,
            gate_tile_layout,
            score_tile_layout,
        ).launch(
            grid=(self.num_seqs * self.num_o_heads, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state: cute.Tensor,
        output_state: cute.Tensor,
        state_layout: cute.Layout,
        qkv_tile_layout: cute.Layout,
        gate_tile_layout: cute.Layout,
        score_tile_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        seq_idx = block_idx // self.num_o_heads
        head_idx = block_idx % self.num_o_heads
        q_head_idx = 0
        k_head_idx = 0
        v_head_idx = 0
        if cutlass.const_expr(self.is_gqa):
            q_head_idx = head_idx
            k_head_idx = head_idx // self.q_per_kv
            v_head_idx = head_idx // self.q_per_kv
        else:
            q_head_idx = head_idx // self.v_per_q
            k_head_idx = head_idx // self.v_per_q
            v_head_idx = head_idx
        v_col = tidx

        smem = utils.SmemAllocator()
        state = smem.allocate_tensor(cutlass.Float32, state_layout, 16)
        q_tile = smem.allocate_tensor(cutlass.BFloat16, qkv_tile_layout, 16)
        k_tile = smem.allocate_tensor(cutlass.BFloat16, qkv_tile_layout, 16)
        v_tile = smem.allocate_tensor(cutlass.BFloat16, qkv_tile_layout, 16)
        new_v_tile = smem.allocate_tensor(cutlass.BFloat16, qkv_tile_layout, 16)
        intra_out_tile = smem.allocate_tensor(cutlass.BFloat16, qkv_tile_layout, 16)
        gate_tile = smem.allocate_tensor(cutlass.Float32, gate_tile_layout, 16)
        beta_tile = smem.allocate_tensor(cutlass.Float32, gate_tile_layout, 16)
        cumprod_tile = smem.allocate_tensor(cutlass.Float32, gate_tile_layout, 16)
        decay_tile = smem.allocate_tensor(cutlass.Float32, gate_tile_layout, 16)
        qk_tile = smem.allocate_tensor(cutlass.Float32, score_tile_layout, 16)
        kk_tile = smem.allocate_tensor(cutlass.Float32, score_tile_layout, 16)
        inv_kk_beta_tile = smem.allocate_tensor(cutlass.Float32, score_tile_layout, 16)
        # Reuse existing bf16 tiles for projection staging to avoid pushing SMEM past
        # the SM90 per-CTA limit before the MMA/TMA rewrite.
        qs_tile = v_tile
        inv_kk_beta_bf16_tile = intra_out_tile

        self._load_initial_state_reference(initial_state, state, seq_idx, head_idx, v_col)

        cute.arch.barrier()

        seq_start = cu_seqlens[seq_idx]
        seq_end = cu_seqlens[seq_idx + 1]

        for chunk_start in cutlass.range(seq_start, seq_end, self.chunk_size, unroll=0):
            self._load_qkv_alpha_beta_reference(
                q,
                k,
                v,
                gate,
                beta,
                q_tile,
                k_tile,
                v_tile,
                gate_tile,
                beta_tile,
                chunk_start,
                seq_end,
                q_head_idx,
                k_head_idx,
                v_head_idx,
                head_idx,
                v_col,
            )

            cute.arch.barrier()

            self._stage_alpha_decay_reference(gate_tile, cumprod_tile, decay_tile, tidx)

            self._compute_aux_reference(
                q_tile,
                k_tile,
                qk_tile,
                kk_tile,
                inv_kk_beta_tile,
                inv_kk_beta_bf16_tile,
                cumprod_tile,
                beta_tile,
                chunk_start,
                seq_end,
                tidx,
            )

            cute.arch.barrier()

            self._compute_state_reference(
                output,
                q_tile,
                k_tile,
                v_tile,
                state,
                qk_tile,
                qs_tile,
                intra_out_tile,
                new_v_tile,
                cumprod_tile,
                decay_tile,
                chunk_start,
                seq_end,
                head_idx,
                v_col,
            )

        if cutlass.const_expr(self.store_final_state):
            if v_col < self.head_size:
                for k_row in cutlass.range(self.head_size, unroll=1):
                    # Match the migrated FlashInfer SM90 path: output_state
                    # is [seq, head, V, K], while initial_state is [K, V].
                    output_state[seq_idx, head_idx, v_col, k_row] = state[k_row, v_col]

    @cute.jit
    def _load_initial_state_reference(
        self,
        initial_state: cute.Tensor,
        state: cute.Tensor,
        seq_idx,
        head_idx,
        v_col,
    ):
        if v_col < self.head_size:
            for k_row in cutlass.range(self.head_size, unroll=1):
                if cutlass.const_expr(self.use_initial_state):
                    state[k_row, v_col] = cutlass.Float32(initial_state[seq_idx, head_idx, k_row, v_col])
                else:
                    state[k_row, v_col] = cutlass.Float32(0.0)

    @cute.jit
    def _load_qkv_alpha_beta_reference(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        q_tile: cute.Tensor,
        k_tile: cute.Tensor,
        v_tile: cute.Tensor,
        gate_tile: cute.Tensor,
        beta_tile: cute.Tensor,
        chunk_start,
        seq_end,
        q_head_idx,
        k_head_idx,
        v_head_idx,
        head_idx,
        v_col,
    ):
        for tile_row in cutlass.range(0, self.chunk_size, unroll=0):
            token_idx = chunk_start + tile_row
            if token_idx < seq_end:
                q_tile[tile_row, v_col] = q[token_idx, q_head_idx, v_col]
                k_tile[tile_row, v_col] = k[token_idx, k_head_idx, v_col]
                v_tile[tile_row, v_col] = v[token_idx, v_head_idx, v_col]
                if v_col == 0:
                    gate_tile[tile_row] = cutlass.Float32(gate[token_idx, head_idx])
                    beta_tile[tile_row] = cutlass.Float32(beta[token_idx, head_idx])
            else:
                q_tile[tile_row, v_col] = cutlass.BFloat16(0.0)
                k_tile[tile_row, v_col] = cutlass.BFloat16(0.0)
                v_tile[tile_row, v_col] = cutlass.BFloat16(0.0)
                if v_col == 0:
                    gate_tile[tile_row] = cutlass.Float32(1.0)
                    beta_tile[tile_row] = cutlass.Float32(0.0)

    @cute.jit
    def _stage_alpha_decay_reference(
        self,
        gate_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        decay_tile: cute.Tensor,
        tidx,
    ):
        if tidx < self.chunk_size:
            cumprod_tile[tidx] = gate_tile[tidx]

        cute.arch.barrier()

        for scan_step in cutlass.range_constexpr(6):
            offset = 1 << scan_step
            prefix_factor = cutlass.Float32(1.0)
            if tidx < self.chunk_size and tidx >= offset:
                prefix_factor = cumprod_tile[tidx - offset]

            cute.arch.barrier()

            if tidx < self.chunk_size and tidx >= offset:
                cumprod_tile[tidx] = cumprod_tile[tidx] * prefix_factor

            cute.arch.barrier()

        if tidx < self.chunk_size:
            decay_tile[tidx] = cumprod_tile[self.chunk_size - 1] / cumprod_tile[tidx]

        cute.arch.barrier()

    @cute.jit
    def _stage_kk_store_and_inv_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        self._stage_kk_diagonal_8x8_inv_reference(kk_tile, inv_kk_beta_tile, chunk_start, seq_end, tidx)

        cute.arch.barrier()

        self._stage_kk_lower_blocks_inv_reference(
            kk_tile,
            inv_kk_beta_tile,
            mma_scratch_tile,
            chunk_start,
            seq_end,
            tidx,
        )

    @cute.jit
    def _stage_kk_diagonal_8x8_inv_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        if tidx < self.chunk_size:
            block_idx = tidx // 8
            local_row = tidx - block_idx * 8
            block_start = block_idx * 8
            row = block_start + local_row
            token_row = chunk_start + row
            for local_col in cutlass.range(0, 8, unroll=0):
                col = block_start + local_col
                value = cutlass.Float32(0.0)
                if token_row < seq_end:
                    if local_col == local_row:
                        value = cutlass.Float32(1.0)
                    elif local_col < local_row:
                        value = kk_tile[row, col]
                inv_kk_beta_tile[row, col] = value

        cute.arch.barrier()

        for src_local_row in cutlass.range(0, 7, unroll=0):
            if tidx < self.chunk_size:
                block_idx = tidx // 8
                local_row = tidx - block_idx * 8
                block_start = block_idx * 8
                row = block_start + local_row
                token_row = chunk_start + row
                if token_row < seq_end and local_row > src_local_row:
                    pivot_col = block_start + src_local_row
                    row_scale = -inv_kk_beta_tile[row, pivot_col]
                    for local_col in cutlass.range(0, 8, unroll=0):
                        if local_col < src_local_row:
                            col = block_start + local_col
                            inv_kk_beta_tile[row, col] = (
                                row_scale * inv_kk_beta_tile[block_start + src_local_row, col]
                                + inv_kk_beta_tile[row, col]
                            )
                    inv_kk_beta_tile[row, pivot_col] = row_scale

            cute.arch.barrier()

    @cute.jit
    def _stage_kk_lower_blocks_inv_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        for row_block in cutlass.range(1, 8, unroll=0):
            for col_block in cutlass.range(0, row_block, unroll=0):
                self._stage_kk_lower_8x8_block_inv_reference(
                    kk_tile,
                    inv_kk_beta_tile,
                    mma_scratch_tile,
                    row_block,
                    col_block,
                    chunk_start,
                    seq_end,
                    tidx,
                )

                cute.arch.barrier()

    @cute.jit
    def _stage_kk_lower_8x8_block_inv_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        row_block,
        col_block,
        chunk_start,
        seq_end,
        tidx,
    ):
        local_row = tidx // 8
        local_col = tidx - local_row * 8
        row_block_start = row_block * 8
        col_block_start = col_block * 8
        col = col_block_start + local_col

        self._stage_kk_lower_8x8_c_inv_a_reference(
            kk_tile,
            inv_kk_beta_tile,
            mma_scratch_tile,
            row_block,
            col_block,
            chunk_start,
            seq_end,
            tidx,
        )

        cute.arch.barrier()

        self._stage_kk_lower_8x8_inv_d_apply_reference(
            kk_tile,
            inv_kk_beta_tile,
            mma_scratch_tile,
            row_block_start,
            col_block_start,
            chunk_start,
            seq_end,
            tidx,
        )

    @cute.jit
    def _stage_kk_lower_8x8_c_inv_a_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        row_block,
        col_block,
        chunk_start,
        seq_end,
        tidx,
    ):
        local_row = tidx // 8
        local_col = tidx - local_row * 8
        row = row_block * 8 + local_row
        col = col_block * 8 + local_col

        if tidx < self.chunk_size:
            inv_kk_beta_tile[row, col] = cutlass.Float32(0.0)

        cute.arch.barrier()

        for dep_block in cutlass.range(0, row_block, unroll=0):
            self._stage_kk_lower_8x8_c_inv_a_dep_block_reference(
                kk_tile,
                inv_kk_beta_tile,
                mma_scratch_tile,
                row_block,
                dep_block,
                col_block,
                chunk_start,
                seq_end,
                tidx,
            )

    @cute.jit
    def _stage_kk_lower_8x8_c_inv_a_dep_block_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        row_block,
        dep_block,
        col_block,
        chunk_start,
        seq_end,
        tidx,
    ):
        lane_id = cute.arch.thread_idx()[0] % 32
        warp_idx = cute.arch.warp_idx() % 4

        scratch_iters = (16 * 32 + self.threads_per_cta - 1) // self.threads_per_cta
        for scratch_iter in cutlass.range(0, scratch_iters, unroll=0):
            scratch_idx = scratch_iter * self.threads_per_cta + tidx
            if scratch_idx < 16 * 32:
                scratch_row = scratch_idx // 32
                scratch_col = scratch_idx - scratch_row * 32
                value = cutlass.BFloat16(0.0)
                if scratch_col < 16:
                    if scratch_row < 8 and scratch_col < 8:
                        row = row_block * 8 + scratch_row
                        token_row = chunk_start + row
                        if token_row < seq_end:
                            value = cutlass.BFloat16(kk_tile[row, dep_block * 8 + scratch_col])
                    mma_scratch_tile[scratch_row, scratch_col] = value
                else:
                    b_col = scratch_col - 16
                    if scratch_row < 8 and b_col < 8:
                        value = cutlass.BFloat16(
                            inv_kk_beta_tile[dep_block * 8 + b_col, col_block * 8 + scratch_row]
                        )
                    mma_scratch_tile[scratch_row, scratch_col] = value

        cute.arch.barrier()

        if warp_idx == 0:
            mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
                ab_dtype=cutlass.BFloat16,
                acc_dtype=cutlass.Float32,
                shape_mnk=(16, 8, 16),
            )
            tiled_mma = cute.make_tiled_mma(
                mma_atom,
                atom_layout_mnk=(1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(lane_id)

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            a_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
            b_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
            a_thr_copy = a_tiled_copy.get_slice(lane_id)
            b_thr_copy = b_tiled_copy.get_slice(lane_id)

            scratch_layout = cute.make_layout(
                (16, 16),
                stride=(self.head_size, 1),
            )
            a_scratch = cute.make_tensor(
                mma_scratch_tile.iterator,
                layout=scratch_layout,
            )
            b_scratch = cute.make_tensor(
                mma_scratch_tile.iterator + 16,
                layout=scratch_layout,
            )

            t_a = thr_mma.make_fragment_A(thr_mma.partition_A(a_scratch))
            t_b = thr_mma.make_fragment_B(thr_mma.partition_B(b_scratch))

            cute.copy(a_tiled_copy, a_thr_copy.partition_S(a_scratch), a_thr_copy.retile(t_a))
            cute.copy(b_tiled_copy, b_thr_copy.partition_S(b_scratch), b_thr_copy.retile(t_b))

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_a, t_b, t_out)

            out_coord = cute.make_identity_tensor((16, 16))
            t_out_coord = thr_mma.partition_C(out_coord)
            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                local_row, local_col = t_out_coord[i]
                if local_row < 8 and local_col < 8:
                    row = row_block * 8 + local_row
                    token_row = chunk_start + row
                    if token_row < seq_end:
                        col = col_block * 8 + local_col
                        inv_kk_beta_tile[row, col] = inv_kk_beta_tile[row, col] + t_out[i]

        cute.arch.barrier()

    @cute.jit
    def _stage_kk_lower_8x8_inv_d_apply_reference(
        self,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        mma_scratch_tile: cute.Tensor,
        row_block_start,
        col_block_start,
        chunk_start,
        seq_end,
        tidx,
    ):
        lane_id = cute.arch.thread_idx()[0] % 32
        warp_idx = cute.arch.warp_idx() % 4

        scratch_iters = (16 * 32 + self.threads_per_cta - 1) // self.threads_per_cta
        for scratch_iter in cutlass.range(0, scratch_iters, unroll=0):
            scratch_idx = scratch_iter * self.threads_per_cta + tidx
            if scratch_idx < 16 * 32:
                scratch_row = scratch_idx // 32
                scratch_col = scratch_idx - scratch_row * 32
                value = cutlass.BFloat16(0.0)
                if scratch_col < 16:
                    if scratch_row < 8 and scratch_col < 8:
                        value = cutlass.BFloat16(
                            inv_kk_beta_tile[row_block_start + scratch_row, row_block_start + scratch_col]
                        )
                    mma_scratch_tile[scratch_row, scratch_col] = value
                else:
                    b_col = scratch_col - 16
                    if scratch_row < 8 and b_col < 8:
                        value = cutlass.BFloat16(
                            inv_kk_beta_tile[row_block_start + b_col, col_block_start + scratch_row]
                        )
                    mma_scratch_tile[scratch_row, scratch_col] = value

        cute.arch.barrier()

        if warp_idx == 0:
            mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
                ab_dtype=cutlass.BFloat16,
                acc_dtype=cutlass.Float32,
                shape_mnk=(16, 8, 16),
            )
            tiled_mma = cute.make_tiled_mma(
                mma_atom,
                atom_layout_mnk=(1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(lane_id)

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            a_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
            b_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
            a_thr_copy = a_tiled_copy.get_slice(lane_id)
            b_thr_copy = b_tiled_copy.get_slice(lane_id)

            scratch_layout = cute.make_layout(
                (16, 16),
                stride=(self.head_size, 1),
            )
            a_scratch = cute.make_tensor(
                mma_scratch_tile.iterator,
                layout=scratch_layout,
            )
            b_scratch = cute.make_tensor(
                mma_scratch_tile.iterator + 16,
                layout=scratch_layout,
            )

            t_a = thr_mma.make_fragment_A(thr_mma.partition_A(a_scratch))
            t_b = thr_mma.make_fragment_B(thr_mma.partition_B(b_scratch))

            cute.copy(a_tiled_copy, a_thr_copy.partition_S(a_scratch), a_thr_copy.retile(t_a))
            cute.copy(b_tiled_copy, b_thr_copy.partition_S(b_scratch), b_thr_copy.retile(t_b))

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_a, t_b, t_out)

            out_coord = cute.make_identity_tensor((16, 16))
            t_out_coord = thr_mma.partition_C(out_coord)
            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                local_row, local_col = t_out_coord[i]
                if local_row < 8 and local_col < 8:
                    row = row_block_start + local_row
                    token_row = chunk_start + row
                    value = cutlass.Float32(0.0)
                    if token_row < seq_end:
                        value = -t_out[i]
                    inv_kk_beta_tile[row, col_block_start + local_col] = value

        cute.arch.barrier()

    @cute.jit
    def _stage_apply_inv_kk_beta_reference(
        self,
        inv_kk_beta_tile: cute.Tensor,
        inv_kk_beta_bf16_tile: cute.Tensor,
        beta_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        if tidx < self.chunk_size:
            col = tidx
            for row in cutlass.range(0, self.chunk_size, unroll=0):
                token_row = chunk_start + row
                value = cutlass.Float32(0.0)
                if token_row < seq_end and col <= row:
                    value = inv_kk_beta_tile[row, col] * beta_tile[col]
                inv_kk_beta_tile[row, col] = value
                inv_kk_beta_bf16_tile[row, col] = cutlass.BFloat16(value)

    @cute.jit
    def _compute_aux_reference(
        self,
        q_tile: cute.Tensor,
        k_tile: cute.Tensor,
        qk_tile: cute.Tensor,
        kk_tile: cute.Tensor,
        inv_kk_beta_tile: cute.Tensor,
        inv_kk_beta_bf16_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        beta_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        self._stage_raw_qk_kk_scores_mma_16x16(q_tile, k_tile, qk_tile, kk_tile)

        cute.arch.barrier()

        self._apply_qk_kk_gdn_epilogue(
            qk_tile,
            kk_tile,
            cumprod_tile,
            beta_tile,
            chunk_start,
            seq_end,
            tidx,
        )

        cute.arch.barrier()

        self._stage_kk_store_and_inv_reference(
            kk_tile,
            inv_kk_beta_tile,
            inv_kk_beta_bf16_tile,
            chunk_start,
            seq_end,
            tidx,
        )

        cute.arch.barrier()

        self._stage_apply_inv_kk_beta_reference(
            inv_kk_beta_tile,
            inv_kk_beta_bf16_tile,
            beta_tile,
            chunk_start,
            seq_end,
            tidx,
        )

    @cute.jit
    def _stage_qk_for_intra_output_reference(
        self,
        qk_tile: cute.Tensor,
        qk_bf16_tile: cute.Tensor,
        v_col,
    ):
        for tile_row in cutlass.range(0, self.chunk_size, unroll=0):
            if v_col < self.chunk_size:
                if v_col <= tile_row:
                    qk_bf16_tile[tile_row, v_col] = cutlass.BFloat16(qk_tile[tile_row, v_col])
                else:
                    qk_bf16_tile[tile_row, v_col] = cutlass.BFloat16(0.0)

    @cute.jit
    def _store_output_and_decay_new_v_reference(
        self,
        output: cute.Tensor,
        qs_tile: cute.Tensor,
        intra_out_tile: cute.Tensor,
        new_v_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        decay_tile: cute.Tensor,
        chunk_start,
        seq_end,
        head_idx,
        v_col,
    ):
        for tile_row in cutlass.range(0, self.chunk_size, unroll=0):
            token_idx = chunk_start + tile_row
            if token_idx < seq_end and v_col < self.head_size:
                output[token_idx, head_idx, v_col] = cutlass.BFloat16(
                    (
                        cumprod_tile[tile_row] * cutlass.Float32(qs_tile[tile_row, v_col])
                        + cutlass.Float32(intra_out_tile[tile_row, v_col])
                    )
                    * self.scale
                )
                new_v_tile[tile_row, v_col] = cutlass.BFloat16(
                    decay_tile[tile_row] * cutlass.Float32(new_v_tile[tile_row, v_col])
                )
            elif v_col < self.head_size:
                new_v_tile[tile_row, v_col] = cutlass.BFloat16(0.0)

    @cute.jit
    def _compute_state_reference(
        self,
        output: cute.Tensor,
        q_tile: cute.Tensor,
        k_tile: cute.Tensor,
        residual_tile: cute.Tensor,
        state: cute.Tensor,
        qk_tile: cute.Tensor,
        qs_tile: cute.Tensor,
        intra_out_tile: cute.Tensor,
        new_v_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        decay_tile: cute.Tensor,
        chunk_start,
        seq_end,
        head_idx,
        v_col,
    ):
        self._stage_residual_mma_16x16(
            k_tile,
            state,
            new_v_tile,
            residual_tile,
            cumprod_tile,
            chunk_start,
            seq_end,
        )

        cute.arch.barrier()

        self._stage_new_v_mma_16x16(intra_out_tile, residual_tile, new_v_tile)

        cute.arch.barrier()

        self._stage_state_projection_mma_16x16(q_tile, state, intra_out_tile, qs_tile)

        cute.arch.barrier()

        self._stage_qk_for_intra_output_reference(qk_tile, q_tile, v_col)

        cute.arch.barrier()

        self._stage_intra_out_mma_16x16(q_tile, new_v_tile, intra_out_tile)

        cute.arch.barrier()

        self._store_output_and_decay_new_v_reference(
            output,
            qs_tile,
            intra_out_tile,
            new_v_tile,
            cumprod_tile,
            decay_tile,
            chunk_start,
            seq_end,
            head_idx,
            v_col,
        )

        cute.arch.barrier()

        self._stage_state_update_mma_16x16(q_tile, k_tile, new_v_tile, state, cumprod_tile)

        cute.arch.barrier()

    @cute.jit
    def _stage_residual_mma_16x16(
        self,
        k_tile: cute.Tensor,
        state: cute.Tensor,
        state_scratch_tile: cute.Tensor,
        residual_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        chunk_start,
        seq_end,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.head_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        k_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        state_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        k_thr_copy = k_tiled_copy.get_slice(lane_id)
        state_thr_copy = state_tiled_copy.get_slice(lane_id)

        k_tiles = cute.flat_divide(k_tile, (16, self.head_size))
        state_scratch_layout = cute.make_layout(
            (16, self.head_size),
            stride=(self.head_size, 1),
        )
        state_scratch = cute.make_tensor(
            state_scratch_tile.iterator,
            layout=state_scratch_layout,
        )
        out_coord = cute.make_identity_tensor((16, 16))
        t_out_coord = thr_mma.partition_C(out_coord)

        num_scratch_iters = (16 * self.head_size + self.threads_per_cta - 1) // self.threads_per_cta
        for col_tile in cutlass.range(0, 8, unroll=0):
            col_base = col_tile * 16

            for scratch_iter in cutlass.range(0, num_scratch_iters, unroll=0):
                scratch_idx = scratch_iter * self.threads_per_cta + cute.arch.thread_idx()[0]
                if scratch_idx < 16 * self.head_size:
                    v_row = scratch_idx // self.head_size
                    k_col = scratch_idx % self.head_size
                    state_scratch[v_row, k_col] = cutlass.BFloat16(state[k_col, col_base + v_row])

            cute.arch.barrier()

            row_tile = warp_idx
            row_base = row_tile * 16

            s_k = k_tiles[None, None, row_tile, 0]

            t_k = thr_mma.make_fragment_A(thr_mma.partition_A(s_k))
            t_state = thr_mma.make_fragment_B(thr_mma.partition_B(state_scratch))

            cute.copy(k_tiled_copy, k_thr_copy.partition_S(s_k), k_thr_copy.retile(t_k))
            cute.copy(state_tiled_copy, state_thr_copy.partition_S(state_scratch), state_thr_copy.retile(t_state))

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_k, t_state, t_out)

            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                row, col = t_out_coord[i]
                token_idx = chunk_start + row_base + row
                if token_idx < seq_end:
                    residual_tile[row_base + row, col_base + col] = cutlass.BFloat16(
                        cutlass.Float32(residual_tile[row_base + row, col_base + col])
                        - cumprod_tile[row_base + row] * t_out[i]
                    )
                else:
                    residual_tile[row_base + row, col_base + col] = cutlass.BFloat16(0.0)

            cute.arch.barrier()

    @cute.jit
    def _stage_raw_qk_kk_scores_mma_16x16(
        self,
        q_tile: cute.Tensor,
        k_tile: cute.Tensor,
        qk_tile: cute.Tensor,
        kk_tile: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.head_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        q_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        k_a_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        k_b_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        q_thr_copy = q_tiled_copy.get_slice(lane_id)
        k_a_thr_copy = k_a_tiled_copy.get_slice(lane_id)
        k_b_thr_copy = k_b_tiled_copy.get_slice(lane_id)

        q_tiles = cute.flat_divide(q_tile, (16, self.head_size))
        k_tiles = cute.flat_divide(k_tile, (16, self.head_size))
        score_coord = cute.make_identity_tensor((16, 16))
        t_score_coord = thr_mma.partition_C(score_coord)

        for tile_group in cutlass.range(0, 4, unroll=0):
            tile_idx = tile_group * 4 + warp_idx
            row_tile = tile_idx // 4
            col_tile = tile_idx % 4
            row_base = row_tile * 16
            col_base = col_tile * 16

            if col_tile <= row_tile:
                s_q = q_tiles[None, None, row_tile, 0]
                s_k_row = k_tiles[None, None, row_tile, 0]
                s_k_col = k_tiles[None, None, col_tile, 0]

                t_q = thr_mma.make_fragment_A(thr_mma.partition_A(s_q))
                t_k_a = thr_mma.make_fragment_A(thr_mma.partition_A(s_k_row))
                t_k_b = thr_mma.make_fragment_B(thr_mma.partition_B(s_k_col))

                cute.copy(q_tiled_copy, q_thr_copy.partition_S(s_q), q_thr_copy.retile(t_q))
                cute.copy(k_a_tiled_copy, k_a_thr_copy.partition_S(s_k_row), k_a_thr_copy.retile(t_k_a))
                cute.copy(k_b_tiled_copy, k_b_thr_copy.partition_S(s_k_col), k_b_thr_copy.retile(t_k_b))

                t_qk = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
                t_kk = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
                t_qk.fill(0.0)
                t_kk.fill(0.0)

                cute.gemm(tiled_mma, t_qk, t_q, t_k_b, t_qk)
                cute.gemm(tiled_mma, t_kk, t_k_a, t_k_b, t_kk)

                for i in cutlass.range_constexpr(cute.size(t_score_coord)):
                    row, col = t_score_coord[i]
                    if col_base + col <= row_base + row:
                        qk_tile[row_base + row, col_base + col] = t_qk[i]
                        kk_tile[row_base + row, col_base + col] = t_kk[i]

    @cute.jit
    def _stage_state_projection_mma_16x16(
        self,
        input_tile: cute.Tensor,
        state: cute.Tensor,
        state_scratch_tile: cute.Tensor,
        output_tile: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.head_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        input_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        state_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        input_thr_copy = input_tiled_copy.get_slice(lane_id)
        state_thr_copy = state_tiled_copy.get_slice(lane_id)

        input_tiles = cute.flat_divide(input_tile, (16, self.head_size))
        state_scratch_layout = cute.make_layout(
            (16, self.head_size),
            stride=(self.head_size, 1),
        )
        state_scratch = cute.make_tensor(
            state_scratch_tile.iterator,
            layout=state_scratch_layout,
        )
        out_coord = cute.make_identity_tensor((16, 16))
        t_out_coord = thr_mma.partition_C(out_coord)

        num_scratch_iters = (16 * self.head_size + self.threads_per_cta - 1) // self.threads_per_cta
        for col_tile in cutlass.range(0, 8, unroll=0):
            col_base = col_tile * 16

            for scratch_iter in cutlass.range(0, num_scratch_iters, unroll=0):
                scratch_idx = scratch_iter * self.threads_per_cta + cute.arch.thread_idx()[0]
                if scratch_idx < 16 * self.head_size:
                    v_row = scratch_idx // self.head_size
                    k_col = scratch_idx % self.head_size
                    state_scratch[v_row, k_col] = cutlass.BFloat16(state[k_col, col_base + v_row])

            cute.arch.barrier()

            row_tile = warp_idx
            row_base = row_tile * 16

            s_input = input_tiles[None, None, row_tile, 0]

            t_input = thr_mma.make_fragment_A(thr_mma.partition_A(s_input))
            t_state = thr_mma.make_fragment_B(thr_mma.partition_B(state_scratch))

            cute.copy(input_tiled_copy, input_thr_copy.partition_S(s_input), input_thr_copy.retile(t_input))
            cute.copy(state_tiled_copy, state_thr_copy.partition_S(state_scratch), state_thr_copy.retile(t_state))

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_input, t_state, t_out)

            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                row, col = t_out_coord[i]
                output_tile[row_base + row, col_base + col] = cutlass.BFloat16(t_out[i])

            cute.arch.barrier()

    @cute.jit
    def _stage_intra_out_mma_16x16(
        self,
        qk_bf16_tile: cute.Tensor,
        new_v_tile: cute.Tensor,
        intra_out_tile: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.chunk_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        qk_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        nv_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(lane_id)
        nv_thr_copy = nv_tiled_copy.get_slice(lane_id)

        qk_tiles = cute.flat_divide(qk_bf16_tile, (16, self.chunk_size))
        nv_scratch_layout = cute.make_layout(
            (16, self.chunk_size),
            stride=(self.head_size, 1),
        )
        nv_scratch = cute.make_tensor(
            qk_bf16_tile.iterator + self.chunk_size,
            layout=nv_scratch_layout,
        )
        out_coord = cute.make_identity_tensor((16, 16))
        t_out_coord = thr_mma.partition_C(out_coord)

        num_scratch_iters = (16 * self.chunk_size + self.threads_per_cta - 1) // self.threads_per_cta
        for col_tile in cutlass.range(0, 8, unroll=0):
            col_base = col_tile * 16

            for scratch_iter in cutlass.range(0, num_scratch_iters, unroll=0):
                scratch_idx = scratch_iter * self.threads_per_cta + cute.arch.thread_idx()[0]
                if scratch_idx < 16 * self.chunk_size:
                    v_row = scratch_idx // self.chunk_size
                    k_col = scratch_idx % self.chunk_size
                    nv_scratch[v_row, k_col] = new_v_tile[k_col, col_base + v_row]

            cute.arch.barrier()

            row_tile = warp_idx
            row_base = row_tile * 16

            s_qk = qk_tiles[None, None, row_tile, 0]

            t_qk = thr_mma.make_fragment_A(thr_mma.partition_A(s_qk))
            t_nv = thr_mma.make_fragment_B(thr_mma.partition_B(nv_scratch))

            cute.copy(qk_tiled_copy, qk_thr_copy.partition_S(s_qk), qk_thr_copy.retile(t_qk))
            cute.copy(nv_tiled_copy, nv_thr_copy.partition_S(nv_scratch), nv_thr_copy.retile(t_nv))

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_qk, t_nv, t_out)

            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                row, col = t_out_coord[i]
                intra_out_tile[row_base + row, col_base + col] = cutlass.BFloat16(t_out[i])

            cute.arch.barrier()

    @cute.jit
    def _stage_new_v_mma_16x16(
        self,
        inv_kk_beta_bf16_tile: cute.Tensor,
        residual_tile: cute.Tensor,
        new_v_tile: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.chunk_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        inv_kk_beta_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        residual_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        inv_kk_beta_thr_copy = inv_kk_beta_tiled_copy.get_slice(lane_id)
        residual_thr_copy = residual_tiled_copy.get_slice(lane_id)

        inv_kk_beta_layout = cute.make_layout(
            (self.chunk_size, self.chunk_size),
            stride=(self.head_size, 1),
        )
        inv_kk_beta_scratch = cute.make_tensor(
            inv_kk_beta_bf16_tile.iterator,
            layout=inv_kk_beta_layout,
        )
        residual_scratch_layout = cute.make_layout(
            (16, self.chunk_size),
            stride=(self.head_size, 1),
        )
        residual_scratch = cute.make_tensor(
            inv_kk_beta_bf16_tile.iterator + self.chunk_size,
            layout=residual_scratch_layout,
        )
        inv_kk_beta_tiles = cute.flat_divide(inv_kk_beta_scratch, (16, self.chunk_size))
        out_coord = cute.make_identity_tensor((16, 16))
        t_out_coord = thr_mma.partition_C(out_coord)

        num_scratch_iters = (16 * self.chunk_size + self.threads_per_cta - 1) // self.threads_per_cta
        for col_tile in cutlass.range(0, 8, unroll=0):
            col_base = col_tile * 16

            for scratch_iter in cutlass.range(0, num_scratch_iters, unroll=0):
                scratch_idx = scratch_iter * self.threads_per_cta + cute.arch.thread_idx()[0]
                if scratch_idx < 16 * self.chunk_size:
                    v_row = scratch_idx // self.chunk_size
                    k_col = scratch_idx % self.chunk_size
                    residual_scratch[v_row, k_col] = residual_tile[k_col, col_base + v_row]

            cute.arch.barrier()

            row_tile = warp_idx
            row_base = row_tile * 16

            s_inv_kk_beta = inv_kk_beta_tiles[None, None, row_tile, 0]

            t_inv_kk_beta = thr_mma.make_fragment_A(thr_mma.partition_A(s_inv_kk_beta))
            t_residual = thr_mma.make_fragment_B(thr_mma.partition_B(residual_scratch))

            cute.copy(
                inv_kk_beta_tiled_copy,
                inv_kk_beta_thr_copy.partition_S(s_inv_kk_beta),
                inv_kk_beta_thr_copy.retile(t_inv_kk_beta),
            )
            cute.copy(
                residual_tiled_copy,
                residual_thr_copy.partition_S(residual_scratch),
                residual_thr_copy.retile(t_residual),
            )

            t_out = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_out.fill(0.0)
            cute.gemm(tiled_mma, t_out, t_inv_kk_beta, t_residual, t_out)

            for i in cutlass.range_constexpr(cute.size(t_out_coord)):
                row, col = t_out_coord[i]
                new_v_tile[row_base + row, col_base + col] = cutlass.BFloat16(t_out[i])

            cute.arch.barrier()

    @cute.jit
    def _stage_state_update_mma_16x16(
        self,
        scratch_tile: cute.Tensor,
        k_tile: cute.Tensor,
        decayed_new_v_tile: cute.Tensor,
        state: cute.Tensor,
        cumprod_tile: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx() % 4
        lane_id = cute.arch.thread_idx()[0] % 32

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            shape_mnk=(16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, self.chunk_size),
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        )
        k_tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        nv_tiled_copy = cute.make_tiled_copy_B(copy_atom, tiled_mma)
        k_thr_copy = k_tiled_copy.get_slice(lane_id)
        nv_thr_copy = nv_tiled_copy.get_slice(lane_id)

        scratch_tiles = cute.flat_divide(scratch_tile, (16, self.chunk_size))
        k_scratch = scratch_tiles[None, None, warp_idx, 0]
        nv_scratch = scratch_tiles[None, None, warp_idx, 1]
        state_coord = cute.make_identity_tensor((16, 16))
        t_state_coord = thr_mma.partition_C(state_coord)

        num_warp_scratch_iters = (16 * self.chunk_size + 31) // 32
        for tile_group in cutlass.range(0, 16, unroll=0):
            tile_idx = tile_group * 4 + warp_idx
            row_tile = tile_idx // 8
            col_tile = tile_idx % 8
            row_base = row_tile * 16
            col_base = col_tile * 16

            for scratch_iter in cutlass.range(0, num_warp_scratch_iters, unroll=0):
                scratch_idx = scratch_iter * 32 + lane_id
                if scratch_idx < 16 * self.chunk_size:
                    scratch_row = scratch_idx // self.chunk_size
                    token_col = scratch_idx % self.chunk_size
                    k_scratch[scratch_row, token_col] = k_tile[token_col, row_base + scratch_row]
                    nv_scratch[scratch_row, token_col] = decayed_new_v_tile[token_col, col_base + scratch_row]

            cute.arch.barrier()

            t_k = thr_mma.make_fragment_A(thr_mma.partition_A(k_scratch))
            t_nv = thr_mma.make_fragment_B(thr_mma.partition_B(nv_scratch))

            cute.copy(k_tiled_copy, k_thr_copy.partition_S(k_scratch), k_thr_copy.retile(t_k))
            cute.copy(nv_tiled_copy, nv_thr_copy.partition_S(nv_scratch), nv_thr_copy.retile(t_nv))

            t_state = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
            t_state.fill(0.0)
            cute.gemm(tiled_mma, t_state, t_k, t_nv, t_state)

            for i in cutlass.range_constexpr(cute.size(t_state_coord)):
                row, col = t_state_coord[i]
                state[row_base + row, col_base + col] = (
                    cumprod_tile[self.chunk_size - 1] * state[row_base + row, col_base + col] + t_state[i]
                )

            cute.arch.barrier()

    @cute.jit
    def _apply_qk_kk_gdn_epilogue(
        self,
        qk_tile: cute.Tensor,
        kk_tile: cute.Tensor,
        cumprod_tile: cute.Tensor,
        beta_tile: cute.Tensor,
        chunk_start,
        seq_end,
        tidx,
    ):
        if tidx < self.chunk_size:
            row = tidx
            token_row = chunk_start + row
            if token_row < seq_end:
                for col in cutlass.range(0, row + 1, unroll=0):
                    transfer = cutlass.Float32(0.0)
                    transfer = cumprod_tile[row] / cumprod_tile[col]
                    qk_tile[row, col] = transfer * qk_tile[row, col]
                    kk_tile[row, col] = (
                        beta_tile[row] * transfer * kk_tile[row, col] if col < row else cutlass.Float32(0.0)
                    )


def _compile_sm90_dsl_prefill(
    inputs,
    initial_state: torch.Tensor,
    output_state: torch.Tensor,
    stream,
):

    kernel = _SM90GDNPrefillReferenceKernel(
        total_tokens=inputs.q.size(0),
        num_seqs=inputs.num_seqs,
        num_q_heads=inputs.q.size(1),
        num_k_heads=inputs.k.size(1),
        num_v_heads=inputs.v.size(1),
        num_o_heads=inputs.num_o_heads,
        use_initial_state=inputs.initial_state is not None,
        store_final_state=inputs.output_state is not None,
        scale=inputs.scale,
    )

    q_cute = from_dlpack(inputs.q, assumed_align=16)
    k_cute = from_dlpack(inputs.k, assumed_align=16)
    v_cute = from_dlpack(inputs.v, assumed_align=16)
    gate_cute = from_dlpack(inputs.g, assumed_align=16)
    beta_cute = from_dlpack(inputs.beta, assumed_align=16)
    output_cute = from_dlpack(inputs.output, assumed_align=16)
    cu_seqlens_cute = from_dlpack(inputs.cu_seqlens_i32, assumed_align=4)
    initial_state_cute = from_dlpack(initial_state, assumed_align=16)
    output_state_cute = from_dlpack(output_state, assumed_align=16)

    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        gate_cute,
        beta_cute,
        output_cute,
        cu_seqlens_cute,
        initial_state_cute,
        output_state_cute,
        stream,
        options=_COMPILE_OPTIONS,
    )


def launch_sm90_gdn_prefill_dsl(inputs) -> None:
    """Launch the experimental SM90 CuTe DSL GDN prefill kernel.

    The current path is a correctness-oriented CuTe DSL scaffold for bf16
    head_size=128. It is deliberately narrow and slow relative to the migrated
    CUTLASS baseline; unsupported shapes fail explicitly so the default
    production path remains unchanged while the FlashInfer SM90 stage graph is
    rewritten.
    """
    if not is_sm90_gdn_prefill_dsl_available(inputs.q.device):
        props = torch.cuda.get_device_properties(inputs.q.device)
        raise RuntimeError(
            "SM90 GDN CuTe DSL prefill requires Hopper SM90, "
            f"got compute capability sm_{props.major}{props.minor}."
        )
    _check_minimal_dsl_support(inputs)

    import cuda.bindings.driver as cuda

    dummy_state = torch.empty(1, dtype=torch.float32, device=inputs.q.device)
    output_state = inputs.output_state if inputs.output_state is not None else dummy_state
    initial_state = inputs.initial_state if inputs.initial_state is not None else dummy_state
    stream = cuda.CUstream(torch.cuda.current_stream(device=inputs.q.device).cuda_stream)
    cache = _get_sm90_dsl_compile_cache(
        inputs.q.size(0),
        inputs.num_seqs,
        inputs.q.size(1),
        inputs.k.size(1),
        inputs.v.size(1),
        inputs.num_o_heads,
        inputs.initial_state is not None,
        inputs.output_state is not None,
        inputs.scale,
    )
    if "compiled" not in cache:
        cache["compiled"] = _compile_sm90_dsl_prefill(inputs, initial_state, output_state, stream)

    cache["compiled"](
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.output,
        inputs.cu_seqlens_i32,
        initial_state,
        output_state,
        stream,
    )

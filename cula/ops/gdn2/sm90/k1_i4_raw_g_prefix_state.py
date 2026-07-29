# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""I4 public-raw-g in-CTA factor/state lookahead candidate.

WG1/WG2 retain D12 distributed common transforms and resident FP32 state.
They construct the exact chunk-local FP32 prefix from public raw G in the
existing two-stage shared arena, then prepare Q/K/B/G-derived operands for
generation ``n + 1`` before executing recurrence generation ``n``. WG0
computes causal QK, erase, and collective inverse for ``n + 1`` concurrently
with that recurrence. No global G-prefix workspace or second launch is used.
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
_STATE0_PREFIX_BARRIER = 6
_STATE1_PREFIX_BARRIER = 7
_STORE_WG_BARRIER = 1
_INVERSE_BARRIER = 13
_FACTOR_EARLY_HALF_BARRIER = 8
_FACTOR_EARLY_HALF_BARRIER_ALT = 9
_QKB_STREAM_TILES = _RAW_KEY_TILES // 2
_FACTOR_PAIR_STAGES = _QKB_STREAM_TILES
_FACTOR_EARLY_HALF_STAGES = 1
_MAX_SEQUENCE_SCHEDULE = 32


@cute.jit
def _stable_lpt32_sequence(
    cu_seqlens: cute.Tensor,
    sequence_rank: cutlass.Int32,
    num_sequences: cutlass.Int32,
    lane: cutlass.Int32,
) -> cutlass.Int32:
    """Return the stable descending chunk-count sequence for one rank."""

    if num_sequences > cutlass.Int32(_MAX_SEQUENCE_SCHEDULE):
        _device_fail_closed()

    chunk_count = cutlass.Int32(-1)
    if lane < num_sequences:
        start = cutlass.Int64(cu_seqlens[lane])
        end = cutlass.Int64(
            cu_seqlens[lane + cutlass.Int32(1)],
        )
        if start < cutlass.Int64(0) or end <= start:
            _device_fail_closed()
        chunk_count = cutlass.Int32(
            (end - start + cutlass.Int64(CHUNK_SIZE - 1)) // cutlass.Int64(CHUNK_SIZE),
        )

    rank = cutlass.Int32(0)
    for source_lane in cutlass.range(num_sequences, unroll=0):
        other_chunk_count = cute.arch.shuffle_sync(
            chunk_count,
            source_lane,
        )
        if other_chunk_count > chunk_count or (other_chunk_count == chunk_count and source_lane < lane):
            rank = rank + cutlass.Int32(1)

    selected = cutlass.Int32(-1)
    if lane < num_sequences and rank == sequence_rank:
        selected = lane
    for delta in (16, 8, 4, 2, 1):
        other_selected = cute.arch.shuffle_sync_down(
            selected,
            delta,
        )
        if other_selected > selected:
            selected = other_selected
    sequence = cute.arch.shuffle_sync(selected, 0)
    if sequence < cutlass.Int32(0):
        _device_fail_closed()
    return sequence


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
def _publish_factor_qk_pair_streamed(
    factor_mma: cute.TiledMma,
    thread_in_group: cutlass.Int32,
    q_bar: cute.Tensor,
    k_bar: cute.Tensor,
    aqk: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    factor_pair_ready_handoff: pipeline.PipelineAsync,
    factor_chunk: cutlass.Int32,
) -> None:
    """Compute factor QK while waiting on low/high factor-K tile pairs."""

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
    for local_pair in cutlass.range_constexpr(_FACTOR_PAIR_STAGES):
        pair_iteration = factor_chunk * cutlass.Int32(_FACTOR_PAIR_STAGES) + cutlass.Int32(local_pair)
        factor_pair_ready_handoff.consumer_wait(
            pipeline.PipelineState(
                _FACTOR_PAIR_STAGES,
                pair_iteration,
                cutlass.Int32(local_pair),
                factor_chunk % cutlass.Int32(2),
            ),
        )
        factor_mma.set(
            warpgroup.Field.ACCUMULATE,
            local_pair != 0,
        )
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, local_pair)],
            k_fragment[(None, None, local_pair)],
            accumulator,
        )
        factor_mma.set(warpgroup.Field.ACCUMULATE, True)
        high_key_tile = local_pair + _FACTOR_PAIR_STAGES
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, high_key_tile)],
            k_fragment[(None, None, high_key_tile)],
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
def _publish_factor_qk_early_half_streamed(
    factor_mma: cute.TiledMma,
    thread_in_group: cutlass.Int32,
    q_bar: cute.Tensor,
    k_bar: cute.Tensor,
    aqk: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    factor_stream_ready_handoff: pipeline.PipelineAsync,
    factor_ready_handoff: pipeline.PipelineAsync,
    factor_chunk: cutlass.Int32,
    factor_consumer_state: pipeline.PipelineState,
    use_cta_barrier: bool,
    use_leader_handoff: bool,
    use_warp_leader_handoff: bool,
) -> None:
    """Issue four QK tiles after one early-half event, then wait full-ready."""

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
    if cutlass.const_expr(use_cta_barrier):
        cute.arch.barrier(
            barrier_id=_FACTOR_EARLY_HALF_BARRIER,
            number_of_threads=_THREADS_PER_CTA,
        )
    elif cutlass.const_expr(use_leader_handoff):
        if thread_in_group == cutlass.Int32(0):
            factor_stream_ready_handoff.consumer_wait(
                pipeline.PipelineState(
                    _FACTOR_EARLY_HALF_STAGES,
                    factor_chunk,
                    cutlass.Int32(0),
                    factor_chunk % cutlass.Int32(2),
                ),
            )
        cute.arch.barrier(
            barrier_id=_FACTOR_EARLY_HALF_BARRIER,
            number_of_threads=_WARP_GROUP_SIZE,
        )
    elif cutlass.const_expr(use_warp_leader_handoff):
        if thread_in_group % cutlass.Int32(32) == cutlass.Int32(0):
            factor_stream_ready_handoff.consumer_wait(
                pipeline.PipelineState(
                    _FACTOR_EARLY_HALF_STAGES,
                    factor_chunk,
                    cutlass.Int32(0),
                    factor_chunk % cutlass.Int32(2),
                ),
            )
        cute.arch.sync_warp()
    else:
        factor_stream_ready_handoff.consumer_wait(
            pipeline.PipelineState(
                _FACTOR_EARLY_HALF_STAGES,
                factor_chunk,
                cutlass.Int32(0),
                factor_chunk % cutlass.Int32(2),
            ),
        )
    for local_tile in cutlass.range_constexpr(2):
        factor_mma.set(
            warpgroup.Field.ACCUMULATE,
            local_tile != 0,
        )
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, local_tile)],
            k_fragment[(None, None, local_tile)],
            accumulator,
        )
    for local_tile in cutlass.range_constexpr(2):
        key_tile = local_tile + _QKB_STREAM_TILES
        factor_mma.set(warpgroup.Field.ACCUMULATE, True)
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, key_tile)],
            k_fragment[(None, None, key_tile)],
            accumulator,
        )

    # The existing full-ready handoff covers the remaining factor operands.
    # Waiting here allows the first four WGMMA operations to execute while
    # State WG0/WG1 finish their second half and common publication fence.
    factor_ready_handoff.consumer_wait(factor_consumer_state)
    for local_tile in cutlass.range_constexpr(2):
        key_tile = local_tile + 2
        factor_mma.set(warpgroup.Field.ACCUMULATE, True)
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, key_tile)],
            k_fragment[(None, None, key_tile)],
            accumulator,
        )
    for local_tile in cutlass.range_constexpr(2):
        key_tile = local_tile + 2 + _QKB_STREAM_TILES
        factor_mma.set(warpgroup.Field.ACCUMULATE, True)
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
def _sync_factor_qk_slab0_pairwise(
    generation: cutlass.Int32,
    named_ring2: bool,
    arrive_only: bool,
) -> None:
    """Arrive at or wait on the slab-0 pairwise named edge."""

    if cutlass.const_expr(named_ring2):
        if generation % cutlass.Int32(2) == cutlass.Int32(0):
            if cutlass.const_expr(arrive_only):
                cute.arch.barrier_arrive(
                    barrier_id=_FACTOR_EARLY_HALF_BARRIER,
                    number_of_threads=2 * _WARP_GROUP_SIZE,
                )
            else:
                cute.arch.barrier(
                    barrier_id=_FACTOR_EARLY_HALF_BARRIER,
                    number_of_threads=2 * _WARP_GROUP_SIZE,
                )
        else:
            if cutlass.const_expr(arrive_only):
                cute.arch.barrier_arrive(
                    barrier_id=_FACTOR_EARLY_HALF_BARRIER_ALT,
                    number_of_threads=2 * _WARP_GROUP_SIZE,
                )
            else:
                cute.arch.barrier(
                    barrier_id=_FACTOR_EARLY_HALF_BARRIER_ALT,
                    number_of_threads=2 * _WARP_GROUP_SIZE,
                )
    else:
        if cutlass.const_expr(arrive_only):
            cute.arch.barrier_arrive(
                barrier_id=_FACTOR_EARLY_HALF_BARRIER,
                number_of_threads=2 * _WARP_GROUP_SIZE,
            )
        else:
            cute.arch.barrier(
                barrier_id=_FACTOR_EARLY_HALF_BARRIER,
                number_of_threads=2 * _WARP_GROUP_SIZE,
            )


@cute.jit
def _publish_factor_qk_slab0_pairwise_named_half(
    factor_mma: cute.TiledMma,
    thread_in_group: cutlass.Int32,
    q_bar: cute.Tensor,
    k_bar: cute.Tensor,
    aqk: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    factor_ready_handoff: pipeline.PipelineAsync,
    factor_consumer_state: pipeline.PipelineState,
    factor_chunk: cutlass.Int32,
    named_ring2: bool,
) -> None:
    """Issue slab-0 QK after one pairwise barrier, then wait full-ready."""

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

    # State WG1 owns factor key tiles 0..3. Its matching site fences those
    # stores before joining this 256-thread rendezvous. State WG2 is not a
    # participant and continues preparing tiles 4..7 for the full-ready
    # handoff below.
    _sync_factor_qk_slab0_pairwise(
        factor_chunk,
        named_ring2,
        False,
    )
    for key_tile in cutlass.range_constexpr(_QKB_STREAM_TILES):
        factor_mma.set(
            warpgroup.Field.ACCUMULATE,
            key_tile != 0,
        )
        cute.gemm(
            factor_mma,
            accumulator,
            q_fragment[(None, None, key_tile)],
            k_fragment[(None, None, key_tile)],
            accumulator,
        )

    # The unchanged full-ready handoff remains the lifetime fence and covers
    # State WG2's factor operands before the high slab is consumed.
    factor_ready_handoff.consumer_wait(factor_consumer_state)
    for local_tile in cutlass.range_constexpr(_QKB_STREAM_TILES):
        key_tile = local_tile + _QKB_STREAM_TILES
        factor_mma.set(warpgroup.Field.ACCUMULATE, True)
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


class A2K1I4RawGPrefixState:
    """I4-r23 State-local public-raw-g prefix lookahead."""

    value_tile = 128
    state_value_tile = _STATE_VALUE_TILE
    threads_per_cta = _THREADS_PER_CTA
    min_blocks_per_mp = 1

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
        subgroup_prefix: bool = False,
        subgroup_exclusive_carry: bool = False,
        pair_stream_factor_qk: bool = False,
        early_half_stream_factor_qk: bool = False,
        cta_early_half_factor_qk: bool = False,
        leader_early_half_factor_qk: bool = False,
        warp_leader_early_half_factor_qk: bool = False,
        slab0_pairwise_named_half_factor_qk: bool = False,
        slab0_pairwise_named_ring2_factor_qk: bool = False,
        slab0_pairwise_arrive_ring2_factor_qk: bool = False,
        elide_state_common_barrier: bool = False,
        elide_state_iteration_done_barrier: bool = False,
        sequence_wave_rotation: int = 0,
        length_ranked_sequence_order: bool = False,
    ) -> None:
        if (
            int(pair_stream_factor_qk)
            + int(early_half_stream_factor_qk)
            + int(cta_early_half_factor_qk)
            + int(leader_early_half_factor_qk)
            + int(warp_leader_early_half_factor_qk)
            + int(slab0_pairwise_named_half_factor_qk)
            + int(slab0_pairwise_named_ring2_factor_qk)
            + int(slab0_pairwise_arrive_ring2_factor_qk)
            > 1
        ):
            raise ValueError("factor-QK stream schedules are mutually exclusive")
        self.has_initial_state = has_initial_state
        self.store_final_state = store_final_state
        self.subgroup_prefix = subgroup_prefix
        self.subgroup_exclusive_carry = subgroup_exclusive_carry
        self.pair_stream_factor_qk = pair_stream_factor_qk
        self.early_half_stream_factor_qk = early_half_stream_factor_qk
        self.cta_early_half_factor_qk = cta_early_half_factor_qk
        self.leader_early_half_factor_qk = leader_early_half_factor_qk
        self.warp_leader_early_half_factor_qk = warp_leader_early_half_factor_qk
        self.slab0_pairwise_named_half_factor_qk = slab0_pairwise_named_half_factor_qk
        self.slab0_pairwise_named_ring2_factor_qk = slab0_pairwise_named_ring2_factor_qk
        self.slab0_pairwise_arrive_ring2_factor_qk = slab0_pairwise_arrive_ring2_factor_qk
        self.elide_state_common_barrier = elide_state_common_barrier
        self.elide_state_iteration_done_barrier = elide_state_iteration_done_barrier
        if sequence_wave_rotation < 0:
            raise ValueError("sequence-wave rotation must be non-negative")
        if sequence_wave_rotation and length_ranked_sequence_order:
            raise ValueError(
                "fixed rotation and length-ranked sequence order are mutually exclusive",
            )
        self.sequence_wave_rotation = sequence_wave_rotation
        self.length_ranked_sequence_order = length_ranked_sequence_order
        self.stream_factor_qk = (
            pair_stream_factor_qk
            or early_half_stream_factor_qk
            or cta_early_half_factor_qk
            or leader_early_half_factor_qk
            or warp_leader_early_half_factor_qk
            or slab0_pairwise_named_half_factor_qk
            or slab0_pairwise_named_ring2_factor_qk
            or slab0_pairwise_arrive_ring2_factor_qk
        )
        self.factor_stream_stages = _FACTOR_PAIR_STAGES if pair_stream_factor_qk else _FACTOR_EARLY_HALF_STAGES

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        g: cute.Tensor,
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
        g_global = cute.make_tensor(g.iterator, raw_q_layout)
        v_global = cute.make_tensor(v.iterator, raw_v_layout)
        w_global = cute.make_tensor(w.iterator, raw_v_layout)

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
            g_global,
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
            factor_pair_ready_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _FACTOR_PAIR_STAGES,
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
            g,
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
        g: cute.Tensor,
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
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        producer_value_work_by_warp = storage.producer_value_work_by_warp.get_tensor(
            cute.make_layout(
                (_PRODUCER_SIGNAL_WARPS,),
                stride=(1,),
            ),
        )

        value_tiles = cutlass.Int32(VALUE_SIZE // self.value_tile)
        sequence_stride = num_v_heads * value_tiles
        sequence_rank = work_index // sequence_stride
        value_work = work_index - sequence_rank * sequence_stride
        sequence = sequence_rank
        if cutlass.const_expr(self.length_ranked_sequence_order):
            if warp_index == cutlass.Int32(4):
                scheduled_sequence = _stable_lpt32_sequence(
                    cu_seqlens,
                    sequence_rank,
                    num_sequences,
                    thread % cutlass.Int32(32),
                )
                if thread % cutlass.Int32(32) == cutlass.Int32(0):
                    producer_value_work_by_warp[0] = scheduled_sequence
            cute.arch.sync_threads()
            sequence = cute.arch.make_warp_uniform(
                producer_value_work_by_warp[0],
            )
        elif cutlass.const_expr(self.sequence_wave_rotation > 0):
            if num_sequences > cutlass.Int32(self.sequence_wave_rotation):
                sequence = sequence + cutlass.Int32(
                    self.sequence_wave_rotation,
                )
                if sequence >= num_sequences:
                    sequence = sequence - num_sequences
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
        factor_stream_producer_threads = 2 if self.leader_early_half_factor_qk else 2 * _WARP_GROUP_SIZE
        factor_stream_consumer_threads = (
            1
            if self.leader_early_half_factor_qk
            else (_PRODUCER_SIGNAL_WARPS if self.warp_leader_early_half_factor_qk else _WARP_GROUP_SIZE)
        )
        if cutlass.const_expr(
            self.stream_factor_qk
            and not (
                self.slab0_pairwise_named_half_factor_qk
                or self.slab0_pairwise_named_ring2_factor_qk
                or self.slab0_pairwise_arrive_ring2_factor_qk
            ),
        ):
            factor_pair_ready_handoff = pipeline.PipelineAsync.create(
                barrier_storage=(storage.factor_pair_ready_barriers.data_ptr()),
                num_stages=self.factor_stream_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    factor_stream_producer_threads,
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    factor_stream_consumer_threads,
                ),
            )
        else:
            # Non-P9 schedules must not pay the side effects of constructing
            # an unused PipelineAsync: its mbarrier initialization includes a
            # CTA-wide agent_sync even if no later producer/consumer uses it.
            factor_pair_ready_handoff = factor_ready_handoff
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
            producer_sequence_rank = producer_work_index // sequence_stride
            producer_value_work = producer_work_index - producer_sequence_rank * sequence_stride
            producer_sequence = sequence
            if cutlass.const_expr(
                not self.length_ranked_sequence_order and self.sequence_wave_rotation > 0,
            ):
                producer_sequence = producer_sequence_rank
                if num_sequences > cutlass.Int32(
                    self.sequence_wave_rotation,
                ):
                    producer_sequence = producer_sequence + cutlass.Int32(
                        self.sequence_wave_rotation,
                    )
                    if producer_sequence >= num_sequences:
                        producer_sequence = producer_sequence - num_sequences
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
                    g_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        g_tma,
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
                        g_use[None, None, q_head],
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
                    if cutlass.const_expr(
                        not self.stream_factor_qk,
                    ):
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
                    if cutlass.const_expr(
                        self.pair_stream_factor_qk,
                    ):
                        _publish_factor_qk_pair_streamed(
                            factor_mma,
                            thread_in_group,
                            factor_q,
                            factor_k,
                            factor_aqk,
                            factor_valid_tokens,
                            scale,
                            factor_pair_ready_handoff,
                            pipeline_step,
                        )
                    elif cutlass.const_expr(
                        self.slab0_pairwise_named_half_factor_qk
                        or self.slab0_pairwise_named_ring2_factor_qk
                        or self.slab0_pairwise_arrive_ring2_factor_qk,
                    ):
                        _publish_factor_qk_slab0_pairwise_named_half(
                            factor_mma,
                            thread_in_group,
                            factor_q,
                            factor_k,
                            factor_aqk,
                            factor_valid_tokens,
                            scale,
                            factor_ready_handoff,
                            factor_consumer_state,
                            pipeline_step,
                            self.slab0_pairwise_named_ring2_factor_qk or self.slab0_pairwise_arrive_ring2_factor_qk,
                        )
                    elif cutlass.const_expr(
                        self.early_half_stream_factor_qk
                        or self.cta_early_half_factor_qk
                        or self.leader_early_half_factor_qk
                        or self.warp_leader_early_half_factor_qk,
                    ):
                        _publish_factor_qk_early_half_streamed(
                            factor_mma,
                            thread_in_group,
                            factor_q,
                            factor_k,
                            factor_aqk,
                            factor_valid_tokens,
                            scale,
                            factor_pair_ready_handoff,
                            factor_ready_handoff,
                            pipeline_step,
                            factor_consumer_state,
                            self.cta_early_half_factor_qk,
                            self.leader_early_half_factor_qk,
                            self.warp_leader_early_half_factor_qk,
                        )
                    else:
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
                    if cutlass.const_expr(
                        self.pair_stream_factor_qk,
                    ):
                        for local_pair in cutlass.range_constexpr(
                            _FACTOR_PAIR_STAGES,
                        ):
                            pair_iteration = pipeline_step * cutlass.Int32(_FACTOR_PAIR_STAGES) + cutlass.Int32(local_pair)
                            factor_pair_ready_handoff.consumer_release(
                                pipeline.PipelineState(
                                    _FACTOR_PAIR_STAGES,
                                    pair_iteration,
                                    cutlass.Int32(local_pair),
                                    pipeline_step % cutlass.Int32(2),
                                ),
                            )
                        factor_ready_handoff.consumer_wait(
                            factor_consumer_state,
                        )
                    elif cutlass.const_expr(
                        self.early_half_stream_factor_qk,
                    ):
                        factor_pair_ready_handoff.consumer_release(
                            pipeline.PipelineState(
                                _FACTOR_EARLY_HALF_STAGES,
                                pipeline_step,
                                cutlass.Int32(0),
                                pipeline_step % cutlass.Int32(2),
                            ),
                        )
                    elif cutlass.const_expr(
                        self.warp_leader_early_half_factor_qk,
                    ):
                        if thread_in_group % cutlass.Int32(32) == cutlass.Int32(0):
                            factor_pair_ready_handoff.consumer_release(
                                pipeline.PipelineState(
                                    _FACTOR_EARLY_HALF_STAGES,
                                    pipeline_step,
                                    cutlass.Int32(0),
                                    pipeline_step % cutlass.Int32(2),
                                ),
                            )
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    if cutlass.const_expr(
                        self.leader_early_half_factor_qk,
                    ):
                        if thread_in_group == cutlass.Int32(0):
                            factor_pair_ready_handoff.consumer_release(
                                pipeline.PipelineState(
                                    _FACTOR_EARLY_HALF_STAGES,
                                    pipeline_step,
                                    cutlass.Int32(0),
                                    pipeline_step % cutlass.Int32(2),
                                ),
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
                current_chunk = state_step - cutlass.Int32(1)
                factor_done_consumer_state = pipeline.PipelineState(
                    _INPUT_STAGES,
                    current_chunk,
                    current_chunk % cutlass.Int32(_INPUT_STAGES),
                    (current_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2),
                )
                if state_step > cutlass.Int32(0):
                    current_stage = current_chunk % cutlass.Int32(_INPUT_STAGES)
                    current_chunk_start = state_sequence_start + current_chunk * cutlass.Int32(CHUNK_SIZE)
                    current_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if current_chunk + cutlass.Int32(1) == state_sequence_chunks:
                        current_valid_tokens = state_sequence_end - current_chunk_start

                    # Streamed children move this wait below next-factor
                    # preparation so early publication can overlap the
                    # current factor.
                    if cutlass.const_expr(
                        not self.stream_factor_qk,
                    ):
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
                        if cutlass.const_expr(self.subgroup_prefix):
                            # Sixteen independent eight-lane subgroups cover
                            # one 16-channel raw-G tile. Each lane scans one
                            # consecutive eight-token segment, then a
                            # subgroup shuffle scan distributes the carry.
                            subgroup_lane = thread_in_group % cutlass.Int32(8)
                            tile_channel = thread_in_group // cutlass.Int32(8)
                            token_base = subgroup_lane * cutlass.Int32(8)
                            local_prefix = cute.make_rmem_tensor(
                                8,
                                cutlass.Float32,
                            )
                            segment_total = cutlass.Float32(0.0)
                            for local_index in cutlass.range_constexpr(8):
                                local_token = token_base + cutlass.Int32(local_index)
                                if local_token < prepare_valid_tokens:
                                    segment_total = segment_total + cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                local_prefix[local_index] = segment_total

                            inclusive_segment_total = segment_total
                            for log_offset in cutlass.range_constexpr(3):
                                offset = 1 << log_offset
                                prior = cute.arch.shuffle_sync_up(
                                    inclusive_segment_total,
                                    offset,
                                    mask_and_clamp=0,
                                )
                                if subgroup_lane >= cutlass.Int32(offset):
                                    inclusive_segment_total = inclusive_segment_total + prior
                            if cutlass.const_expr(
                                self.subgroup_exclusive_carry,
                            ):
                                carry = inclusive_segment_total - segment_total
                            else:
                                carry = cutlass.Float32(0.0)
                                prior_segment_total = cute.arch.shuffle_sync_up(
                                    inclusive_segment_total,
                                    1,
                                    mask_and_clamp=0,
                                )
                                if subgroup_lane > cutlass.Int32(0):
                                    carry = prior_segment_total

                            for local_index in cutlass.range_constexpr(8):
                                local_token = token_base + cutlass.Int32(local_index)
                                if local_token < prepare_valid_tokens:
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ] = local_prefix[local_index] + carry
                                else:
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ] = cutlass.Float32(0.0)
                        elif thread_in_group < cutlass.Int32(_WGMMA_K):
                            prefix = cutlass.Float32(0.0)
                            for local_token in cutlass.range_constexpr(
                                CHUNK_SIZE,
                            ):
                                if cutlass.Int32(local_token) < prepare_valid_tokens:
                                    prefix = prefix + cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            thread_in_group,
                                            raw_stage,
                                        ],
                                    )
                                    raw_g[
                                        local_token,
                                        thread_in_group,
                                        raw_stage,
                                    ] = prefix
                                else:
                                    raw_g[
                                        local_token,
                                        thread_in_group,
                                        raw_stage,
                                    ] = cutlass.Float32(0.0)
                        if state_slab == cutlass.Int32(0):
                            cute.arch.barrier(
                                barrier_id=_STATE0_PREFIX_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                        else:
                            cute.arch.barrier(
                                barrier_id=_STATE1_PREFIX_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                        if cutlass.const_expr(
                            self.pair_stream_factor_qk,
                        ):
                            factor_pair_ready_handoff.producer_acquire(
                                pipeline.PipelineState(
                                    _FACTOR_PAIR_STAGES,
                                    prepare_chunk
                                    * cutlass.Int32(
                                        _FACTOR_PAIR_STAGES,
                                    )
                                    + local_key_tile,
                                    local_key_tile,
                                    cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                ),
                            )
                        elif cutlass.const_expr(
                            self.early_half_stream_factor_qk or self.warp_leader_early_half_factor_qk,
                        ):
                            if local_key_tile == cutlass.Int32(0):
                                factor_pair_ready_handoff.producer_acquire(
                                    pipeline.PipelineState(
                                        _FACTOR_EARLY_HALF_STAGES,
                                        prepare_chunk,
                                        cutlass.Int32(0),
                                        cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                    ),
                                )
                        elif cutlass.const_expr(
                            self.leader_early_half_factor_qk,
                        ):
                            if local_key_tile == cutlass.Int32(0) and thread_in_group == cutlass.Int32(0):
                                factor_pair_ready_handoff.producer_acquire(
                                    pipeline.PipelineState(
                                        _FACTOR_EARLY_HALF_STAGES,
                                        prepare_chunk,
                                        cutlass.Int32(0),
                                        cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                    ),
                                )
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
                        if cutlass.const_expr(
                            self.pair_stream_factor_qk,
                        ):
                            cute.arch.fence_proxy(
                                "async.shared",
                                space="cta",
                            )
                            factor_pair_ready_handoff.producer_commit(
                                pipeline.PipelineState(
                                    _FACTOR_PAIR_STAGES,
                                    prepare_chunk
                                    * cutlass.Int32(
                                        _FACTOR_PAIR_STAGES,
                                    )
                                    + local_key_tile,
                                    local_key_tile,
                                    cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                ),
                            )
                        elif cutlass.const_expr(
                            self.early_half_stream_factor_qk or self.warp_leader_early_half_factor_qk,
                        ):
                            if local_key_tile == cutlass.Int32(1):
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                factor_pair_ready_handoff.producer_commit(
                                    pipeline.PipelineState(
                                        _FACTOR_EARLY_HALF_STAGES,
                                        prepare_chunk,
                                        cutlass.Int32(0),
                                        cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                    ),
                                )
                        elif cutlass.const_expr(
                            self.cta_early_half_factor_qk,
                        ):
                            if local_key_tile == cutlass.Int32(1):
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                cute.arch.barrier(
                                    barrier_id=_FACTOR_EARLY_HALF_BARRIER,
                                    number_of_threads=_THREADS_PER_CTA,
                                )
                        elif cutlass.const_expr(
                            self.leader_early_half_factor_qk,
                        ):
                            if local_key_tile == cutlass.Int32(1):
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                if state_slab == cutlass.Int32(0):
                                    cute.arch.barrier(
                                        barrier_id=_STATE0_PREFIX_BARRIER,
                                        number_of_threads=_WARP_GROUP_SIZE,
                                    )
                                else:
                                    cute.arch.barrier(
                                        barrier_id=_STATE1_PREFIX_BARRIER,
                                        number_of_threads=_WARP_GROUP_SIZE,
                                    )
                                if thread_in_group == cutlass.Int32(0):
                                    factor_pair_ready_handoff.producer_commit(
                                        pipeline.PipelineState(
                                            _FACTOR_EARLY_HALF_STAGES,
                                            prepare_chunk,
                                            cutlass.Int32(0),
                                            cutlass.Int32(1) - prepare_chunk % cutlass.Int32(2),
                                        ),
                                    )

                        if state_slab == cutlass.Int32(0):
                            qkb0_pipeline.consumer_release(qkb_consumer)
                        else:
                            qkb1_pipeline.consumer_release(qkb_consumer)
                        qkb_consumer.advance()

                    if cutlass.const_expr(
                        self.slab0_pairwise_named_half_factor_qk
                        or self.slab0_pairwise_named_ring2_factor_qk
                        or self.slab0_pairwise_arrive_ring2_factor_qk,
                    ):
                        if state_slab == cutlass.Int32(0):
                            cute.arch.fence_proxy(
                                "async.shared",
                                space="cta",
                            )
                            _sync_factor_qk_slab0_pairwise(
                                prepare_chunk,
                                self.slab0_pairwise_named_ring2_factor_qk or self.slab0_pairwise_arrive_ring2_factor_qk,
                                self.slab0_pairwise_arrive_ring2_factor_qk,
                            )

                    raw_handoff.consumer_release(
                        pipeline.PipelineState(
                            1,
                            prepare_chunk,
                            cutlass.Int32(0),
                            prepare_chunk % cutlass.Int32(2),
                        ),
                    )

                    # Both State WGs publish disjoint halves of the common
                    # factor operands. The R30 child uses the existing
                    # 256-producer factor-ready mbarrier as the only
                    # rendezvous: each producer fences its preceding stores
                    # before the release arrive, and Factor WG0 returns from
                    # the acquire wait only after all 256 arrivals.
                    if cutlass.const_expr(
                        not self.elide_state_common_barrier,
                    ):
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
                    if cutlass.const_expr(
                        self.stream_factor_qk,
                    ):
                        factor_done_handoff.consumer_wait(
                            factor_done_consumer_state,
                        )
                        factor_done_handoff.consumer_release(
                            factor_done_consumer_state,
                        )
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
                    if cutlass.const_expr(
                        not self.elide_state_iteration_done_barrier,
                    ):
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


class A2K1I4RawGSubgroupPrefixState(A2K1I4RawGPrefixState):
    """I4-r24r1 eight-lane subgroup raw-G prefix child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
        )


class A2K1I4RawGSubgroupExclusiveCarryState(
    A2K1I4RawGPrefixState,
):
    """I4-r24r2 subgroup prefix with subtraction-derived carry."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
        )


class A2K1I4RawGPairStreamedFactorState(
    A2K1I4RawGPrefixState,
):
    """I4-r25 pair-streamed factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            pair_stream_factor_qk=True,
        )


class A2K1I4RawGEarlyHalfStreamedFactorState(
    A2K1I4RawGPrefixState,
):
    """I4-r26 single-early-half factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            early_half_stream_factor_qk=True,
        )


class A2K1I4RawGCtaEarlyHalfFactorState(
    A2K1I4RawGPrefixState,
):
    """I4-r27 CTA-barrier early-half factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            cta_early_half_factor_qk=True,
        )


class A2K1I4RawGLeaderEarlyHalfFactorState(
    A2K1I4RawGPrefixState,
):
    """I4-r28 leader-only early-half factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            leader_early_half_factor_qk=True,
        )


class A2K1I4RawGWarpLeaderEarlyHalfFactorState(
    A2K1I4RawGPrefixState,
):
    """I4-r29 four-warp-leader early-half factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            warp_leader_early_half_factor_qk=True,
        )


class A2K1I4RawGFullReadyMbarrierOnlyState(
    A2K1I4RawGPrefixState,
):
    """I4-r30 full-ready-mbarrier-only publication child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            elide_state_common_barrier=True,
        )


class A2K1I4RawGFactorReadyLockstepOnlyState(
    A2K1I4RawGPrefixState,
):
    """I4-r31 factor-ready/output-handoff lockstep child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
        )


class A2K1I4RawGSlab0PairwiseNamedHalfState(
    A2K1I4RawGPrefixState,
):
    """I4-r32 low-slab pairwise named-half factor-QK child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            slab0_pairwise_named_half_factor_qk=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
        )


class A2K1I4RawGSlab0PairwiseNamedRing2State(
    A2K1I4RawGPrefixState,
):
    """I4-r33 low-slab pairwise two-named-barrier-ring child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            slab0_pairwise_named_ring2_factor_qk=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
        )


class A2K1I4RawGSlab0PairwiseArriveRing2State(
    A2K1I4RawGPrefixState,
):
    """I4-r34 low-slab nonblocking pairwise arrive-ring2 child."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            slab0_pairwise_arrive_ring2_factor_qk=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
        )


class A2K1I4RawGSequenceWaveRotate12State(
    A2K1I4RawGPrefixState,
):
    """I4-r35 sequence-wave rotation child of R31."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
            sequence_wave_rotation=12,
        )


class A2K1I4RawGStableLpt32State(
    A2K1I4RawGPrefixState,
):
    """R36 stable descending sequence-chunk schedule for N at most 32."""

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
    ) -> None:
        super().__init__(
            has_initial_state=has_initial_state,
            store_final_state=store_final_state,
            subgroup_prefix=True,
            subgroup_exclusive_carry=True,
            elide_state_common_barrier=True,
            elide_state_iteration_done_barrier=True,
            length_ranked_sequence_order=True,
        )

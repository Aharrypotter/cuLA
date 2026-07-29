# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Candidate M channel-wise GDN2 transform stage."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .config import CHUNK_SIZE, HEAD_SIZE, THREADS_PER_CTA

_INV_LN2 = 1.4426950408889634


class VectorTransform:
    """Publish the transformed operands consumed by the two WGMMA stages."""

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        g: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cumulative_log: cute.Tensor,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        pseudo_value: cute.Tensor,
        valid_tokens: cutlass.Int32,
        num_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            q,
            k,
            v,
            g,
            b,
            w,
            cumulative_log,
            q_bar,
            k_bar,
            e_bar,
            pseudo_value,
            valid_tokens,
        ).launch(
            grid=(num_heads, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        g: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cumulative_log: cute.Tensor,
        q_bar: cute.Tensor,
        k_bar: cute.Tensor,
        e_bar: cute.Tensor,
        pseudo_value: cute.Tensor,
        valid_tokens: cutlass.Int32,
    ) -> None:
        channel, _, _ = cute.arch.thread_idx()
        head, _, _ = cute.arch.block_idx()
        prefix = cutlass.Float32(0.0)

        for token in cutlass.range_constexpr(CHUNK_SIZE):
            if token < valid_tokens:
                prefix = prefix + cutlass.Float32(g[token, head, channel])
                gamma = cute.math.exp2(
                    prefix * cutlass.Float32(_INV_LN2),
                    fastmath=True,
                )
                gamma_inverse = cute.math.exp2(
                    -prefix * cutlass.Float32(_INV_LN2),
                    fastmath=True,
                )
                q_value = cutlass.Float32(q[token, head, channel])
                k_value = cutlass.Float32(k[token, head, channel])
                v_value = cutlass.Float32(v[token, head, channel])
                b_value = cutlass.Float32(b[token, head, channel])
                w_value = cutlass.Float32(w[token, head, channel])
                cumulative_log[head, token, channel] = prefix
                q_bar[head, token, channel] = cutlass.BFloat16(q_value * gamma)
                k_bar[head, token, channel] = cutlass.BFloat16(k_value * gamma_inverse)
                e_bar[head, token, channel] = cutlass.BFloat16(b_value * k_value * gamma)
                pseudo_value[head, token, channel] = cutlass.BFloat16(w_value * v_value)
            else:
                cumulative_log[head, token, channel] = cutlass.Float32(0.0)
                q_bar[head, token, channel] = cutlass.BFloat16(0.0)
                k_bar[head, token, channel] = cutlass.BFloat16(0.0)
                e_bar[head, token, channel] = cutlass.BFloat16(0.0)
                pseudo_value[head, token, channel] = cutlass.BFloat16(0.0)

        if cutlass.const_expr(THREADS_PER_CTA != HEAD_SIZE):
            cute.arch.sync_threads()

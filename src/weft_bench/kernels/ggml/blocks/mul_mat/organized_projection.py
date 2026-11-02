"""Reusable decoded fields, numerical groups and output-state scopes for K blocks."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.integrations import ggml_formats
from ..quantize import q8_k


@triton.jit
def prepare(tile, Q6: tl.constexpr, SHARED: tl.constexpr):
    weight = tl.load(tile)
    if Q6:
        codes = (tl.i8(weight.ql) | (tl.i8(weight.qh) << tl.u8(4))) - tl.i8(32)
        scales = tl.i32(weight.scales)
        minima, minimum_scale = scales * 0, tl.f32(0)
    else:
        codes, scales = weight.q, tl.i32(weight.sc)
        minima, minimum_scale = tl.i32(weight.m), tl.f32(weight.dmin)
    scale = tl.f32(weight.d)
    if SHARED:
        codes, scales, scale = tl.stage(codes), tl.stage(scales), tl.stage(scale)
        minima, minimum_scale = tl.stage(minima), tl.stage(minimum_scale)
    return codes, scales, scale, minima, minimum_scale


@triton.jit
def calculate(prepared, activation, Q6: tl.constexpr, GROUPED: tl.constexpr):
    codes, scales, scale, minima, minimum_scale = prepared
    coordinate = tl.arange(0, 64, dtype=tl.u32, axis="K")
    if GROUPED:
        partial = tl.widen(tl.mac_groups(activation.q, codes, n=4, into=tl.i16), tl.i32)
    else:
        partial = tl.mac_groups(activation.q, codes, n=4, into=tl.i32)
    group_size: tl.constexpr = 4 if Q6 else 8
    integer = tl.reduce(partial * scales[:, coordinate // group_size], axis="K")
    value = activation.ds * scale * tl.f32(integer)
    if not Q6:
        correction = tl.dot(tl.sum_pairs(activation.bsum), minima,
                            over="K", acc_dtype=tl.i32, engine="vector")
        value -= activation.ds * minimum_scale * tl.f32(correction)
    return value


@triton.jit
def update_outputs(W, Xq, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   MR: tl.constexpr, NR: tl.constexpr,
                   SHARED: tl.constexpr, GROUPED: tl.constexpr, Q6: tl.constexpr):
    for m in range(M):
        for n in range(N):
            tl.store(Y[m, n], tl.f32(0))
    with tl.level.cols(N, group=NR) as columns:
        with tl.level.blocks(K, extent=256) as block:
            if SHARED:
                prepared = prepare(W[columns, block], Q6, SHARED)
            with tl.level.rows(M, group=MR) as rows:
                if not SHARED:
                    prepared = prepare(W[columns, block], Q6, SHARED)
                value = calculate(prepared, tl.load(Xq[rows, block]), Q6, GROUPED)
                tl.store(Y[rows, columns], tl.load(Y[rows, columns]) + value)


@triton.jit
def retain_outputs(W, Xq, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   MR: tl.constexpr, NR: tl.constexpr, MC: tl.constexpr,
                   SHARED: tl.constexpr, GROUPED: tl.constexpr, Q6: tl.constexpr):
    with tl.level.rows(M, group=MC) as cohort:
        with tl.level.cols(N, group=NR) as columns:
            accumulator = tl.state(tl.f32, [MC, NR], init=0)
            with tl.level.blocks(K, extent=256) as block:
                if SHARED:
                    prepared = prepare(W[columns, block], Q6, SHARED)
                with tl.level.rows(cohort, group=MR) as rows:
                    if not SHARED:
                        prepared = prepare(W[columns, block], Q6, SHARED)
                    value = calculate(prepared, tl.load(Xq[rows, block]), Q6, GROUPED)
                    if MC == MR:
                        accumulator += value
                    else:
                        accumulator[rows] = accumulator[rows] + value
            tl.store(Y[cohort, columns], accumulator)


@triton.jit
def projection(W, X, Xq, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               MR: tl.constexpr, NR: tl.constexpr, MC: tl.constexpr,
               SHARED: tl.constexpr, GROUPED: tl.constexpr,
               RETAIN_OUTPUT: tl.constexpr, Q6: tl.constexpr):
    q8_k.quantize_matrix(X, Xq)
    if RETAIN_OUTPUT:
        retain_outputs(W, Xq, Y, M, N, K, MR, NR, MC, SHARED, GROUPED, Q6)
    else:
        update_outputs(W, Xq, Y, M, N, K, MR, NR, SHARED, GROUPED, Q6)


@triton.jit(model="weft", alias_groups={"W": "weights", "X": "input", "Xq": "workspace", "Y": "output"})
def q4_projection(W: tl.View[ggml_formats.Q4_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
                  MR: tl.constexpr, NR: tl.constexpr, SHARED: tl.constexpr,
                  GROUPED: tl.constexpr, RETAIN_OUTPUT: tl.constexpr = False,
                  MC: tl.constexpr = 16):
    projection(W, X, Xq, Y, M, N, K, MR, NR, MC, SHARED, GROUPED, RETAIN_OUTPUT, False)


@triton.jit(model="weft", alias_groups={"W": "weights", "X": "input", "Xq": "workspace", "Y": "output"})
def q6_projection(W: tl.View[ggml_formats.Q6_K, (N, K)], X: tl.View[tl.f32, (M, K)],
                  Xq: tl.View[ggml_formats.Q8_K, (M, K)], Y: tl.View[tl.f32, (M, N)],
                  MR: tl.constexpr, NR: tl.constexpr, SHARED: tl.constexpr,
                  GROUPED: tl.constexpr, RETAIN_OUTPUT: tl.constexpr = False,
                  MC: tl.constexpr = 16):
    projection(W, X, Xq, Y, M, N, K, MR, NR, MC, SHARED, GROUPED, RETAIN_OUTPUT, True)

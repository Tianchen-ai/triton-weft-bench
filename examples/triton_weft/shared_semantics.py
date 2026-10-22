"""Shared Triton numerical semantics with fixed View IO.

These kernels exercise the common frontend. They do not establish support for
ordinary pointer parameters or masked pointer accesses.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _ordinary_add(left, right):
    return left + right


@triton.jit(model="weft")
def fixed_arange(X: tl.View[tl.f32, (128,)], Y: tl.View[tl.f32, (128,)]):
    values = tl.load(X)
    offsets = tl.arange(0, 128)
    tl.store(Y, _ordinary_add(values, offsets.to(tl.float32)))


@triton.jit(model="weft")
def fixed_sum(X: tl.View[tl.f32, (4, 128)], Y: tl.View[tl.f32, (4,)]):
    values = tl.load(X)
    tl.store(Y, tl.sum(values, axis=1))


@triton.jit(model="weft")
def fixed_max(X: tl.View[tl.f32, (4, 128)], Y: tl.View[tl.f32, (4,)]):
    values = tl.load(X)
    tl.store(Y, tl.max(values, axis=1))


@triton.jit(model="weft")
def fixed_softmax(X: tl.View[tl.f32, (4, 128)], Y: tl.View[tl.f32, (4, 128)]):
    values = tl.load(X)
    numerator = tl.exp(values - tl.max(values, axis=1)[:, None])
    tl.store(Y, numerator / tl.sum(numerator, axis=1)[:, None])


@triton.jit(model="weft")
def fixed_dot_acc(A: tl.View[tl.f32, (16, 32)], B: tl.View[tl.f32, (32, 16)],
                  Acc: tl.View[tl.f32, (16, 16)], Y: tl.View[tl.f32, (16, 16)]):
    a = tl.load(A)
    b = tl.load(B)
    accumulator = tl.load(Acc)
    tl.store(Y, tl.dot(a, b, accumulator))


@triton.jit(model="weft")
def fixed_batch_dot_acc(A: tl.View[tl.f32, (2, 16, 32)], B: tl.View[tl.f32, (2, 32, 16)],
                        Acc: tl.View[tl.f32, (2, 16, 16)], Y: tl.View[tl.f32, (2, 16, 16)]):
    a = tl.load(A)
    b = tl.load(B)
    accumulator = tl.load(Acc)
    tl.store(Y, tl.dot(a, b, accumulator))


@triton.jit(model="weft")
def fixed_for_while(X: tl.View[tl.f32, (128,)], Y: tl.View[tl.f32, (128,)]):
    value = tl.load(X)
    for i in range(3):
        value = value + 1.0
    index = 0
    while index < 2:
        value = value * 0.5
        index += 1
    tl.store(Y, value)


@triton.jit(model="weft")
def fixed_nested_level(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)]):
    with tl.level.rows(M, group=2) as rows:
        with tl.level.cols(N, group=32) as columns:
            values = tl.load(X[rows, columns])
            tl.store(Y[rows, columns], values * 1.25 - 0.5)

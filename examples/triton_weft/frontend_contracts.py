"""Scalar SSA and invalid scope examples for an explicit verification plan."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def mixed_scalar_ssa(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    total = tl.f32(0.0)
    for index in range(3):
        if index == 1:
            total = total + 1
        else:
            total = total + tl.f32(1.0)
    with tl.level.blocks(N, extent=32) as part:
        tl.store(Y[part], (tl.load(X[part]) + total) * 0.25)


@triton.jit(model="weft")
def loop_iv_escape(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    for index in range(3):
        value = tl.load(X[index])
    tl.store(Y[0], tl.f32(index))


@triton.jit(model="weft")
def level_point_escape(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    with tl.level.rows(N, group=1) as point:
        value = tl.load(X[point])
    tl.store(Y[point], value)

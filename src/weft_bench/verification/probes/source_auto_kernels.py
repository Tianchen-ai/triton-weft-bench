"""Binding-only gaps not represented by the public parameterized examples."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"))
def unbound(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)]):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto("B")) as block:
            tl.store(Y[row, block], tl.load(X[row, block]))


@triton.jit(model="weft", strided=("X", "Y"))
def finite_domain(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)]):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto(32, 64)) as block:
            tl.store(Y[row, block], tl.load(X[row, block]))

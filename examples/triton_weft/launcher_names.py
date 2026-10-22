"""Ordinary user names must remain distinct from generated launcher identifiers."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def linear(linear: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], extra_0: tl.i64):
    for i in range(extra_0):
        tl.store(Y[i], (tl.load(linear[i]) + 3.0) * 0.25)


@triton.jit(model="weft")
def entry(entry: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], extra_0: tl.i64):
    for i in range(extra_0):
        tl.store(Y[i], (tl.load(entry[i]) + 3.0) * 0.25)


@triton.jit(model="weft")
def instance(instance: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], extra_0: tl.i64):
    for i in range(extra_0):
        tl.store(Y[i], (tl.load(instance[i]) + 3.0) * 0.25)

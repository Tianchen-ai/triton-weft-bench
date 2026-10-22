"""One native kernel observes the execution thread's floating-point mode."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def dynamic_rounding(X32: tl.View[tl.f32, (N,)], X64: tl.View[tl.f64, (N,)],
                     Y32: tl.View[tl.i8, (N,)], Y64: tl.View[tl.i8, (N,)]):
    with tl.level.blocks(N, extent=16) as columns:
        a = tl.narrow(tl.load(X32[columns]), tl.i8, rounding="dynamic", saturation=True)
        b = tl.narrow(tl.load(X64[columns]), tl.i8, rounding="dynamic", saturation=True)
        tl.store(Y32[columns], a)
        tl.store(Y64[columns], b)

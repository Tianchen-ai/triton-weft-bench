from __future__ import annotations

import triton.integrations.ggml_formats as ggml
from ....functions import quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft")
def row_dequantize_q4_1(W: wl.View[ggml.Q4_1, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        wl.store(Y[kb], qf.min_affine(w.q, w.d, w.m))

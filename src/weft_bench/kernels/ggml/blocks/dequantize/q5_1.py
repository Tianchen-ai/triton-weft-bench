from __future__ import annotations

import triton.integrations.ggml_formats as ggml
import weft_bench.kernels.functions.quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft")
def row_dequantize_q5_1(W: wl.View[ggml.Q5_1, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        q = wl.i32(wl.u8(w.q) | wl.u8(w.qh) << wl.u8(4))
        wl.store(Y[kb], qf.min_affine(q, w.d, w.m))

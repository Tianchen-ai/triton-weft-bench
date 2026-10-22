from __future__ import annotations

import triton.integrations.ggml_formats as ggml
import weft_bench.kernels.functions.quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft")
def row_dequantize_tq2_0(W: wl.View[ggml.TQ2_0, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        q = wl.i32(wl.i8(w.q) - wl.i8(1))
        wl.store(Y[kb], qf.ternary_radix(q, w.d))

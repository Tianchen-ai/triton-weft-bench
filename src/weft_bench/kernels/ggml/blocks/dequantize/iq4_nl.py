from __future__ import annotations

import triton.integrations.ggml_formats as ggml
import weft_bench.kernels.functions.quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft")
def row_dequantize_iq4_nl(
    W: wl.View[ggml.IQ4_NL, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (K,)],
):
    table = wl.stage(wl.load(codebook))
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        q = qf.small_nonlinear_lookup(table, w.q)
        wl.store(Y[kb], qf.fp4_codebook(q, w.d))

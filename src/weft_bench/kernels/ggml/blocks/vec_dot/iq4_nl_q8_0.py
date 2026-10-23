from __future__ import annotations

import triton.integrations.ggml_formats as ggml
from ....functions import quantization as qf
import triton
import triton.language as wl


@triton.jit(model="weft")
def compute(
    W: wl.View[ggml.IQ4_NL, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    DEFERRED: wl.constexpr = False,
):
    table = wl.stage(wl.load(codebook))
    result = wl.f32(0.0)
    if DEFERRED:
        lane = wl.arange(0, 16, dtype=wl.u16, axis="lane")
        result = wl.f32(lane) * wl.f32(0)
    with wl.level.blocks(K, extent=32) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        if DEFERRED:
            # The fixed GGML IQ4_NL codebook has |entry| <= 127, so the
            # two signed-byte products fit i16: 2 * 127 * 128 = 32512.
            first = wl.lookup(table, wl.u8(w.q[lane]), bounds="in_bounds")
            second = wl.lookup(table, wl.u8(w.q[lane + wl.u16(16)]), bounds="in_bounds")
            integer = wl.i16(first) * wl.i16(x.q[lane])
            integer += wl.i16(second) * wl.i16(x.q[lane + wl.u16(16)])
        else:
            integer = qf.dot_codebook32(w.q, x.q, table)
        result += wl.f32(x.d) * wl.f32(w.d) * wl.f32(integer)
    if DEFERRED:
        return wl.reduce(result, axis="lane")
    return result


@triton.jit(model="weft")
def quantized_vec_dot_iq4_nl_q8_0(
    W: wl.View[ggml.IQ4_NL, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (1,)],
    *, DEFERRED: wl.constexpr = False,
):
    wl.store(Y[0], compute(W, X, codebook, DEFERRED))

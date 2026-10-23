from __future__ import annotations

import triton.integrations.ggml_formats as ggml
import triton
import triton.language as wl


@triton.jit
def compute(W: wl.View[ggml.Q4_K, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        group = wl.arange(0, 8, dtype=wl.u8, axis="group")
        lane = wl.arange(0, 32, dtype=wl.u8, axis="k")
        coordinate = group * wl.u8(32) + lane
        partial = wl.reduce_dot(
            w.q[coordinate], x.q[coordinate], over="k", acc_dtype=wl.i32
        )
        total = wl.reduce(partial * wl.i32(w.sc[group]), axis="group")
        min_group = wl.arange(0, 8, dtype=wl.u32, axis="min_group")
        minimum = wl.reduce(
            wl.widen(w.m[min_group], wl.i32)
            * (
                wl.widen(x.bsum[min_group * wl.u32(2)], wl.i32)
                + wl.widen(x.bsum[min_group * wl.u32(2) + wl.u32(1)], wl.i32)
            ),
            axis="min_group",
        )
        # Preserve GGML's block scale/update order before the next quantized layer.
        result -= (wl.f32(x.ds) * wl.f32(w.dmin)) * wl.f32(minimum)
        result += (wl.f32(x.ds) * wl.f32(w.d)) * wl.f32(total)
    return result


@triton.jit(model="weft")
def quantized_vec_dot_q4_k_q8_k(
    W: wl.View[ggml.Q4_K, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.store(Y[0], compute(W, X))


@triton.jit(model="weft")
def quantized_rows_q4_k_q8_k(
    W: wl.View[ggml.Q4_K, (rows, K)], X: wl.View[ggml.Q8_K, (K,)],
    Y: wl.View[wl.f32, (rows,)], BLOCK_ROWS: wl.constexpr = 4
):
    with wl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = wl.state(wl.f32, [BLOCK_ROWS], init=0)
        with wl.level.blocks(K, extent=256) as kb:
            w, x = wl.load(W[rb, kb]), wl.load(X[kb])
            group = wl.arange(0, 8, dtype=wl.u8, axis="group")
            lane = wl.arange(0, 32, dtype=wl.u8, axis="k")
            coordinate = group * wl.u8(32) + lane
            partial = wl.reduce_dot(w.q[:, coordinate], x.q[coordinate], over="k", acc_dtype=wl.i32)
            total = wl.reduce(partial * wl.i32(w.sc[:, group]), axis="group")
            minimum_group = wl.arange(0, 8, dtype=wl.u32, axis="minimum_group")
            minimum = wl.reduce(wl.i32(w.m[:, minimum_group]) * (
                wl.i32(x.bsum[minimum_group * wl.u32(2)]) +
                wl.i32(x.bsum[minimum_group * wl.u32(2) + wl.u32(1)])), axis="minimum_group")
            result -= (wl.f32(x.ds) * wl.f32(w.dmin)) * wl.f32(minimum)
            result += (wl.f32(x.ds) * wl.f32(w.d)) * wl.f32(total)
        wl.store(Y[rb], result)


@triton.jit(model="weft")
def quantized_rows_window_q4_k_q8_k(
    W: wl.View[ggml.Q4_K, (rows, K)], X: wl.View[ggml.Q8_K, (K,)],
    Y: wl.View[wl.f32, (rows,)], BLOCK_ROWS: wl.constexpr = 2,
    WINDOW_GROUPS: wl.constexpr = 2,
):
    """Bound the shared activation and wide products to a few scale groups."""
    wl.static_assert(WINDOW_GROUPS == 1 or WINDOW_GROUPS == 2 or WINDOW_GROUPS == 4 or WINDOW_GROUPS == 8)
    group = wl.arange(0, WINDOW_GROUPS, dtype=wl.u8, axis="group")
    lane = wl.arange(0, 32, dtype=wl.u8, axis="k")
    with wl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = wl.state(wl.f32, [BLOCK_ROWS], init=0)
        with wl.level.blocks(K, extent=256) as kb:
            w, x = wl.load(W[rb, kb]), wl.load(X[kb])
            for window in wl.static_range(8 // WINDOW_GROUPS):
                g = group + wl.u8(window * WINDOW_GROUPS)
                coordinate = g * wl.u8(32) + lane
                partial = wl.reduce_dot(w.q[:, coordinate], x.q[coordinate], over="k", acc_dtype=wl.i32)
                contribution = wl.reduce(partial * wl.i32(w.sc[:, g]), axis="group")
                if window == 0:
                    total = contribution
                else:
                    total += contribution
            minimum_group = wl.arange(0, 8, dtype=wl.u32, axis="minimum_group")
            minimum = wl.reduce(wl.i32(w.m[:, minimum_group]) * (
                wl.i32(x.bsum[minimum_group * wl.u32(2)]) +
                wl.i32(x.bsum[minimum_group * wl.u32(2) + wl.u32(1)])), axis="minimum_group")
            result -= (wl.f32(x.ds) * wl.f32(w.dmin)) * wl.f32(minimum)
            result += (wl.f32(x.ds) * wl.f32(w.d)) * wl.f32(total)
        wl.store(Y[rb], result)


@triton.jit(model="weft")
def quantized_rows_group_stream_q4_k_q8_k(
    W: wl.View[ggml.Q4_K, (rows, K)], X: wl.View[ggml.Q8_K, (K,)],
    Y: wl.View[wl.f32, (rows,)], BLOCK_ROWS: wl.constexpr = 2
):
    """Reuse one activation group while accumulating independent output rows."""
    lane = wl.arange(0, 32, dtype=wl.u8, axis="k")
    with wl.level.rows(rows, group=BLOCK_ROWS) as rb:
        result = wl.state(wl.f32, [BLOCK_ROWS], init=0)
        with wl.level.blocks(K, extent=256) as kb:
            x = wl.load(X[kb])
            w = wl.load(W[rb, kb])
            total = wl.state(wl.i32, [BLOCK_ROWS], init=0)
            minimum = wl.state(wl.i32, [BLOCK_ROWS], init=0)
            for group in wl.static_range(8):
                coordinate = wl.u8(group * 32) + lane
                activation = x.q[coordinate]
                bsum = wl.i32(x.bsum[group * 2]) + wl.i32(x.bsum[group * 2 + 1])
                for row in wl.static_range(BLOCK_ROWS):
                    if row < wl.extent(W[rb, kb], 0):
                        partial = wl.reduce_dot(w.q[row, coordinate], activation, over="k", acc_dtype=wl.i32)
                        total[row] = total[row] + partial * wl.i32(w.sc[row, group])
                        minimum[row] = minimum[row] + bsum * wl.i32(w.m[row, group])
            for row in wl.static_range(BLOCK_ROWS):
                if row < wl.extent(W[rb, kb], 0):
                    value = result[row]
                    value -= (wl.f32(x.ds) * wl.f32(w.dmin[row])) * wl.f32(minimum[row])
                    value += (wl.f32(x.ds) * wl.f32(w.d[row])) * wl.f32(total[row])
                    result[row] = value
        wl.store(Y[rb], result)

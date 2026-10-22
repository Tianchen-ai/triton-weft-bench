from __future__ import annotations
import triton
import triton.language as wl


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def affine(X: wl.View[wl.f32, (M, N)], Y: wl.View[wl.f32, (M, N)],
           scale: wl.f32 = 1.25, *, bias: wl.f32 = -0.5, BLOCK: wl.constexpr = 64):
    with wl.level.rows(M, group=1) as row:
        with wl.level.blocks(N, extent=wl.auto("BLOCK")) as block:
            wl.store(Y[row, block], wl.load(X[row, block]) * scale + bias)


@triton.jit(model="weft")
def coordinates(Y: wl.View[wl.f32, (A, B, C)]):
    i = wl.program_id(0)
    j = wl.program_id(1)
    k = wl.program_id(2)
    value = i + 10*j + 100*k + 1000*wl.num_programs(0) + 10000*wl.num_programs(1) + 100000*wl.num_programs(2)
    wl.store(Y[i, j, k], wl.f32(value))

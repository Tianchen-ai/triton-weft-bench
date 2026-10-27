from __future__ import annotations
import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def affine(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
           scale: tl.f32 = 1.25, *, bias: tl.f32 = -0.5, BLOCK: tl.constexpr = 64):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=tl.auto("BLOCK")) as block:
            tl.store(Y[row, block], tl.load(X[row, block]) * scale + bias)


@triton.jit(model="weft")
def coordinates(Y: tl.View[tl.f32, (A, B, C)]):
    i = tl.program_id(0)
    j = tl.program_id(1)
    k = tl.program_id(2)
    value = i + 10*j + 100*k + 1000*tl.num_programs(0) + 10000*tl.num_programs(1) + 100000*tl.num_programs(2)
    tl.store(Y[i, j, k], tl.f32(value))

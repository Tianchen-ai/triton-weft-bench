"""Fixed-block variants of the established dense View examples."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit(model="weft", strided=("X", "Y"), alias_groups={"X": "input", "Y": "output"})
def affine_fixed(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                 scale: tl.f32 = 1.25, *, bias: tl.f32 = -0.5):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=64) as block:
            tl.store(Y[row, block], tl.load(X[row, block]) * scale + bias)


@triton.jit(model="weft", strided=("X", "Y"))
def tiled_2d(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
             ROWS: tl.constexpr = 2, COLS: tl.constexpr = 8):
    with tl.level.rows(M, group=ROWS) as row:
        with tl.level.cols(N, group=COLS) as column:
            tl.store(Y[row, column], tl.load(X[row, column]) * 1.25 - 0.5)


@triton.jit(model="weft")
def tail_reductions(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (4,)]):
    with tl.level.blocks(N, extent=64) as block:
        values = tl.load(X[block])
        maximum, index = tl.max(values, axis=0, return_indices=True, return_indices_tie_break_left=True)
        tl.store(Y[0], maximum)
        tl.store(Y[1], tl.sum(values, axis=0))
        tl.store(Y[2], tl.min(values, axis=0))
        tl.store(Y[3], index.to(tl.float32))


@triton.jit(model="weft", strided=("X", "Y"))
def bfloat_tile_copy(X: tl.View[tl.bf16, (M, N)], Y: tl.View[tl.bf16, (M, N)]):
    with tl.level.rows(M, group=32) as row:
        with tl.level.cols(N, group=64) as column:
            tl.store(Y[row, column], tl.load(X[row, column]))


@triton.jit(model="weft")
def tail_atomic_counts(X: tl.View[tl.i32, (N,)], Y):
    with tl.level.blocks(N, extent=64) as block:
        values = tl.load(X[block])
        indices = (values + 1 // values) % 4
        tl.atomic_add(Y + indices, 1)


@triton.jit(model="weft", strided=("X", "Y"))
def same_axis_subtiles(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    with tl.level.blocks(N, extent=16) as block:
        parent_x = X[block]
        parent_y = Y[block]
        with tl.level.subtiles(block, extent=4) as sub:
            panel = tl.stage(tl.load(parent_x[sub]))
            tl.store(parent_y[sub], panel * 1.25 - 0.5)


@triton.jit(model="weft", strided=("X", "Y"))
def subview_extent(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                   Sizes: tl.View[tl.i32, (2,)], start: tl.index):
    source = tl.subview(X, offsets=(start,), extents=(8,))
    destination = tl.subview(Y, offsets=(start,), extents=(8,))
    tl.store(destination, tl.load(source) * 1.25 - 0.5)
    tl.store(Sizes[0], tl.i32(tl.extent(X, axis=0)))
    tl.store(Sizes[1], tl.i32(tl.extent(source, axis=0)))


@triton.jit(model="weft", strided=("X", "Y"))
def subview_branch(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                   Sizes: tl.View[tl.i32, (2,)], start: tl.index):
    if start < 16:
        source = tl.subview(X, offsets=(5,), extents=(8,))
    else:
        source = tl.subview(X, offsets=(21,), extents=(8,))
    destination = tl.subview(Y, offsets=(start,), extents=(8,))
    tl.store(destination, tl.load(source) * 1.25 - 0.5)
    tl.store(Sizes[0], tl.i32(tl.extent(X, axis=0)))
    tl.store(Sizes[1], tl.i32(tl.extent(source, axis=0)))


@triton.jit(model="weft", strided=("State", "Delta"))
def state_update_fixed(State: tl.View[tl.f32, (M, N)], Delta: tl.View[tl.f32, (M, N)]):
    with tl.level.rows(M, group=1) as row:
        with tl.level.blocks(N, extent=8) as block:
            state = tl.load(State[row, block])
            delta = tl.load(Delta[row, block])
            tl.store(State[row, block], state * 0.75 + delta)


@triton.jit(model="weft", strided=("Y",))
def broadcast_snapshot(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N, C)], C: tl.constexpr = 16):
    columns = tl.f32(tl.arange(0, C, axis="C"))
    replacement = tl.f32(tl.arange(0, 8, axis="N")) * 0 + 1000
    with tl.level.rows(N, group=8) as rows:
        saved = tl.load(X[rows])
        tl.store(X[rows], replacement)
        tl.store(Y[rows, :], saved + columns)

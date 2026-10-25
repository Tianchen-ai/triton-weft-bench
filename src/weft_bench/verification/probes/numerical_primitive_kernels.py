"""Public grouped arithmetic and in-bounds lookup through the Triton frontend."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft", alias_groups={"Y": "output"})
def pipelined_scaled_groups(A: tl.View[tl.i8, (B, G, K)],
                            X: tl.View[tl.i8, (B, G, K)],
                            S: tl.View[tl.i8, (B, G)], Y: tl.View[tl.i32, (1,)],
                            G: tl.constexpr, K: tl.constexpr, INITIAL: tl.constexpr):
    total = tl.i32(INITIAL)
    with tl.level.rows(B, group=1) as row:
        partial = tl.reduce_dot(tl.load(A[row]), tl.load(X[row]), over="K", acc_dtype=tl.i32)
        scaled = partial * tl.i32(tl.load(S[row]))
        total += tl.sum(tl.sum(scaled, 1), 0)
    tl.store(Y[0], total)


@triton.jit(model="weft", alias_groups={"A": "input", "B": "input"})
def ime_closed_reduction(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)],
                         C: tl.View[tl.i32, (M, N)],
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         SNAPSHOT: tl.constexpr = False):
    a = tl.load(A)
    b = tl.load(B)
    if SNAPSHOT:
        tl.store(A, a * tl.i8(0))
    c = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine="matrix")
    tl.store(C, c)


@triton.jit(model="weft")
def ime_issue_reduction(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)],
                        C: tl.View[tl.i32, (M, N)],
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        INITIAL: tl.constexpr = 0, SHARED: tl.constexpr = False):
    partial = tl.state(tl.i32, [M, N], init=INITIAL)
    extra = tl.state(tl.i32, [M, N], init=0)
    with tl.level.blocks(K, extent=8) as block:
        a, b = tl.load(A[:, block]), tl.load(B[:, block])
        product = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine="matrix")
        partial += product
        if SHARED:
            extra += product * tl.i32(3)
    tl.store(C, partial + extra)


@triton.jit(model="weft")
def ime_independent_reductions(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)],
                              C: tl.View[tl.i32, (2, M, N)],
                              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    a, b = tl.load(A), tl.load(B)
    first = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine="matrix")
    second = tl.reduce_dot(a + tl.i8(1), b, over="K", acc_dtype=tl.i32, engine="matrix")
    tl.store(C[0], first)
    tl.store(C[1], second)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def byte_plane_fields(X: tl.View[tl.u8, (N,)], Y: tl.View[tl.u8, (N,)],
                      WIDTH: tl.constexpr, STEP: tl.constexpr):
    with tl.level.blocks(N, extent=256) as block:
        lane = tl.arange(0, 256, dtype=tl.u16, axis="N")
        source = (lane // STEP) % (9 - WIDTH)
        destination = ((lane // STEP) // (9 - WIDTH)) % (9 - WIDTH)
        bits = tl.u16(tl.load(X[block])) >> source
        if WIDTH == 1:
            field = tl.u1(bits)
        elif WIDTH == 2:
            field = tl.u2(bits)
        else:
            field = tl.u4(bits)
        tl.store(Y[block], tl.u8(field) << tl.u8(destination))


@triton.jit
def streaming_body(X, Y, N, BLOCK: tl.constexpr = 64):
    with tl.level.blocks(N, extent=BLOCK) as block:
        tl.store(Y[block], tl.load(X[block]), cache_modifier=".cs")


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output"})
def streaming_copy(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], BLOCK: tl.constexpr = 64):
    streaming_body(X, Y, N, BLOCK)


@triton.jit(model="weft")
def streaming_overlap(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    streaming_body(X, Y, N)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def bfloat_contraction(A: tl.View[tl.bf16, (M, K)], B: tl.View[tl.bf16, (N, K)],
                       Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr = 2, BN: tl.constexpr = 16,
                       BK: tl.constexpr = 32):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            total = tl.state(tl.f32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as block:
                total += tl.reduce_dot(tl.load(A[rows, block]), tl.load(B[columns, block]),
                                       over="K", acc_dtype=tl.f32)
            tl.store(Y[rows, columns], total)


@triton.jit
def matrix_axis_body(A, B, Y, M, N, K, ACC: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     LEFT_LAST: tl.constexpr, RIGHT_LAST: tl.constexpr, ENGINE: tl.constexpr):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            total = tl.state(ACC, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as block:
                a, b = tl.load(A[block, rows]), tl.load(B[block, columns])
                if LEFT_LAST:
                    a = tl.reshape(a, shape=(BM, BK), axes=("m", "k"), order=("m", "k"))
                if RIGHT_LAST:
                    b = tl.reshape(b, shape=(BN, BK), axes=("n", "k"), order=("n", "k"))
                total += tl.reduce_dot(a, b, over="K", acc_dtype=ACC, engine=ENGINE)
            tl.store(Y[rows, columns], total)


@triton.jit(model="weft")
def scheduled_contraction_window(A: tl.View[tl.f32, (M, K)], B: tl.View[tl.f32, (K, N)],
                                  Y: tl.View[tl.f32, (M, N)], M: tl.constexpr, N: tl.constexpr,
                                  K: tl.constexpr, REDUCTION_LAST: tl.constexpr, MUTATE: tl.constexpr = False):
    a, b = tl.load(A), tl.load(B)
    if MUTATE:
        tl.store(A, a * tl.f32(0))
    if REDUCTION_LAST:
        b = tl.reshape(b, shape=(N, K), axes=("n", "k"), order=("n", "k"))
    result = tl.reduce_dot(a, b, over="K", acc_dtype=tl.f32, engine="vector")
    tl.store(Y, result)


@triton.jit(model="weft")
def matrix_standard_pointer(A, B, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                            BK: tl.constexpr, ENGINE: tl.constexpr):
    rows, columns, terms = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BK)
    left = tl.load(A + rows[:, None] * K + terms[None, :],
                   (rows[:, None] < M) & (terms[None, :] < K), other=0)
    right = tl.load(B + terms[:, None] * N + columns[None, :],
                    (terms[:, None] < K) & (columns[None, :] < N), other=0)
    result = tl.dot(left, right, engine=ENGINE)
    tl.store(Y + rows[:, None] * N + columns[None, :], result,
             (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit(model="weft")
def matrix_axis_signed(A: tl.View[tl.i8, (K, M)], B: tl.View[tl.i8, (K, N)],
                       Y: tl.View[tl.i32, (M, N)], BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, LEFT_LAST: tl.constexpr, RIGHT_LAST: tl.constexpr,
                       ENGINE: tl.constexpr = "auto"):
    matrix_axis_body(A, B, Y, M, N, K, tl.i32, BM, BN, BK, LEFT_LAST, RIGHT_LAST, ENGINE)


@triton.jit(model="weft")
def matrix_axis_unsigned(A: tl.View[tl.u8, (K, M)], B: tl.View[tl.i8, (K, N)],
                         Y: tl.View[tl.i32, (M, N)], BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr, LEFT_LAST: tl.constexpr, RIGHT_LAST: tl.constexpr,
                         ENGINE: tl.constexpr = "auto"):
    matrix_axis_body(A, B, Y, M, N, K, tl.i32, BM, BN, BK, LEFT_LAST, RIGHT_LAST, ENGINE)


@triton.jit(model="weft")
def matrix_axis_word(A: tl.View[tl.i8, (K, M)], B: tl.View[tl.i16, (K, N)],
                     Y: tl.View[tl.i32, (M, N)], BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, LEFT_LAST: tl.constexpr, RIGHT_LAST: tl.constexpr,
                     ENGINE: tl.constexpr = "auto"):
    matrix_axis_body(A, B, Y, M, N, K, tl.i32, BM, BN, BK, LEFT_LAST, RIGHT_LAST, ENGINE)


@triton.jit(model="weft")
def matrix_axis_bfloat(A: tl.View[tl.bf16, (K, M)], B: tl.View[tl.bf16, (K, N)],
                       Y: tl.View[tl.f32, (M, N)], BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, LEFT_LAST: tl.constexpr, RIGHT_LAST: tl.constexpr):
    matrix_axis_body(A, B, Y, M, N, K, tl.f32, BM, BN, BK, LEFT_LAST, RIGHT_LAST, "auto")


@triton.jit(model="weft", strided=("X", "W", "Y"))
def array_contraction(X: tl.View[tl.f32, (M, K)], W: tl.View[tl.f32, (N, K)],
                      Y: tl.View[tl.f32, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(W), over="k"))


@triton.jit(model="weft", strided=("X", "W", "Y"))
def array_contraction_integer(X: tl.View[tl.i16, (M, K)], W: tl.View[tl.u8, (N, K)],
                              Y: tl.View[tl.i32, (M, N)]):
    tl.store(Y, tl.dot(tl.load(X), tl.load(W), over="k", acc_dtype=tl.i32))


@triton.jit(model="weft")
def array_contraction_batched(X: tl.View[tl.f32, (B, M, K, L)],
                              W: tl.View[tl.f32, (B, N, K, L)],
                              Y: tl.View[tl.f32, (B, M, N)]):
    tl.store(Y, tl.reduce_dot(tl.load(X), tl.load(W), over=("k", "l")))


@triton.jit(model="weft")
def named_standard_reduction(X: tl.View[tl.f32, (M, N)], Scale: tl.View[tl.f32, (M,)],
                             Y: tl.View[tl.f32, (M,)]):
    tl.store(Y, tl.sum(tl.load(X), 1) * tl.load(Scale))


@triton.jit(model="weft")
def pointer_packet_wrap(X, Y, Base):
    lane = tl.arange(0, 64)
    narrow = (lane.to(tl.int8) + Base.to(tl.int8)).to(tl.int8)
    offset = narrow.to(tl.int64) + 128
    value = tl.load(X + offset, lane < 57, other=-13.0)
    tl.store(Y + lane, value)


@triton.jit(model="weft")
def indexed_subbyte(X: tl.View[tl.i8, (256,)], Indices: tl.View[tl.u32, (N,)],
                    Y: tl.View[tl.i32, (N,)], KIND: tl.constexpr):
    original = tl.load(X)
    if KIND == 0:
        value = tl.u2(original)
    elif KIND == 1:
        value = tl.u4(original)
    elif KIND == 2:
        value = tl.i4(original)
    else:
        value = original < 0
    tl.store(X, original + 1)
    for row in range(N):
        index = tl.load(Indices[row])
        tl.store(Y[row], tl.i32(tl.lookup(value, index, bounds="in_bounds")))


@triton.jit(model="weft", strided=("X", "Y", "Flags", "Rows", "Columns", "Wide", "Integer"))
def array_map_reduce(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                     Flags: tl.View[tl.i1, (M, N)], Rows: tl.View[tl.f32, (M, 3)],
                     Columns: tl.View[tl.f32, (N,)], Wide: tl.View[tl.f64, (M,)],
                     Integer: tl.View[tl.i32, (M,)]):
    value = tl.load(X)
    negative = value < 0.0
    transformed = tl.select(negative, value * 0.5, value - 2.0)
    tl.store(Y, transformed)
    tl.store(Flags, negative)
    tl.store(Rows[:, 0], tl.reduce(transformed, op="add", axis="n"))
    tl.store(Rows[:, 1], tl.reduce(transformed, op="max", axis="n"))
    tl.store(Rows[:, 2], tl.reduce(transformed, op="min", axis="n"))
    tl.store(Columns, tl.reduce(transformed, op="add", axis="m"))
    tl.store(Wide, tl.reduce(tl.f64(value), op="add", axis="n"))
    tl.store(Integer, tl.reduce(tl.i32(value * 8.0), op="add", axis="n"))


@triton.jit(model="weft")
def array_integer_absolute(X: tl.View[tl.i64, (N,)], Y: tl.View[tl.i64, (N,)]):
    tl.store(Y, tl.abs(tl.load(X)))


@triton.jit(model="weft")
def array_cost_classes(X: tl.View[tl.f32, (N,)], Integer: tl.View[tl.u32, (N,)],
                       Exp: tl.View[tl.f32, (N,)], Quotient: tl.View[tl.u32, (N,)],
                       DIVISOR: tl.constexpr):
    value = tl.load(X)
    unsigned = tl.load(Integer)
    tl.store(Exp, tl.exp(value * tl.f32(0.125)))
    tl.store(Quotient, unsigned // tl.u32(DIVISOR))


@triton.jit(model="weft")
def array_tail_reduce(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                      Rows: tl.View[tl.f32, (M, 3)], BLOCK: tl.constexpr):
    with tl.level.rows(M, group=2) as row:
        with tl.level.cols(N, group=BLOCK) as column:
            value = tl.load(X[row, column])
            transformed = value * 0.5 - 2.0
            tl.store(Y[row, column], transformed)
            tl.store(Rows[row, 0], tl.reduce(transformed, op="add", axis="n"))
            tl.store(Rows[row, 1], tl.reduce(transformed, op="max", axis="n"))
            tl.store(Rows[row, 2], tl.reduce(transformed, op="min", axis="n"))


@triton.jit(model="weft")
def array_special_reduce(X: tl.View[tl.f32, (M, N)], Rows: tl.View[tl.f32, (M, 3)]):
    value = tl.load(X)
    tl.store(Rows[:, 0], tl.reduce(value, op="add", axis="n"))
    tl.store(Rows[:, 1], tl.reduce(value, op="max", axis="n"))
    tl.store(Rows[:, 2], tl.reduce(value, op="min", axis="n"))


@triton.jit(model="weft")
def fixed_number_extrema(X: tl.View[tl.f32, (N,)], Other: tl.View[tl.f32, (N,)],
                         Pair: tl.View[tl.f32, (N, 4)], Reduced: tl.View[tl.f32, (3,)]):
    with tl.level.blocks(N, extent=17) as block:
        x, other = tl.load(X[block]), tl.load(Other[block])
        tl.store(Pair[block, 0], tl.maximum(x, other, propagate_nan=tl.PropagateNan.NONE))
        tl.store(Pair[block, 1], tl.minimum(x, other, propagate_nan=tl.PropagateNan.NONE))
        tl.store(Pair[block, 2], tl.maximum(x, tl.f32(0.), propagate_nan=tl.PropagateNan.NONE))
        tl.store(Pair[block, 3], tl.minimum(x, tl.f32(-0.), propagate_nan=tl.PropagateNan.NONE))
        tl.store(Reduced[0], tl.reduce(x, op="max"))
        tl.store(Reduced[1], tl.reduce(x, op="min"))
        tl.store(Reduced[2], tl.reduce(tl.abs(x), op="max"))


@triton.jit(model="weft")
def absolute_maxima(X: tl.View[tl.f32, (M, K)], Y: tl.View[tl.f32, (M,)]):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.blocks(K, extent=32) as block:
            tl.store(Y[rows], tl.reduce(tl.abs(tl.load(X[rows, block])), op="max", axis="K"))


@triton.jit(model="weft")
def scalar_half_widen(X: tl.View[tl.f16, (N,)], Y: tl.View[tl.f32, (N,)]):
    for i in range(N):
        tl.store(Y[i], tl.f32(tl.load(X[i])))


@triton.jit(model="weft", alias_groups={"Y": "output", "Wide": "wide"})
def independent_half_pair(X: tl.View[tl.f16, (N,)], Other: tl.View[tl.f16, (N,)],
                          Y: tl.View[tl.f32, (N, 2)], Wide: tl.View[tl.f64, (N, 2)]):
    with tl.level.blocks(N, extent=1) as point:
        pair = tl.join(tl.load(X[point])[0], tl.load(Other[point])[0])
        narrow, wide = tl.f32(pair), tl.f64(pair)
        tl.store(Y[point][0, 0], narrow[0]); tl.store(Y[point][0, 1], narrow[1])
        tl.store(Wide[point][0, 0], wide[0]); tl.store(Wide[point][0, 1], wide[1])


@triton.jit(model="weft", alias_groups={"Y": "output", "Wide": "wide"})
def snapshot_half_pair(X: tl.View[tl.f16, (N, 2)],
                       Y: tl.View[tl.f32, (N, 2)], Wide: tl.View[tl.f64, (N, 2)]):
    with tl.level.blocks(N, extent=1) as point:
        saved = tl.load(X[point])
        pair = tl.join(saved[0, 0], saved[0, 1])
        narrow, wide = tl.f32(pair), tl.f64(pair)
        tl.store(Y[point][0, 0], narrow[0]); tl.store(Y[point][0, 1], narrow[1])
        tl.store(Wide[point][0, 0], wide[0]); tl.store(Wide[point][0, 1], wide[1])


@triton.jit(model="weft", strided=("X",), alias_groups={"Y": "output", "Wide": "wide"})
def broadcast_half_snapshot(X: tl.View[tl.f16, (N,)], Y: tl.View[tl.f32, (N, C)],
                            Wide: tl.View[tl.f64, (N, C)], C: tl.constexpr = 32):
    columns = tl.arange(0, C, axis="C")
    replacement = tl.f16(tl.arange(0, 32, axis="N") * 0 + 1)
    with tl.level.rows(N, group=32) as rows:
        saved = tl.load(X[rows])
        tl.store(X[rows], replacement)
        tl.store(Y[rows, :], tl.where(columns >= 0, tl.f32(saved), tl.f32(0)))
        tl.store(Wide[rows, :], tl.where(columns >= 0, tl.f64(saved), tl.f64(0)))


@triton.jit(model="weft")
def mixed_integer_mac(A: tl.View[tl.i32, (N,)], B: tl.View[tl.i32, (N,)],
                      Acc: tl.View[tl.i32, (N,)], Y: tl.View[tl.i32, (N,)],
                      N: tl.constexpr, BITS: tl.constexpr, SWAP: tl.constexpr,
                      NIBBLE: tl.constexpr = False):
    a, b, acc = tl.load(A), tl.load(B), tl.load(Acc)
    if BITS == 8:
        signed = tl.i8(a)
        if NIBBLE:
            unsigned = tl.u4(b)
        else:
            unsigned = tl.u8(b)
        if SWAP:
            product = tl.i16(unsigned) * tl.i16(signed)
        else:
            product = tl.i16(signed) * tl.i16(unsigned)
        result = tl.i16(acc) + product
    else:
        signed, unsigned = tl.i16(a), tl.u16(b)
        if SWAP:
            product = tl.i32(unsigned) * tl.i32(signed)
        else:
            product = tl.i32(signed) * tl.i32(unsigned)
        result = acc + product
    tl.store(Y, tl.i32(result))


@triton.jit(model="weft")
def byte_sum_tail(X: tl.View[tl.i8, (N,)], U: tl.View[tl.u8, (N,)], Y: tl.View[tl.i32, (4,)]):
    with tl.level.blocks(N, extent=257) as block:
        x, u = tl.load(X[block]), tl.load(U[block])
        tl.store(Y[0], tl.i32(tl.sum(x, 0, dtype=tl.i16)))
        tl.store(Y[1], tl.reduce(tl.widen(x, tl.i32), op="add"))
        tl.store(Y[2], tl.i32(tl.sum(u, 0, dtype=tl.i16)))
        tl.store(Y[3], tl.reduce(tl.widen(u, tl.i32), op="add"))


@triton.jit(model="weft")
def byte_sum_rows(X: tl.View[tl.i8, (M, N)], U: tl.View[tl.u8, (M, N)],
                  Y: tl.View[tl.i32, (M, 4)], ROWS: tl.constexpr = 4, BLOCK: tl.constexpr = 257):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(N, extent=BLOCK) as block:
            x, u = tl.load(X[rows, block]), tl.load(U[rows, block])
            tl.store(Y[rows, 0], tl.i32(tl.sum(x, 1, dtype=tl.i16)))
            tl.store(Y[rows, 1], tl.sum(tl.i32(x), 1))
            tl.store(Y[rows, 2], tl.i32(tl.sum(u, 1, dtype=tl.i16)))
            tl.store(Y[rows, 3], tl.sum(tl.i32(u), 1))


@triton.jit(model="weft")
def integer_sum_rows(X: tl.View[tl.i64, (M, N)], Y: tl.View[tl.i64, (M,)],
                     BITS: tl.constexpr, UNSIGNED: tl.constexpr):
    with tl.level.rows(M, group=3) as rows:
        with tl.level.blocks(N, extent=257) as block:
            source = tl.load(X[rows, block])
            if BITS == 8:
                values = tl.u8(source) if UNSIGNED else tl.i8(source)
            else:
                values = tl.u64(source) if UNSIGNED else source
            tl.store(Y[rows], tl.i64(tl.reduce(values, op="add", axis="N")))


@triton.jit
def byte_contraction_body(A, B, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            total = tl.state(tl.i32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as block:
                a, b = tl.load(A[rows, block]), tl.load(B[columns, block])
                total += tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], total)


@triton.jit(model="weft")
def byte_contraction_unsigned(A: tl.View[tl.u8, (M, K)], B: tl.View[tl.i8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                              BM: tl.constexpr = 4, BN: tl.constexpr = 4, BK: tl.constexpr = 32):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_contraction_signed(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                            BM: tl.constexpr = 4, BN: tl.constexpr = 4, BK: tl.constexpr = 32):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_contraction_both_unsigned(A: tl.View[tl.u8, (M, K)], B: tl.View[tl.u8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                                   BM: tl.constexpr = 4, BN: tl.constexpr = 4, BK: tl.constexpr = 32):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_contraction_signed_unsigned(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.u8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                                     BM: tl.constexpr = 4, BN: tl.constexpr = 4, BK: tl.constexpr = 32):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def word_byte_contraction(A: tl.View[tl.i16, (M, K)], B: tl.View[tl.i8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def uword_byte_contraction(A: tl.View[tl.u16, (M, K)], B: tl.View[tl.u8, (N, K)], Y: tl.View[tl.i32, (M, N)],
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_word_contraction(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)], Y: tl.View[tl.i32, (M, N)],
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_uword_contraction(A: tl.View[tl.u8, (M, K)], B: tl.View[tl.u16, (N, K)], Y: tl.View[tl.i32, (M, N)],
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit(model="weft")
def byte_scaled_word_contraction(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)],
                                 S: tl.View[tl.i16, (N, K)], Y: tl.View[tl.i32, (M, N)],
                                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                 MUTATE: tl.constexpr = False):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            result = tl.state(tl.i32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=BK) as block:
                scaled = tl.i16(tl.load(B[columns, block]) * tl.load(S[columns, block]))
                if MUTATE:
                    tl.store(B[columns, block], scaled * tl.i16(0))
                result += tl.reduce_dot(tl.load(A[rows, block]), scaled, over="K", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft")
def word_word_contraction(A: tl.View[tl.i16, (M, K)], B: tl.View[tl.i16, (N, K)], Y: tl.View[tl.i32, (M, N)],
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    byte_contraction_body(A, B, Y, M, N, K, BM, BN, BK)


@triton.jit
def partial_axis_helper(X, point, ROWS: tl.constexpr):
    LANES: tl.constexpr = 32 // 4
    partial = tl.state(tl.f32, [ROWS, LANES], init=0)
    value = tl.reshape(tl.load(X), (ROWS, LANES), axes=("M", "LANES"), order=("M", "K"))
    for step in range(2):
        partial += value
    return partial


@triton.jit(model="weft", strided=("A",))
def byte_contraction_snapshot(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i8, (N, K)],
                              Y: tl.View[tl.i32, (M, N)], MUTATE: tl.constexpr,
                              BM: tl.constexpr = 16, BN: tl.constexpr = 16, LOW_MASK: tl.constexpr = 255):
    with tl.level.rows(M, group=BM) as rows:
        with tl.level.cols(N, group=BN) as columns:
            total = tl.state(tl.i32, [BM, BN], init=0)
            with tl.level.blocks(K, extent=32) as block:
                a = tl.load(A[rows, block])
                if MUTATE:
                    tl.store(A[rows, block], a * tl.i8(0))
                a = tl.i8(tl.i16(a) & tl.i16(LOW_MASK))
                b = tl.load(B[columns, block])
                total += tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32)
            tl.store(Y[rows, columns], total)


@triton.jit(model="weft")
def word_snapshot_reuse(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)],
                        Y: tl.View[tl.i32, (M, N)], MUTATE: tl.constexpr,
                        K: tl.constexpr = 128, N: tl.constexpr = 16, BM: tl.constexpr = 8):
    saved = tl.load(B)
    if MUTATE:
        tl.store(B, saved * tl.i16(0))
    with tl.level.rows(M, group=BM) as rows:
        a = tl.load(A[rows])
        result = tl.reduce_dot(a, saved, over="K", acc_dtype=tl.i32)
        tl.store(Y[rows], result)


@triton.jit(model="weft")
def word_snapshot_branches(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)],
                           Y: tl.View[tl.i32, (M, N)], Z: tl.View[tl.i32, (M, N)],
                           enabled: tl.i32, K: tl.constexpr = 128,
                           N: tl.constexpr = 16, BM: tl.constexpr = 8):
    saved = tl.load(B)
    with tl.level.rows(M, group=BM) as rows:
        a = tl.load(A[rows])
        first = tl.reduce_dot(a, saved, over="K", acc_dtype=tl.i32)
        tl.store(Y[rows], first)
        if enabled:
            shifted = a + tl.i8(1)
            second = tl.reduce_dot(shifted, saved, over="K", acc_dtype=tl.i32)
            tl.store(Z[rows], second)


@triton.jit(model="weft")
def word_packing_iterations(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)],
                            Y: tl.View[tl.i32, (M, N)], START: tl.constexpr,
                            STOP: tl.constexpr, STEP: tl.constexpr,
                            M: tl.constexpr = 8, N: tl.constexpr = 16, K: tl.constexpr = 128):
    a, b = tl.load(A), tl.load(B)
    for iteration in range(START, STOP, STEP):
        value = tl.reduce_dot(a, b, over="K", acc_dtype=tl.i32, engine="matrix")
        tl.store(Y, value + tl.i32(iteration))


@triton.jit(model="weft")
def word_snapshot_pressure(A: tl.View[tl.i8, (M, K)], B: tl.View[tl.i16, (N, K)],
                           Y: tl.View[tl.i32, (M, N)], K: tl.constexpr = 128,
                           N: tl.constexpr = 32, BM: tl.constexpr = 8):
    saved = tl.load(B)
    with tl.level.rows(M, group=BM) as rows:
        total = tl.state(tl.i32, [BM, N], init=0)
        a = tl.load(A[rows])
        for step in tl.static_range(9):
            changed = saved + tl.i16(step)
            total += tl.reduce_dot(a, changed, over="K", acc_dtype=tl.i32)
        tl.store(Y[rows], total)


@triton.jit(model="weft")
def storage_lifetime_control(X: tl.View[tl.i32, (N,)], Y: tl.View[tl.i32, (4,)],
                             steps: tl.i32, choose: tl.i32):
    original = tl.load(X)
    saved = original
    step = 0
    while step < steps:
        if (step + choose) % 2:
            selected = saved
        else:
            selected = original
        # Later reads must see the write, while both saved SSA values retain
        # their original bytes even across a branch and a loop back edge.
        tl.store(X, tl.load(X) + tl.i32(3))
        tl.store(Y[step], tl.sum(selected))
        saved = selected + tl.i32(1)
        step += 1


@triton.jit(model="weft")
def bounded_storage_lifetime_control(X: tl.View[tl.i32, (CAP,)], Y: tl.View[tl.i32, (4,)],
                                     steps: tl.i32, choose: tl.i32, CAP: tl.constexpr):
    storage_lifetime_control(X, Y, steps, choose)


@triton.jit(model="weft")
def dense_stage_residency(X: tl.View[tl.i32, (M, N)], Y: tl.View[tl.i32, (M, N)],
                          Z: tl.View[tl.i32, (M, N)], MUTATE: tl.constexpr,
                          BLOCK: tl.constexpr, REDUCE: tl.constexpr = False):
    with tl.level.rows(M, group=2) as rows:
        with tl.level.blocks(N, extent=BLOCK) as block:
            saved = tl.stage(tl.load(X[rows, block]))
            if MUTATE:
                tl.store(X[rows, block], saved * tl.i32(0) + tl.i32(7))
            with tl.level.subtiles(rows, extent=1) as row:
                with tl.level.subtiles(block, extent=8) as small:
                    value = saved[row, small]
                    if REDUCE == 4:
                        delta = tl.sum(tl.reshape(value, (8,)) * tl.i32(2) + tl.i32(1))
                    elif REDUCE == 2:
                        delta = tl.max(value)
                    elif REDUCE == 3:
                        delta = tl.min(value)
                    else:
                        delta = tl.sum(value) if REDUCE else tl.i32(1)
                    tl.store(Y[row, small], value + delta)
            with tl.level.subtiles(rows, extent=1) as row:
                with tl.level.subtiles(block, extent=16) as large:
                    value = saved[row, large]
                    if REDUCE == 4:
                        delta = tl.sum(tl.reshape(value, (16,)) * tl.i32(2) + tl.i32(1))
                    elif REDUCE == 2:
                        delta = tl.max(value)
                    elif REDUCE == 3:
                        delta = tl.min(value)
                    else:
                        delta = tl.sum(value) if REDUCE else tl.i32(1)
                    tl.store(Z[row, large], value - delta)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Z": "output"})
def dense_stage_disjoint(X: tl.View[tl.i32, (M, N)], Y: tl.View[tl.i32, (M, N)],
                         Z: tl.View[tl.i32, (M, N)], MUTATE: tl.constexpr,
                         BLOCK: tl.constexpr, REDUCE: tl.constexpr = False):
    dense_stage_residency(X, Y, Z, MUTATE=MUTATE, BLOCK=BLOCK, REDUCE=REDUCE)


@triton.jit(model="weft")
def partial_axis_identity(X: tl.View[tl.f32, (M, K)], Y: tl.View[tl.f32, (M, K)]):
    ROWS: tl.constexpr = 8
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=8) as block:
            value = partial_axis_helper(X[rows, block], rows, ROWS)
            tl.store(Y[rows, block], tl.reshape(value, (ROWS, 8), axes=("M", "K"), order=("M", "LANES")))


@triton.jit(model="weft", strided=("X",))
def array_scalar_reduce(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (3,)]):
    value = tl.load(X)
    tl.store(Y[0], tl.reduce(value * value, op="add", axis="n"))
    tl.store(Y[1], tl.reduce(value, op="max", axis="n"))
    tl.store(Y[2], tl.reduce(value, op="min", axis="n"))


@triton.jit(model="weft", strided=("X8", "Y8", "X16", "Y16", "Half", "HalfOut", "Wide", "WideOut"))
def array_storage(X8: tl.View[tl.i8, (N,)], Y8: tl.View[tl.i8, (N,)],
                  X16: tl.View[tl.i16, (N,)], Y16: tl.View[tl.i16, (N,)],
                  Half: tl.View[tl.f16, (N,)], HalfOut: tl.View[tl.f16, (N,)],
                  Wide: tl.View[tl.f64, (N,)], WideOut: tl.View[tl.f64, (N,)]):
    tl.store(Y8, tl.load(X8))
    tl.store(Y16, tl.load(X16))
    tl.store(HalfOut, tl.load(Half))
    tl.store(WideOut, tl.load(Wide))


@triton.jit(noinline=True)
def bfloat_snapshot(X: tl.View[tl.bf16, (N,)]):
    return tl.load(X)


@triton.jit(model="weft", strided=("X", "Saved", "Wide"))
def array_bfloat_storage(X: tl.View[tl.bf16, (N,)], Saved: tl.View[tl.bf16, (N,)],
                         Wide: tl.View[tl.f32, (N,)]):
    snapshot = bfloat_snapshot(X)
    tl.store(X, tl.bf16(tl.f32(snapshot) + 1.0))
    tl.store(Saved, snapshot)
    tl.store(Wide, tl.f32(snapshot))


@triton.jit(model="weft")
def array_bfloat_rounding(X: tl.View[tl.f32, (N,)], Nearest: tl.View[tl.bf16, (N,)],
                          Truncated: tl.View[tl.bf16, (N,)]):
    values = tl.load(X)
    tl.store(Nearest, values.to(tl.bfloat16))
    tl.store(Truncated, values.to(tl.bfloat16, fp_downcast_rounding="rtz"))


@triton.jit(model="weft", strided=("X",))
def array_forward_snapshot(X: tl.View[tl.f32, (N,)], Stats: tl.View[tl.f32, (3,)]):
    original = tl.load(X)
    scaled = original * 2.0 + 1.0
    squared = scaled * scaled
    tl.store(X, original - 3.0)
    tl.store(Stats[0], tl.reduce(squared, op="add", axis="n"))
    tl.store(Stats[1], tl.reduce(scaled, op="add", axis="n"))
    tl.store(Stats[2], tl.reduce(original, op="max", axis="n"))


@triton.jit(model="weft")
def array_forward_integer(X: tl.View[tl.i32, (N,)], Out: tl.View[tl.i32, (1,)]):
    narrowed = tl.i16(tl.load(X))
    product = tl.i32(narrowed) * 131071
    tl.store(Out[0], tl.reduce(product, op="add", axis="n"))


@triton.jit(model="weft")
def array_resident_reduce(X: tl.View[tl.f32, (N,)], Keep: tl.View[tl.f32, (RESIDENT,)],
                          Out: tl.View[tl.f32, (RESIDENT,)], Stats: tl.View[tl.f32, (1,)],
                          RESIDENT: tl.constexpr):
    held = tl.load(Keep)
    value = tl.load(X)
    result = tl.reduce(value * value, op="add", axis="n")
    tl.store(Stats[0], result)
    tl.store(Keep, held - 1.0)
    tl.store(Out, held + result)


@triton.jit(model="weft")
def array_forward_stage(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    with tl.level.blocks(N, extent=5008) as block:
        frozen = tl.stage(tl.load(X[block]) * tl.load(X[block]))
        value = (frozen - tl.f32(1.000000238418579)) * tl.f32(1099511627776.0)
        tl.store(Y[block], value)


@triton.jit(model="weft")
def grouped_mac(A: tl.View[tl.i32, (BATCH, M, K)], B: tl.View[tl.i32, (BATCH, H, K)],
                Out, MODE: tl.constexpr):
    width: tl.constexpr = 8 if MODE == 0 else 4
    with tl.level.rows(BATCH, group=2) as batch:
        with tl.level.rows(M, group=3) as rows:
            with tl.level.cols(H, group=2) as heads:
                with tl.level.blocks(K, extent=16) as part:
                    left, right = tl.load(A[batch, rows, part]), tl.load(B[batch, heads, part])
                    if MODE == 0:
                        result = tl.mac_pairs(left.to(tl.int8), right.to(tl.uint8), into=tl.i16)
                    elif MODE == 1:
                        result = tl.mac_groups(left.to(tl.uint8), right.to(tl.uint8), n=4, into=tl.u16)
                    elif MODE == 2:
                        result = tl.mac_groups(left.to(tl.int16), right.to(tl.int16), n=4, into=tl.i32)
                    elif MODE == 3:
                        result = tl.mac_groups(tl.u8(left), tl.i8(right), n=4, into=tl.i32)
                    elif MODE == 4:
                        result = tl.mac_groups(tl.i8(left), tl.i8(right), n=4, into=tl.i32)
                    else:
                        result = tl.mac_groups(tl.u8(left), tl.u8(right), n=4, into=tl.i32)
                    for b in range(2):
                        for m in range(3):
                            for h in range(2):
                                for g in range(width):
                                    tl.store(Out + ((b*3+m)*2+h)*width+g, result[b, m, h, g])


@triton.jit(noinline=True)
def grouped_scalar_handoff(value, alternate, steps):
    carried = alternate
    for iteration in range(steps):
        if iteration + 1 == steps:
            carried = value
    return carried


@triton.jit(model="weft")
def byte_dot_tail(A: tl.View[tl.i8, (N,)], B: tl.View[tl.i8, (N,)], Out, Total,
                  CONTROL: tl.constexpr = False):
    with tl.level.blocks(N, extent=96) as block:
        a, b = tl.load(A[block]), tl.load(B[block])
        result = tl.mac_groups(a, b, n=4, into=tl.i32)
        for i in tl.static_range(24):
            value = result[i]
            if CONTROL:
                value = grouped_scalar_handoff(value, result[(i + 1) % 24], tl.extent(A, 0) % 3 + 1)
            tl.store(Out + i, value)
        tl.store(Total, tl.reduce_dot(a, b, over="N", acc_dtype=tl.i32))


@triton.jit(model="weft")
def aligned_integer_pairs(A: tl.View[tl.i32, (N,)], B: tl.View[tl.i32, (N,)], Out, MODE: tl.constexpr):
    with tl.level.blocks(N, extent=32) as block:
        left, right = tl.load(A[block]), tl.load(B[block])
        if MODE == 0:
            result = tl.mac_pairs(tl.u4(left), tl.i8(right), into=tl.i16)
        elif MODE == 1:
            result = tl.mac_pairs(tl.u8(left), tl.i8(right), into=tl.i16)
        elif MODE == 2:
            result = tl.mac_pairs(tl.i16(left), tl.i16(right), into=tl.i32)
        elif MODE == 3:
            result = tl.mac_pairs(left > tl.i32(0), tl.i16(right), into=tl.i32)
        elif MODE == 7:
            lane = tl.arange(0, 16, dtype=tl.u8, axis="table")
            table = (lane * tl.u8(10)) & tl.u8(127)
            selected = tl.lookup(table, tl.u8(left) & tl.u8(15), bounds="in_bounds")
            result = tl.mac_pairs(selected, tl.i8(right), into=tl.i16)
        else:
            MASK: tl.constexpr = 127 if MODE == 4 else 128 if MODE == 5 else 129
            result = tl.mac_pairs(tl.u8(left) & tl.u8(MASK), tl.i8(right), into=tl.i16)
        for index in tl.static_range(16):
            tl.store(Out + index, tl.i32(result[index]))


@triton.jit
def equal_extent_helper(X: tl.View[tl.i32, (ROWS, PARTIALS)]):
    result = tl.state(tl.i32, [ROWS, PARTIALS], init=0)
    for repeat in range(2):
        result += tl.load(X)
    return tl.sum(result, 1)


@triton.jit(model="weft")
def byte_dot_narrow(A: tl.View[tl.i32, (N,)], B: tl.View[tl.i32, (N,)],
                    Out: tl.View[tl.i32, (GROUPS,)], MODE: tl.constexpr):
    with tl.level.blocks(N, extent=32) as block:
        a, b = tl.load(A[block]), tl.load(B[block])
        if MODE == 0:
            result = tl.mac_groups(tl.u8(a), tl.i8(b), n=4, into=tl.i16)
        elif MODE == 1:
            result = tl.mac_groups(tl.u4(a), tl.i8(b), n=4, into=tl.i16)
        elif MODE == 2:
            result = tl.mac_groups(tl.i8(a), tl.i8(b), n=4, into=tl.i16)
        elif MODE == 3:
            result = tl.mac_groups(tl.u8(a), tl.i8(b), n=4, into=tl.u16)
        elif MODE == 5:
            result = tl.mac_groups(tl.i16(tl.u4(a)), tl.i16(tl.i8(b)), n=4, into=tl.i16)
        elif MODE == 6:
            result = tl.mac_groups(tl.u8(a), tl.u8(b) & tl.u8(127), n=4, into=tl.i16)
        else:
            result = tl.mac_groups(tl.u4(a), tl.u4(b), n=4, into=tl.u16)
        wide = tl.i32(result)
        for index in tl.static_range(8):
            tl.store(Out[index], wide[index])


@triton.jit(noinline=True)
def grouped_while_handoff(value, alternate, steps):
    carried = alternate
    iteration = 0
    while iteration < steps:
        if iteration + 1 == steps:
            carried = value
        iteration += 1
    return carried


@triton.jit(noinline=True)
def grouped_level_handoff(value, alternate, steps):
    carried = alternate
    with tl.level.blocks(steps, extent=1) as iteration:
        carried = value
    return carried


@triton.jit(model="weft")
def grouped_consumers(A: tl.View[tl.i32, (M, N)], B: tl.View[tl.i32, (M, N)],
                      Out: tl.View[tl.i32, (ROWS, GROUPS)], Totals: tl.View[tl.i32, (ROWS,)],
                      CONTROL: tl.constexpr, REDUCTION: tl.constexpr):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.blocks(N, extent=32) as block:
            values = tl.mac_groups(tl.i8(tl.load(A[rows, block])), tl.i8(tl.load(B[rows, block])), n=4, into=tl.i16)
            if CONTROL == 1:
                values = grouped_while_handoff(values, values + tl.i16(1), tl.extent(A, 1) % 3 + 1)
            elif CONTROL == 2:
                values = grouped_level_handoff(values, values + tl.i16(1), tl.extent(A, 1) % 3 + 1)
            # Renamed output axes expose complete storage capacity, including an
            # inactive fourth row and partial groups in the last active row.
            stored = tl.reshape(tl.i32(values), shape=(4, 8), axes=("ROWS", "GROUPS"), order=("M", "N"))
            tl.store(tl.subview(Out, offsets=(0, 0), extents=(4, 8)), stored)
            reduced = tl.reduce(values, op=REDUCTION, axis="N")
            renamed = tl.reshape(tl.i32(reduced), shape=(4,), axes=("ROWS",), order=("M",))
            tl.store(tl.subview(Totals, offsets=(0,), extents=(4,)), renamed)


@triton.jit(model="weft")
def equal_extent_axes(X: tl.View[tl.i32, (M, K)], Y: tl.View[tl.i32, (M,)]):
    with tl.level.rows(M, group=4) as rows:
        tl.store(Y[rows], equal_extent_helper(X[rows]))


@triton.jit(model="weft")
def extrema_identity(X: tl.View[tl.f32, (N,)], Out, KIND: tl.constexpr, BLOCK: tl.constexpr):
    if BLOCK == 0:
        tl.store(Out, tl.reduce(tl.load(X), op=KIND, axis="N"))
    else:
        with tl.level.blocks(N, extent=BLOCK) as block:
            tl.store(Out, tl.reduce(tl.load(X[block]), op=KIND, axis="N"))


@triton.jit(model="weft")
def grouped_mac_float(A: tl.View[tl.f32, (BATCH, M, K)], B: tl.View[tl.f32, (BATCH, H, K)], Out):
    with tl.level.rows(BATCH, group=2) as batch:
        with tl.level.rows(M, group=3) as rows:
            with tl.level.cols(H, group=2) as heads:
                with tl.level.blocks(K, extent=12) as part:
                    result = tl.mac_groups(tl.load(A[batch, rows, part]), tl.load(B[batch, heads, part]),
                                           n=3, into=tl.f32)
                    for b in range(2):
                        for m in range(3):
                            for h in range(2):
                                for g in range(4):
                                    tl.store(Out + ((b*3+m)*2+h)*4+g, result[b, m, h, g])


@triton.jit(model="weft")
def fold_pairs(X: tl.View[tl.i32, (BATCH, M, K)], Out, UNSIGNED: tl.constexpr):
    with tl.level.rows(BATCH, group=2) as batch:
        with tl.level.rows(M, group=3) as rows:
            with tl.level.blocks(K, extent=16) as part:
                values = tl.load(X[batch, rows, part])
                if UNSIGNED:
                    result = tl.sum_pairs(values.to(tl.uint16))
                else:
                    result = tl.sum_pairs(values.to(tl.int16))
                for b in range(2):
                    for m in range(3):
                        for g in range(8):
                            tl.store(Out + (b*3+m)*8+g, result[b, m, g])


@triton.jit(model="weft")
def grouped_tail(X: tl.View[tl.i8, (N,)], Weight: tl.View[tl.u8, (N,)], Groups, Pairs, Counts):
    cursor = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=32) as block:
        values, weights = tl.load(X[block]), tl.load(Weight[block])
        groups = tl.mac_groups(values, weights, n=4, into=tl.i16)
        pairs = tl.sum_pairs(values)
        for g in range(8):
            value = groups[g]
            tl.store(Groups + cursor//4+g, value)
            tl.atomic_add(Counts + (value.to(tl.int32) & 1), 1)
        for g in range(16):
            value = pairs[g]
            tl.store(Pairs + cursor//2+g, value)
            tl.atomic_add(Counts + 2+(value & 1), 1)
        cursor += 32


@triton.jit(model="weft", alias_groups={"Table": "table", "Indices": "indices",
            "Old": "old", "New": "new", "Scalar": "scalar"})
def lookup_snapshot(Table: tl.View[tl.i32, (TABLE_SIZE,)], Indices: tl.View[tl.u32, (M, N)],
                    Old, New, Scalar, *, TABLE_SIZE: tl.constexpr = 16, SCALAR_INDEX: tl.constexpr = 3):
    saved = tl.load(Table)
    tl.store(Table, saved + 1000)
    with tl.level.rows(M, group=2) as rows:
        with tl.level.blocks(N, extent=8) as columns:
            indices = tl.load(Indices[rows, columns])
            old = tl.lookup(saved, indices, bounds="in_bounds")
            new = tl.lookup(Table, indices, bounds="in_bounds")
            for row in range(2):
                for column in range(8):
                    tl.store(Old + row*8+column, old[row, column])
                    tl.store(New + row*8+column, new[row, column])
    tl.store(Scalar, tl.lookup(saved, tl.u32(SCALAR_INDEX), bounds="in_bounds"))


@triton.jit(model="weft")
def lookup_tail(Table: tl.View[tl.i32, (TABLE_SIZE,)], Indices: tl.View[tl.u32, (N,)], Out, Counts,
                *, TABLE_SIZE: tl.constexpr = 16):
    table = tl.load(Table)
    cursor = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=32) as block:
        values = tl.lookup(table, tl.load(Indices[block]), bounds="in_bounds")
        for lane in range(32):
            value = values[lane]
            tl.store(Out + cursor+lane, value)
            tl.atomic_add(Counts + (value & 3), 1)
        cursor += 32


@triton.jit(model="weft", strided=("Table",))
def lookup_strided(Table: tl.View[tl.i32, (R, C)], Indices: tl.View[tl.u32, (N,)],
                   Y: tl.View[tl.i32, (N,)]):
    with tl.level.blocks(N, extent=17) as block:
        tl.store(Y[block], tl.lookup(Table, tl.load(Indices[block]), bounds="in_bounds"))


@triton.jit(model="weft")
def lookup_effect_barrier(Table: tl.View[tl.i32, (N,)], Y: tl.View[tl.i32, (2,)]):
    saved = tl.load(Table)
    tl.store(Table[0], tl.i32(-77))
    tl.store(Y[0], tl.lookup(saved, tl.u32(0), bounds="in_bounds"))
    tl.store(Y[1], tl.lookup(Table, tl.u32(0), bounds="in_bounds"))


@triton.jit(model="weft", strided=("Table",))
def lookup_repeated(Table: tl.View[tl.i32, (N,)], Indices: tl.View[tl.u32, (2,)],
                    Y: tl.View[tl.i32, (M,)]):
    lane = tl.arange(0, 16, dtype=tl.u32, axis="lane")
    indices = tl.load(Indices)
    with tl.level.blocks(M, extent=16) as block:
        offsets = tl.lookup(indices, lane // tl.u32(8), bounds="in_bounds")
        values = tl.lookup(Table, offsets, bounds="in_bounds")
        tl.store(Y[block], tl.reshape(values, (16,), axes=("M",), order=("lane",)))


@triton.jit(model="weft", strided=("Table",))
def lookup_runs(Table: tl.View[tl.i8, (256,)], Base: tl.View[tl.u8, (M,)],
                Y: tl.View[tl.i8, (M, N)]):
    lane = tl.arange(0, 16, dtype=tl.u8, axis="n")
    with tl.level.rows(M, group=2) as row:
        with tl.level.blocks(N, extent=16) as column:
            indices = tl.load(Base[row]) + lane
            tl.store(Y[row, column], tl.lookup(Table, indices, bounds="in_bounds"))


@triton.jit
def lookup_record_body(Table, Base, Y, M, N, WRAP: tl.constexpr):
    lane = tl.arange(0, 8, dtype=tl.u32, axis="n")
    with tl.level.rows(M, group=4) as row:
        with tl.level.blocks(N, extent=8) as column:
            base = tl.u32(tl.load(Base[row])) & tl.u32(1023)
            index = base * tl.u32(8) + lane
            if WRAP:
                index = tl.u32(tl.u8(index + tl.u32(253)))
            tl.store(Y[row, column], tl.lookup(Table, index, bounds="in_bounds"))


@triton.jit(model="weft")
def lookup_record_rows(Table: tl.View[tl.i8, (1024, 8)],
                       Base: tl.View[tl.u16, (M,)], Y: tl.View[tl.i8, (M, N)],
                       WRAP: tl.constexpr):
    lookup_record_body(Table, Base, Y, M, N, WRAP)


@triton.jit(model="weft", strided=("Table",))
def lookup_record_strided(Table: tl.View[tl.i8, (1024, 8)],
                          Base: tl.View[tl.u16, (M,)], Y: tl.View[tl.i8, (M, N)],
                          WRAP: tl.constexpr):
    lookup_record_body(Table, Base, Y, M, N, WRAP)


@triton.jit(model="weft")
def indexed_row_packets(X: tl.View[tl.u8, (M, K)], Y: tl.View[tl.i32, (M, N)],
                        Start: tl.u8, K: tl.constexpr = 256, N: tl.constexpr = 32):
    with tl.level.rows(M, group=4) as rows:
        saved = tl.load(X[rows])
        tl.store(X[rows], saved ^ tl.u8(17))
        positions = Start + tl.arange(0, N, dtype=tl.u8, axis="K")
        selected = tl.i32(saved[:, positions])
        tl.store(Y[rows], tl.reshape(selected, (4, N), axes=("M", "N"), order=("M", "K")))


@triton.jit(model="weft")
def indexed_loop_packets(X: tl.View[tl.u8, (M, K)], Y: tl.View[tl.i32, (M, N)], K: tl.constexpr = 256, N: tl.constexpr = 32):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.cols(N, group=N) as columns:
            saved = tl.load(X[rows])
            tl.store(X[rows], saved ^ tl.u8(17))
            result = tl.state(tl.i32, [4, N], init=0)
            for group in range(8):
                positions = tl.u32(group * 32) + tl.arange(0, N, dtype=tl.u32, axis="K")
                selected = tl.i32(saved[:, positions])
                result += tl.reshape(selected, (4, N), axes=("M", "N"), order=("M", "K"))
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft")
def indexed_wide(X: tl.View[tl.f32, (257,)], Bytes: tl.View[tl.i8, (257,)],
                 Indices: tl.View[tl.u32, (N,)], Y: tl.View[tl.f32, (N,)], Yi: tl.View[tl.i32, (N,)]):
    saved = tl.load(X)
    saved_bytes = tl.load(Bytes)
    tl.store(X, saved + 1)
    tl.store(Bytes, saved_bytes + tl.i8(1))
    for row in range(N):
        index = tl.load(Indices[row])
        tl.store(Y[row], saved[index])
        tl.store(Yi[row], tl.i32(saved_bytes[index]))


@triton.jit
def project_helper(X: tl.View[tl.f32, (Rows, Cols)], Y: tl.View[tl.f32, (Rows, Cols)]):
    saved = tl.load(X)
    tl.store(X, saved + 1.0)
    with tl.level.rows(Rows, group=tl.auto(3)) as rows:
        with tl.level.blocks(Cols, extent=tl.auto(17)) as columns:
            tl.store(Y[rows, columns], saved[rows, columns])


@triton.jit(model="weft", strided=("X", "Y"))
def array_project_snapshot(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)]):
    project_helper(X, Y)


@triton.jit(model="weft", strided=("X", "Y"))
def specialized_project_snapshot(X: tl.View[tl.f32, (M, N)], Y: tl.View[tl.f32, (M, N)],
                                 *, N: tl.constexpr):
    project_helper(X, Y)


@triton.jit(model="weft")
def array_project_bits(X: tl.View[tl.i8, (N,)], Y: tl.View[tl.i8, (N,)], KIND: tl.constexpr):
    source = tl.load(X)
    if KIND == 0:
        saved = tl.u4(source)
    elif KIND == 1:
        saved = tl.i4(source)
    else:
        saved = source < 0
    tl.store(X, source + tl.i8(1))
    with tl.level.blocks(N, extent=33) as block:
        tl.store(Y[block], tl.i8(saved[block]))


@triton.jit(model="weft", strided=("X", "Y"))
def array_project_bfloat(X: tl.View[tl.bf16, (M, N)], Y: tl.View[tl.bf16, (M, N)]):
    saved = tl.load(X)
    tl.store(X, tl.bf16(tl.f32(saved) + 1.0))
    with tl.level.rows(M, group=3) as rows:
        with tl.level.blocks(N, extent=17) as columns:
            tl.store(Y[rows, columns], saved[rows, columns])


@triton.jit
def lookup_bits_body(Table, Indices, Old, New, N, TABLE_SIZE: tl.constexpr,
                     BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    table = tl.load(Table)
    tl.store(Table, tl.where(table == table, 0, 0).to(DTYPE))
    cursor = tl.index(0)
    with tl.level.blocks(N, extent=BLOCK) as block:
        indices = tl.load(Indices[block])
        before = tl.lookup(table, indices, bounds="in_bounds")
        after = tl.lookup(Table, indices, bounds="in_bounds")
        for lane in range(BLOCK):
            tl.store(Old + cursor + lane, before[lane])
            tl.store(New + cursor + lane, after[lane])
        cursor += BLOCK


@triton.jit(model="weft")
def lookup_byte_bits(Table: tl.View[tl.i8, (TABLE_SIZE,)], Indices: tl.View[tl.u32, (N,)],
                     Old, New, *, TABLE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    lookup_bits_body(Table, Indices, Old, New, N, TABLE_SIZE, BLOCK, tl.int8)


@triton.jit(model="weft")
def lookup_float_bits(Table: tl.View[tl.f32, (TABLE_SIZE,)], Indices: tl.View[tl.u32, (N,)],
                      Old, New, *, TABLE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    lookup_bits_body(Table, Indices, Old, New, N, TABLE_SIZE, BLOCK, tl.float32)


@triton.jit(model="weft")
def named_selection(X: tl.View[tl.f32, (M, N)], Rows: tl.View[tl.i32, (M,)],
                    Cols: tl.View[tl.f32, (N,)], Out: tl.View[tl.f32, (M, N)],
                    Transposed: tl.View[tl.f32, (N, M)]):
    predicate = tl.load(X) >= 0
    rows, columns = tl.load(Rows), tl.load(Cols)
    tl.store(Out, tl.where(predicate, rows, columns))
    tl.store(Transposed, tl.where(predicate, columns, rows))


@triton.jit(noinline=True)
def scalar_tail_helper(value, mode):
    result = value.to(tl.int16)
    if mode == 0:
        return (result + 1).to(tl.int16)
    if mode == 1:
        for step in range(3):
            result = (result + 2).to(tl.int16)
            if step == 1:
                return result
        return result
    step = tl.full((), 0, tl.int32)
    while step < 3:
        result = (result + 3).to(tl.int16)
        if step == 1:
            return result
        step += 1
    return result


@triton.jit(model="weft")
def scalar_validity(X: tl.View[tl.i8, (N,)], Out, Counts):
    cursor = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=32) as block:
        values = tl.load(X[block])
        for lane in range(32):
            value = scalar_tail_helper(values[lane], (cursor+lane) % 3)
            carried = value.to(tl.int32)
            for iteration in range(2):
                carried = carried + 1
            iteration = tl.full((), 0, tl.int32)
            while iteration < 2:
                carried = carried + 2
                iteration += 1
            if (cursor+lane) % 2 == 0:
                carried = carried + 3
            else:
                carried = carried - 2
            slot = (carried + 5) & 3
            tl.store(Out + cursor+lane, slot)
            tl.atomic_add(Counts + slot, 1)
        cursor += 32

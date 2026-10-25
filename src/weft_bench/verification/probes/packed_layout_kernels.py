"""Public Encoding declarations; shared field operations, without target branches."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.encoding
class BitSigns:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 64
    bit: tl.u1[64] @ tl.grouped(elements=8) @ tl.bit_layers(elements=1, order=tl.lo_first)


@triton.encoding
class UnalignedWords:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 8
    tag: tl.u16
    word: tl.u32[8]


@triton.jit(model="weft", alias_groups={"Y": "output"})
def unaligned_word_windows(X: tl.View[UnalignedWords, (K,)], Y: tl.View[tl.u32, (R,)],
                           UPDATE: tl.constexpr, WHOLE: tl.constexpr):
    record = tl.index(0)
    with tl.level.blocks(K, extent=8) as block:
        saved = tl.load(X[block])
        if UPDATE:
            tl.store(X[block].word, saved.word ^ tl.u32(0x13579bdf))
        if WHOLE:
            total = tl.sum(saved.word)
        else:
            total = tl.u32(0)
            lane = tl.arange(0, 2, dtype=tl.u32, axis="lane")
            for group in range(4):
                total += tl.sum(saved.word[tl.u32(group) * tl.u32(2) + lane])
        tl.store(Y[record], total)
        record += tl.index(1)


@triton.jit(model="weft")
def encoded_sign_product(Signs: tl.View[BitSigns, (N,)], X: tl.View[tl.i8, (N,)],
                         Y: tl.View[tl.i8, (N,)], Bits: tl.View[tl.i8, (N,)],
                         Wide: tl.View[tl.u32, (N,)],
                         SELECT: tl.constexpr, KEEP: tl.constexpr):
    with tl.level.blocks(N, extent=64) as block:
        bit = tl.load(Signs[block]).bit
        numeric = tl.where(bit != tl.u1(0), tl.i8(1), tl.i8(0)) if SELECT else tl.i8(bit)
        tl.store(Y[block], tl.load(X[block]) * (tl.i8(1) - tl.i8(2) * numeric))
        if KEEP:
            tl.store(Bits[block], numeric)
            tl.store(Wide[block], tl.u32(bit))


@triton.jit(model="weft")
def encoded_sign_reduction(Signs: tl.View[BitSigns, (N,)], X: tl.View[tl.i8, (N,)],
                           Y: tl.View[tl.i32, (1,)], BASE: tl.constexpr):
    total = tl.i32(0)
    with tl.level.blocks(N, extent=64) as block:
        bit = tl.i8(tl.load(Signs[block]).bit)
        data = tl.i8(BASE) + bit * tl.i8(0)
        signed = data * (tl.i8(1) - tl.i8(2) * bit)
        total += tl.reduce_dot(signed, tl.load(X[block]), over="n", acc_dtype=tl.i32)
    tl.store(Y[0], total)


@triton.encoding
class NibblePair:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    s: tl.i4[32]
    u: tl.u4[32]
    bias: tl.i32


@triton.jit(model="weft", alias_groups={"Values": "values"})
def encoded_snapshot_rows(X: tl.View[NibblePair, (M, K)],
                          Y: tl.View[NibblePair, (M, K)],
                          Values: tl.View[tl.i32, (M, K)]):
    with tl.level.rows(M, group=4) as rows:
        saved = tl.stage(tl.load(X[rows]))
        for step in range(2):
            with tl.level.rows(rows, group=1) as row:
                with tl.level.blocks(K, extent=32) as block:
                    value = saved[row, block]
                    tl.store(Y[row, block].s, value.s ^ tl.i4(1))
                    tl.store(Values[row, block], tl.i32(value.s) + step * 1000)


@triton.encoding
class WindowFields:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 32
    low: tl.u4[32] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.lo_first)
    high: tl.u2[32] @ tl.grouped(elements=16) @ tl.bit_layers(elements=4, order=tl.lo_first)
    scales: tl.i8[8]


@triton.derive
def WindowRows(X: tl.View[WindowFields, (M, K)], *, rows: tl.static[int]) -> tl.View[WindowRows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"low": 4, "high": 4, "scales": 1})


@triton.jit
def translated_windows(X, Y, M, K, ROWS: tl.constexpr, WIDTH: tl.constexpr, UPDATE: tl.constexpr):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=32) as block:
            source, destination = X[rows, block], Y[rows, block]
            for part in range(32 // WIDTH):
                offset = part * WIDTH
                low = tl.load(tl.subview(source.low, offsets=(0, offset), extents=(ROWS, WIDTH)))
                high = tl.load(tl.subview(source.high, offsets=(0, offset), extents=(ROWS, WIDTH)))
                scale = tl.load(source.scales[:, offset // 4])
                value = tl.i32(low) + tl.i32(high) * 16 + tl.i32(scale) * 64
                tl.store(tl.subview(destination, offsets=(0, offset), extents=(ROWS, WIDTH)), value)
                if UPDATE:
                    tl.store(source.scales[:, offset // 4], scale ^ tl.i8(1))


@triton.jit(model="weft", alias_groups={"Y": "output"})
def translated_field_windows(X: tl.View[WindowRows[4], (M, K)],
                             Y: tl.View[tl.i32, (M, K)], ROWS: tl.constexpr,
                             WIDTH: tl.constexpr, UPDATE: tl.constexpr = False):
    translated_windows(X, Y, M, K, ROWS, WIDTH, UPDATE)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def translated_plain_windows(X: tl.View[WindowFields, (M, K)],
                             Y: tl.View[tl.i32, (M, K)], ROWS: tl.constexpr,
                             WIDTH: tl.constexpr, UPDATE: tl.constexpr = False):
    translated_windows(X, Y, M, K, ROWS, WIDTH, UPDATE)


@triton.derive
def NibbleRows(X: tl.View[NibblePair, (M, K)], *, rows: tl.static[int]) -> tl.View[NibbleRows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"s": 16, "u": 16, "bias": 4})


@triton.jit(model="weft", alias_groups={"Y": "output"})
def nibble_contraction(X: tl.View[NibbleRows[4], (M, K)],
                       W: tl.View[NibbleRows[32], (N, K)],
                       Y: tl.View[tl.i32, (M, N)], MR: tl.constexpr,
                       UNSIGNED: tl.constexpr, BIAS: tl.constexpr = False,
                       NATIVE_I4: tl.constexpr = False, MIXED: tl.constexpr = False,
                       NR: tl.constexpr = 32):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                x = tl.load(X[rows, block])
                w = tl.load(W[columns, block])
                left = x.u if UNSIGNED else x.s
                right = w.s if UNSIGNED and MIXED else w.u if UNSIGNED or MIXED else w.s
                if NATIVE_I4:
                    product = tl.dot(left, right, over="k", acc_dtype=tl.i32)
                else:
                    product = tl.dot(tl.i8(left), tl.i8(right), over="k", acc_dtype=tl.i32)
                if BIAS:
                    product += x.bias * w.bias
                result += product
            tl.store(Y[rows, columns], result)


@triton.encoding
class FloatScale:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 32
    d: tl.f32
    q: tl.i8[32]


@triton.derive
def FloatScaleRows(X: tl.View[FloatScale, (M, K)], *, rows: tl.static[int]) -> tl.View[FloatScaleRows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"d": 4, "q": 32})


@triton.jit(model="weft", alias_groups={"Y": "output"})
def byte_nibble_contraction(X: tl.View[FloatScaleRows[4], (M, K)],
                            W: tl.View[NibbleRows[32], (N, K)],
                            Y: tl.View[tl.i32, (M, N)], MR: tl.constexpr,
                            SHIFT: tl.constexpr, UNSIGNED: tl.constexpr,
                            COEFFICIENT: tl.constexpr = 0, NR: tl.constexpr = 32):
    with tl.level.rows(M, group=MR) as rows:
        with tl.level.cols(N, group=NR) as columns:
            result = tl.state(tl.i32, [MR, NR], init=0)
            with tl.level.blocks(K, extent=32) as block:
                x = tl.load(X[rows, block])
                w = tl.load(W[columns, block])
                values = x.q >> 4 if SHIFT else x.q
                left = tl.u4(values) if UNSIGNED else tl.i4(values)
                product = tl.dot(left, w.u, over="k", acc_dtype=tl.i32)
                if COEFFICIENT:
                    low = tl.dot(tl.u4(x.q), w.u, over="k", acc_dtype=tl.i32)
                    product = product * COEFFICIENT + low
                result += product
            tl.store(Y[rows, columns], result)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def cohort_float_scale(X: tl.View[FloatScaleRows[4], (M, K)],
                       Z: tl.View[tl.f32, (N, K)], Y: tl.View[tl.f32, (N, K)],
                       First: tl.constexpr, ROWS: tl.constexpr = 4,
                       RECORDS: tl.constexpr = 1):
    for record in tl.static_range(RECORDS):
        part = tl.subview(X, offsets=(First, record * 32), extents=(ROWS, 32))
        scale = tl.reshape(tl.load(part.d), (ROWS,), axes=("N",), order=("M",))
        source = tl.subview(Z, offsets=(0, record * 32), extents=(ROWS, 32))
        destination = tl.subview(Y, offsets=(0, record * 32), extents=(ROWS, 32))
        tl.store(destination, tl.load(source) * scale)


@triton.encoding
class ByteFields:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 1
    elements = 8
    tag: tl.u16
    a: tl.u8[8]
    b: tl.u8[8]


@triton.derive
def ByteRows(X: tl.View[ByteFields, (M, K)], *, rows: tl.static[int]) -> tl.View[ByteRows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"tag": 2, "a": 4, "b": 2})


@triton.jit(model="weft", alias_groups={"Y": "output"})
def cohort_byte_write(X: tl.View[tl.u8, (M, K)], Y: tl.View[ByteRows[4], (M, K)], ROWS: tl.constexpr = 8):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=8) as block:
            value = tl.load(tl.subview(X[rows, block], offsets=(0, 2), extents=(ROWS, 5)))
            tl.store(tl.subview(Y[rows, block].a, offsets=(0, 2), extents=(ROWS, 5)), value)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def cohort_byte_index_write(X: tl.View[tl.u8, (M, K)], Y: tl.View[ByteRows[4], (M, K)], ROWS: tl.constexpr = 8):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=8) as block:
            for index in tl.static_range(2, 7):
                value = tl.load(X[rows, block][:, index])
                tl.store(Y[rows, block].a[:, index], value)


@triton.jit(model="weft", alias_groups={"Y": "output"})
def cohort_byte_slice(X: tl.View[tl.u8, (M, K)], Y: tl.View[ByteRows[4], (M, K)], First: tl.i32, ROWS: tl.constexpr = 4):
    value = tl.load(tl.subview(X, offsets=(First, 2), extents=(ROWS, 5)))
    destination = tl.subview(Y, offsets=(First, 0), extents=(ROWS, 8))
    tl.store(tl.subview(destination.a, offsets=(0, 2), extents=(ROWS, 5)), value)


@triton.encoding
class NaturalLE:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 8
    tag: tl.u16
    q: tl.i4[8]
    s: tl.u6[8]


@triton.encoding
class LayeredJoined:
    layout = (tl.bitorder.lsb_first, tl.byteorder.little)
    alignment = 2
    elements = 8
    tag: tl.u16
    q: tl.i4[8] @ tl.grouped(elements=8) @ tl.bit_layers(elements=4, order=tl.hi_first)
    s: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)
    m: tl.u6[8] @ tl.pack_fields(group=4, fields=2, low_bits=4, order=tl.lo_first)


@triton.encoding
class BigOrder:
    layout = (tl.bitorder.msb_first, tl.byteorder.big)
    alignment = 1
    elements = 8
    tag: tl.u16
    flags: tl.u1[8]


@triton.jit
def reshape_records(X, Y, Observed, ORDERED: tl.constexpr, JOINED: tl.constexpr, SWAP: tl.constexpr):
    saved = tl.reshape(tl.load(tl.subview(X, offsets=(0,), extents=(32,))),
                       (2, 2, 8), axes=("u", "v", "K"), order=("L",))
    if SWAP:
        saved = tl.reshape(saved, (4, 8), axes=("R", "K"), order=("v", "u", "K"))
    else:
        saved = tl.reshape(saved, (4, 8), axes=("R", "K"), order=("u", "v", "K"))
    tl.store(tl.subview(Y, offsets=(0, 0), extents=(4, 8)), saved)
    if ORDERED:
        observed = tl.i32(saved.tag) + tl.i32(saved.flags)
    else:
        observed = tl.i32(saved.tag) + tl.i32(saved.q) + tl.i32(saved.s) * 16
        if JOINED:
            observed += tl.i32(saved.m) * 1024
    tl.store(tl.subview(Observed, offsets=(0, 0), extents=(4, 8)), observed)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def natural_reshape(X: tl.View[NaturalLE, (L,)], Y: tl.View[NaturalLE, (R, K)],
                    Observed: tl.View[tl.i32, (R, K)], SWAP: tl.constexpr = False):
    reshape_records(X, Y, Observed, False, False, SWAP)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def layered_reshape(X: tl.View[LayeredJoined, (L,)], Y: tl.View[LayeredJoined, (R, K)],
                    Observed: tl.View[tl.i32, (R, K)], SWAP: tl.constexpr = False):
    reshape_records(X, Y, Observed, False, True, SWAP)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def ordered_reshape(X: tl.View[BigOrder, (L,)], Y: tl.View[BigOrder, (R, K)],
                    Observed: tl.View[tl.i32, (R, K)], SWAP: tl.constexpr = False):
    reshape_records(X, Y, Observed, True, False, SWAP)


@triton.jit(model="weft", alias_groups={"Observed": "observed"})
def natural_alias_reshape(X: tl.View[NaturalLE, (L,)], Y: tl.View[NaturalLE, (R, K)],
                          Observed: tl.View[tl.i32, (R, K)]):
    reshape_records(X, Y, Observed, False, False, True)


@triton.jit(model="weft", alias_groups={"Observed": "observed"})
def layered_alias_reshape(X: tl.View[LayeredJoined, (L,)], Y: tl.View[LayeredJoined, (R, K)],
                          Observed: tl.View[tl.i32, (R, K)]):
    reshape_records(X, Y, Observed, False, True, True)


@triton.jit(model="weft", alias_groups={"Observed": "observed"})
def ordered_alias_reshape(X: tl.View[BigOrder, (L,)], Y: tl.View[BigOrder, (R, K)],
                          Observed: tl.View[tl.i32, (R, K)]):
    reshape_records(X, Y, Observed, True, False, True)


@triton.jit
def nested_reshape_records(X, Y, Observed, M, K, ORDERED: tl.constexpr, JOINED: tl.constexpr):
    with tl.level.rows(M, group=8) as outer:
        with tl.level.rows(outer, group=4) as rows:
            with tl.level.blocks(K, extent=8) as block:
                saved = tl.reshape(tl.load(X[outer, block][rows, :]), (2, 2, 8),
                                   axes=("u", "v", "K"), order=("M", "K"))
                saved = tl.reshape(saved, (4, 8), axes=("M", "K"), order=("v", "u", "K"))
                tl.store(Y[outer, block][rows, :], saved)
                if ORDERED:
                    observed = tl.i32(saved.tag) + tl.i32(saved.flags)
                else:
                    observed = tl.i32(saved.tag) + tl.i32(saved.q) + tl.i32(saved.s) * 16
                    if JOINED:
                        observed += tl.i32(saved.m) * 1024
                tl.store(Observed[outer, block][rows, :], observed)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def natural_nested_reshape(X: tl.View[NaturalLE, (M, K)], Y: tl.View[NaturalLE, (M, K)],
                           Observed: tl.View[tl.i32, (M, K)]):
    nested_reshape_records(X, Y, Observed, M, K, False, False)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def layered_nested_reshape(X: tl.View[LayeredJoined, (M, K)], Y: tl.View[LayeredJoined, (M, K)],
                           Observed: tl.View[tl.i32, (M, K)]):
    nested_reshape_records(X, Y, Observed, M, K, False, True)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def ordered_nested_reshape(X: tl.View[BigOrder, (M, K)], Y: tl.View[BigOrder, (M, K)],
                           Observed: tl.View[tl.i32, (M, K)]):
    nested_reshape_records(X, Y, Observed, M, K, True, False)


@triton.derive
def NaturalRows(X: tl.View[NaturalLE, (M, K)], *, rows: tl.static[int]) -> tl.View[NaturalRows[rows], (M, K)]:
    return tl.interleave(X, rows=rows, field_chunks={"tag": 2, "q": 1, "s": 2})


@triton.jit(model="weft")
def cohort_fields(X: tl.View[NaturalRows[4], (M, K)], Y: tl.View[tl.i32, (M, K)], ROWS: tl.constexpr = 4):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=8) as block:
            tile = X[rows, block]
            tag = tl.i32(tl.load(tile.tag))
            first = tl.i32(tl.load(tl.subview(tile.q, offsets=(0, 0), extents=(ROWS, 3))))
            first += tl.i32(tl.load(tl.subview(tile.s, offsets=(0, 0), extents=(ROWS, 3)))) * 16
            second = tl.i32(tl.load(tl.subview(tile.q, offsets=(0, 3), extents=(ROWS, 5))))
            second += tl.i32(tl.load(tl.subview(tile.s, offsets=(0, 3), extents=(ROWS, 5)))) * 16
            tl.store(tl.subview(Y[rows, block], offsets=(0, 0), extents=(ROWS, 3)), tag + first)
            tl.store(tl.subview(Y[rows, block], offsets=(0, 3), extents=(ROWS, 5)), tag + second)


@triton.jit(model="weft")
def cohort_indexed_fields(X: tl.View[NaturalRows[4], (M, K)], Y: tl.View[tl.i32, (M, K)], ROWS: tl.constexpr = 4):
    with tl.level.rows(M, group=ROWS) as rows:
        with tl.level.blocks(K, extent=8) as block:
            tile = X[rows, block]
            tag = tl.i32(tl.load(tile.tag))
            for index in tl.static_range(8):
                result = tag + tl.i32(tl.load(tile.q[:, index]))
                result += tl.i32(tl.load(tile.s[:, index])) * 16
                tl.store(Y[rows, block][:, index], result)


@triton.jit(model="weft")
def cohort_index_slice(X: tl.View[NaturalRows[4], (M, K)], Y: tl.View[tl.i32, (N, K)],
                       First: tl.constexpr, ROWS: tl.constexpr):
    tile = tl.subview(X, offsets=(First, 0), extents=(ROWS, 8))
    destination = tl.subview(Y, offsets=(0, 0), extents=(ROWS, 8))
    tag = tl.i32(tl.load(tile.tag))
    for index in tl.static_range(8):
        result = tag + tl.i32(tl.load(tile.q[:, index]))
        result += tl.i32(tl.load(tile.s[:, index])) * 16
        tl.store(destination[:, index], tl.reshape(result, (ROWS,), axes=("N",), order=("M",)))


@triton.jit(model="weft")
def cohort_slice(X: tl.View[NaturalRows[4], (M, K)], Y: tl.View[tl.i32, (N, K)], First: tl.i32, ROWS: tl.constexpr = 4):
    tile = tl.subview(X, offsets=(First, 0), extents=(ROWS, 8))
    result = tl.i32(tl.load(tile.tag)) + tl.i32(tl.load(tile.q)) + tl.i32(tl.load(tile.s))*16
    destination = tl.subview(Y, offsets=(0, 0), extents=(ROWS, 8))
    tl.store(destination, tl.reshape(result, (ROWS, 8), axes=("N", "K"), order=("M", "K")))


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def natural_fields(X: tl.View[NaturalLE, (M, K)], Y: tl.View[NaturalLE, (M, K)],
                   Observed: tl.View[tl.i32, (M, K)]):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.blocks(K, extent=8) as block:
            saved = tl.stage(tl.load(X[rows, block]))
            tl.store(X[rows, block].q, saved.q ^ tl.i4(1))
            tl.store(X[rows, block].s, saved.s ^ tl.u6(32))
            tl.store(Y[rows, block].tag, tl.load(X[rows, block].tag) + tl.u16(1))
            tl.store(Y[rows, block].q, tl.load(X[rows, block].q))
            tl.store(Y[rows, block].s, tl.load(X[rows, block].s))
            tl.store(Observed[rows, block], tl.i32(saved.tag) + tl.i32(saved.q) + tl.i32(saved.s)*16)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def layered_joined_fields(X: tl.View[LayeredJoined, (M, K)], Y: tl.View[LayeredJoined, (M, K)],
                          Observed: tl.View[tl.i32, (M, K)]):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.blocks(K, extent=8) as block:
            saved = tl.stage(tl.load(X[rows, block]))
            tl.store(X[rows, block].q, saved.q ^ tl.i4(1))
            tl.store(X[rows, block].s, saved.s ^ tl.u6(32))
            tl.store(X[rows, block].m, saved.m ^ tl.u6(16))
            tl.store(Y[rows, block].tag, tl.load(X[rows, block].tag) + tl.u16(1))
            tl.store(Y[rows, block].q, tl.load(X[rows, block].q))
            tl.store(Y[rows, block].s, tl.load(X[rows, block].s))
            tl.store(Y[rows, block].m, tl.load(X[rows, block].m))
            tl.store(Observed[rows, block], tl.i32(saved.tag) + tl.i32(saved.q) +
                     tl.i32(saved.s)*16 + tl.i32(saved.m)*1024)


@triton.jit(model="weft", alias_groups={"X": "input", "Y": "output", "Observed": "observed"})
def ordered_fields(X: tl.View[BigOrder, (M, K)], Y: tl.View[BigOrder, (M, K)],
                   Observed: tl.View[tl.i32, (M, K)]):
    with tl.level.rows(M, group=4) as rows:
        with tl.level.blocks(K, extent=8) as block:
            saved = tl.stage(tl.load(X[rows, block]))
            tl.store(X[rows, block].flags, saved.flags ^ tl.u1(1))
            tl.store(Y[rows, block].tag, tl.load(X[rows, block].tag) + tl.u16(1))
            tl.store(Y[rows, block].flags, tl.load(X[rows, block].flags))
            tl.store(Observed[rows, block], tl.i32(saved.tag) + tl.i32(saved.flags))

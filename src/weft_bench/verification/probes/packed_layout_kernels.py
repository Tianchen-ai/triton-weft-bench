"""Public Encoding declarations; shared field operations, without target branches."""
from __future__ import annotations

import triton
import triton.language as tl


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

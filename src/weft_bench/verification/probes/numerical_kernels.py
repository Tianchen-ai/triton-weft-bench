"""Ordinary Triton syntax used by the numerical-contract workloads."""
import triton
import triton.language as tl


@triton.jit(model="weft")
def pointer_exp(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, SCALAR: tl.constexpr = False):
    if SCALAR:
        offsets = tl.program_id(0)
    else:
        offsets = tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0.0)
    tl.store(Y + offsets, tl.exp(value), offsets < N)


@triton.jit(model="weft")
def integer_to_float(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, value.to(Y.dtype.element_ty), offsets < N)


@triton.jit(model="weft")
def cast_roundtrip(X, Y, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr, SCALAR: tl.constexpr = False):
    if SCALAR:
        offsets = tl.program_id(0)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0)
    if MODE == 0:
        result = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    elif MODE == 1:
        result = value.to(tl.float16, fp_downcast_rounding="rtz").to(tl.float32)
    elif MODE == 2:
        result = value.to(tl.float64).to(tl.float32)
    elif MODE == 3:
        result = value.to(tl.int1).to(tl.float32)
    elif MODE == 4:
        result = value.to(tl.int32).to(tl.float32)
    elif MODE == 5:
        result = value.to(tl.uint32, bitcast=True).to(tl.float32, bitcast=True)
    elif MODE == 6:
        result = value.to(tl.float16).to(tl.int32).to(tl.float32)
    elif MODE == 7:
        result = value.to(tl.int32, fp_downcast_rounding="rtne").to(tl.float32)
    elif MODE == 8:
        result = value.to(tl.float64, fp_downcast_rounding="rtz").to(tl.float32)
    elif MODE == 9:
        result = value.to(tl.uint16, bitcast=True).to(tl.float32)
    elif MODE == 10:
        result = value.to(tl.int32).to(tl.int8).to(tl.float32)
    elif MODE == 11:
        result = value.to(tl.int32).to(tl.uint8).to(tl.float32)
    elif MODE == 12:
        result = value.to(tl.int32).to(tl.uint32).to(tl.float32)
    elif MODE == 13:
        result = value.to(tl.int32).to(tl.uint64).to(tl.int64).to(tl.int32).to(tl.float32)
    elif MODE == 14:
        result = value.to(tl.int32).to(tl.uint16).to(tl.float32)
    elif MODE == 15:
        result = value.to(tl.int1).to(tl.uint8).to(tl.float32)
    elif MODE == 16:
        result = value.to(tl.int32).to(tl.uint16).to(tl.int8).to(tl.float32)
    elif MODE == 17:
        result = value.to(tl.int32).to(tl.int16).to(tl.uint8).to(tl.float32)
    else:
        tl.static_assert(False, "unknown numerical contract mode")
    tl.store(Y + offsets, result, offsets < N)

"""Pointer address contracts and ordinary Triton dot options."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(model="weft")
def affine_pointer_control(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0.0)
    tl.store(Y + offsets, (values + 3.0) * 0.25, offsets < N)


@triton.jit(model="weft")
def permuted_control(X, Y, N: tl.constexpr, WRAP: tl.constexpr):
    offsets = tl.arange(0, N)
    if WRAP:
        # The cast crosses the signed-byte boundary. Its widened addresses are
        # a permutation, not an affine interval in the destination format.
        offsets = (offsets + 120).to(tl.int8).to(tl.int64) + 128
    else:
        offsets = N - 1 - offsets
    values = tl.load(X + offsets)
    tl.store(Y + offsets, (values + 3.0) * 0.25)


@triton.jit(model="weft")
def standard_dot_options(A: tl.View[tl.f32, (16, 32)], B: tl.View[tl.f32, (32, 16)],
                         Acc: tl.View[tl.f32, (16, 16)], Y: tl.View[tl.f32, (16, 16)],
                         MODE: tl.constexpr):
    a = tl.load(A)
    b = tl.load(B)
    acc = tl.load(Acc)
    if MODE == 0:
        value = tl.dot(a, b, acc, input_precision="IEEE")
    elif MODE == 1:
        # Upstream retains fp32 accumulation for fp32 operands.
        value = tl.dot(a, b, out_dtype=tl.float16)
        tl.static_assert(value.dtype == tl.float32)
        value += acc
    elif MODE == 2:
        value = tl.dot(a, b, acc, out_dtype=tl.bfloat16)
    elif MODE == 4:
        value = tl.dot(a, b, acc, max_num_imprecise_acc=4)
    else:
        value = tl.dot(a, b, acc, input_precision="tf32")
    tl.store(Y, value)

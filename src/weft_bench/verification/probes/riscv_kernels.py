"""Standard Triton scans/atomics and helper control flow on the Weft target."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def scan_add(left, right):
    return left + right


@triton.jit
def scan_pair(left_sum, left_max, right_sum, right_max):
    return left_sum + right_sum, tl.maximum(left_max, right_max)


@triton.jit
def scan_affine_combine(left_a, left_b, right_a, right_b):
    return left_a * right_a, left_b * right_a + right_b


@triton.jit(model="weft")
def scan_sum(X, Y, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, REVERSE: tl.constexpr):
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    values = tl.load(X + offsets)
    tl.store(Y + offsets, tl.associative_scan(values, AXIS, scan_add, reverse=REVERSE))


@triton.jit(model="weft")
def scan_affine(X, Values, Y, Z, M: tl.constexpr, N: tl.constexpr,
                AXIS: tl.constexpr, REVERSE: tl.constexpr):
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    a, b = tl.load(X + offsets), tl.load(Values + offsets)
    result_a, result_b = tl.associative_scan((a, b), AXIS, scan_affine_combine, reverse=REVERSE)
    tl.store(Y + offsets, result_a)
    tl.store(Z + offsets, result_b)


@triton.jit(model="weft")
def scan_axes(X, Y, Z, M: tl.constexpr, N: tl.constexpr,
              AXIS: tl.constexpr, REVERSE: tl.constexpr, TUPLE: tl.constexpr):
    rows = tl.arange(0, M)
    columns = tl.arange(0, N)
    offsets = rows[:, None] * N + columns[None, :]
    values = tl.load(X + offsets)
    if TUPLE:
        summed, maximum = tl.associative_scan((values, values), AXIS, scan_pair, reverse=REVERSE)
        tl.store(Z + offsets, maximum)
    else:
        summed = tl.associative_scan(values, AXIS, scan_add, reverse=REVERSE)
    tl.store(Y + offsets, summed)


@triton.jit(model="weft")
def atomic_rmw(State, Values, Old, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    value = tl.load(Values + offsets, mask, other=0)
    if OP == 0:
        old = tl.atomic_add(State + offsets, value, mask=mask, sem="relaxed")
    elif OP == 1:
        old = tl.atomic_max(State + offsets, value, mask=mask)
    elif OP == 2:
        old = tl.atomic_min(State + offsets, value, mask=mask)
    elif OP == 3:
        old = tl.atomic_and(State + offsets, value, mask=mask)
    elif OP == 4:
        old = tl.atomic_or(State + offsets, value, mask=mask)
    elif OP == 5:
        old = tl.atomic_xor(State + offsets, value, mask=mask)
    else:
        old = tl.atomic_xchg(State + offsets, value, mask=mask)
    tl.store(Old + offsets, old, mask)


@triton.jit(model="weft")
def atomic_cas_slots(State, Compare, Values, Old, N: tl.constexpr):
    index = tl.program_id(0)
    # atomic_cas has no mask argument in Triton; guard the scalar invocation.
    if index < N:
        old = tl.atomic_cas(State + index, tl.load(Compare + index), tl.load(Values + index))
        tl.store(Old + index, old)


@triton.jit(model="weft")
def atomic_contended(State, Old, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    old = tl.atomic_add(State + offsets % 2, 1, mask=offsets < N, sem="relaxed")
    tl.store(Old + offsets, old, mask=offsets < N)


@triton.jit(model="weft")
def atomic_cas_contended(State, Old):
    index = tl.program_id(0)
    old = tl.atomic_cas(State, 0, index + 1)
    tl.store(Old + index, old)


@triton.jit(noinline=True)
def noinline_adjust(value, mode):
    if mode < 0:
        return value + 1.0
    return value + 2.0


@triton.jit
def return_from_for(value, stop):
    total = value
    for step in range(4):
        total += 1.0
        if step == stop:
            return total
        total += 2.0
    return total + 10.0


@triton.jit
def return_from_while(value, stop):
    total = value
    step = 0
    while step < 4:
        total += 1.0
        if step == stop:
            return total
        total += 2.0
        step += 1
    return total + 10.0


@triton.jit
def return_from_level(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], stop):
    for step in range(3):
        with tl.level.blocks(N, extent=64) as block:
            values = tl.load(X[block])
            tl.store(Y[block], values + step + 1.0)
            if step == stop:
                return
        # A return must leave both the level and the containing loop.
    with tl.level.blocks(N, extent=64) as block:
        tl.store(Y[block], tl.load(X[block]) + 10.0)


@triton.jit(model="weft")
def helper_returns(X, Y, N: tl.constexpr, MODE: tl.constexpr):
    index = tl.program_id(0)
    if index < N:
        value = tl.load(X + index)
        stop = index % 6 - 1
        if MODE == 0:
            result = return_from_for(value, stop)
        elif MODE == 1:
            result = return_from_while(value, stop)
        else:
            result = noinline_adjust(value, index % 2 - 1)
        # The helper returns to this caller, not out of the entry kernel.
        tl.store(Y + index, result * 2.0)


@triton.jit(model="weft")
def helper_context_return(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)],
                          Flag: tl.View[tl.i32, (1,)], stop: tl.i32):
    return_from_level(X, Y, stop)
    tl.store(Flag[0], tl.i32(77))

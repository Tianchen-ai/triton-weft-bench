"""Helper function regions checked with the existing pointer_control workload."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit(noinline=True)
def scalar_adjust(value, mode):
    if mode < 0:
        return value + 1.0
    if mode == 0:
        return value + 2.0
    return value + 3.0


@triton.jit(noinline=True)
def shaped_adjust(value, mode):
    if mode == 0:
        return value + 1.0
    return value + 2.0


@triton.jit(noinline=True)
def shaped_pair(value, mode):
    if mode == 0:
        return value + 1.0, value + 5.0
    return value + 2.0, value + 4.0


@triton.jit(model="weft")
def scalar_entry(X, Y, N: tl.constexpr):
    index = tl.program_id(0)
    if index < N:
        mode = index % 3 - 1
        value = scalar_adjust(tl.load(X + index), mode)
        tl.store(Y + index, (value + (1-mode).to(tl.float32)) * 0.25)


@triton.jit(model="weft")
def shaped_entry(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, TUPLE: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, other=0)
    mode = tl.program_id(0) % 2
    if TUPLE:
        left, right = shaped_pair(value, mode)
        result = (left + right) * 0.125
    else:
        result = (shaped_adjust(value, mode) + (2-mode).to(tl.float32)) * 0.25
    tl.store(Y + index, result, index < N)


@triton.jit
def loop_for(value, stop):
    total = value
    for step in range(4):
        total += 1.0
        if step == stop:
            return total
        total += 2.0
    return total + 10.0


@triton.jit(noinline=True)
def loop_while(value, stop):
    total = value
    step = 0
    while step < 4:
        total += 1.0
        if step == stop:
            return total
        total += 2.0
        step += 1
    return total + 10.0


@triton.jit(model="weft")
def loop_entry(X, Y, N: tl.constexpr, WHILE: tl.constexpr):
    index = tl.program_id(0)
    if index < N:
        stop = index % 6 - 1
        value = tl.load(X + index)
        if WHILE:
            value = loop_while(value, stop)
        else:
            value = loop_for(value, stop)
        offset = 22
        if stop >= 0 and stop < 4:
            offset = 3 * stop + 1
        tl.store(Y + index, (value - offset.to(tl.float32) + 3.0) * 0.25)


@triton.jit
def level_return(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], stop):
    for step in range(3):
        with tl.level.blocks(N, extent=256) as block:
            tl.store(Y[block], tl.load(X[block]) + step + 1.0)
            if step == stop:
                return
    with tl.level.blocks(N, extent=256) as block:
        tl.store(Y[block], tl.load(X[block]) + 10.0)


@triton.jit(model="weft")
def level_entry(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)], OFFSET: tl.constexpr):
    stop = tl.program_id(0) + OFFSET
    level_return(X, Y, stop)
    offset = 10
    if stop >= 0 and stop < 3:
        offset = stop + 1
    with tl.level.blocks(N, extent=256) as block:
        # This store is a caller effect after the helper's return.
        tl.store(Y[block], (tl.load(Y[block]) - offset.to(tl.float32) + 3.0) * 0.25)

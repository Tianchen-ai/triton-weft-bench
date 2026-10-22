"""Public Level lifetime programs with ordinary pointer/scalar arguments."""
import triton
import triton.language as tl


@triton.jit(model="weft", do_not_specialize=["N"])
def level_births(X, Y, N):
    slot = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=16) as block:
        local = tl.state(tl.f32, [], init=tl.load(X))
        saved = tl.stage(tl.load(X))
        tl.store(X, tl.load(X) + 64.0)
        local = local + 1.0
        tl.store(Y + slot, local + saved)
        slot += 1
    tl.store(Y + slot, slot.to(tl.float32))


@triton.jit(model="weft", do_not_specialize=["N"])
def level_staged_children(X, Y, N):
    slot = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=16) as parent:
        saved = tl.stage(tl.load(X))
        with tl.level.subtiles(parent, extent=4) as child:
            tl.store(X, tl.load(X) + 1.0)
            tl.store(Y + slot, saved)
            slot += 1
    tl.store(Y + slot, slot.to(tl.float32))


@triton.jit(model="weft", do_not_specialize=["N"])
def level_outer_carry(X, Y, N):
    total = tl.state(tl.f32, [], init=3.0)
    cursor = tl.full((), 0, tl.int32)
    visits = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=16) as block:
        panel = tl.stage(tl.load(X))
        total = total * 0.5 + panel
        cursor += 16
        visits += 1
    tl.store(Y, total)
    tl.store(Y + 1, visits.to(tl.float32))
    tl.store(Y + 2, cursor.to(tl.float32))


@triton.jit(model="weft", do_not_specialize=["N"])
def level_subtile_tail(X, Y, N):
    cursor = tl.full((), 0, tl.int32)
    visits = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=16) as parent:
        offset = tl.full((), 0, tl.int32)
        with tl.level.subtiles(parent, extent=6) as child:
            lane = tl.arange(0, 8)
            index = cursor + offset + lane
            valid = (lane < 6) & (offset + lane < 16) & (index < N)
            value = tl.load(X + index, valid, other=0.0)
            tl.store(Y + index, value + 1.0, valid)
            visits += 1
            offset += 6
        cursor += 16
    tl.store(Y + N, visits.to(tl.float32))


@triton.jit(model="weft", do_not_specialize=["N", "STEPS", "MODE"])
def level_outer_control(X, Y, N, STEPS, MODE):
    total = tl.state(tl.f32, [], init=1.0)
    visits = tl.full((), 0, tl.int32)
    if MODE > 0:
        for iteration in range(STEPS):
            with tl.level.blocks(N, extent=16) as block:
                local = tl.state(tl.f32, [], init=tl.load(X))
                bias = tl.stage(iteration.to(tl.float32))
                total = total * 0.5 + local + bias
                visits += 1
    else:
        iteration = STEPS - STEPS
        while iteration < STEPS:
            with tl.level.blocks(N, extent=16) as block:
                local = tl.state(tl.f32, [], init=tl.load(X))
                bias = tl.stage(iteration.to(tl.float32))
                total = total * 0.5 + local + bias
                visits += 1
            iteration += 1
    tl.store(Y, total)
    tl.store(Y + 1, visits.to(tl.float32))
    tl.store(Y + 2, MODE.to(tl.float32))


@triton.jit(model="weft", do_not_specialize=["N", "STEPS"])
def level_inner_control(X, Y, N, STEPS, ROWS: tl.constexpr):
    # Reusing an ordinary loop's induction name inside a later Level must not
    # import that completed loop's region-local placeholder as Level state.
    for iteration in range(STEPS):
        tl.store(Y, iteration.to(tl.float32))
    slot = tl.full((), 0, tl.int32)
    with tl.level.blocks(N, extent=16) as block:
        local = tl.state(tl.f32, [ROWS, 4], init=1.0)
        panel = tl.stage(tl.load(X))
        for iteration in range(STEPS):
            if iteration % 2 == 0:
                local = local * 2.0 + panel
            else:
                local = local - 0.5
            local[iteration % ROWS, iteration % 4] = local[iteration % ROWS, iteration % 4] + panel
        iteration = tl.full((), 0, tl.int32)
        while iteration < 2:
            local = local * 0.5 + 1.0
            iteration += 1
        row = local[ROWS - 1]
        for lane in range(4):
            for r in range(ROWS - 1):
                tl.store(Y + slot * ROWS * 4 + r * 4 + lane, local[r, lane])
            tl.store(Y + slot * ROWS * 4 + (ROWS - 1) * 4 + lane, row[lane])
        tl.store(X, tl.load(X) + 2.0)
        slot += 1
    tl.store(Y + slot * ROWS * 4, slot.to(tl.float32))

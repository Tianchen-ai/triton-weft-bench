"""Independent small NumPy oracles for public Level lifetime programs."""
import numpy as np

SENTINEL = -777.0
SCENARIOS = ("births", "staged_children", "outer_carry", "subtile_tail",
             "zero_extent", "outer_control", "inner_control")


def scenarios(name):
    if name not in SCENARIOS:
        raise ValueError("unknown Level scenario " + name)
    variants = (("for", 3, 1), ("while", 2, 0)) if name == "outer_control" else ((name, 3, 1),)
    if name == "inner_control":
        variants = (("two-rows", 3, 2), ("three-rows", 3, 3), ("empty-for", 0, 3))
    results = []
    for label, steps, mode in variants:
        n = 0 if name == "zero_extent" else 37
        x = np.arange(64, dtype=np.float32) / 8 + 1
        expected_x = x.copy()
        expected_y = np.full(128, SENTINEL, dtype=np.float32)
        scalars = {"N": n}
        symbol = "level_" + ("outer_carry" if name == "zero_extent" else name)
        visits, values = 0, []
        if name == "births":
            for cursor in range(0, n, 16):
                values.append(float(expected_x[0]) * 2 + 1)
                expected_x[0] += 64
            values.append(len(values))
        elif name == "staged_children":
            for cursor in range(0, n, 16):
                count = (min(16, n-cursor) + 3)//4
                values.extend([float(expected_x[0])] * count)
                expected_x[0] += count
            values.append(len(values))
        elif name in ("outer_carry", "zero_extent"):
            total = 3.0
            for cursor in range(0, n, 16):
                total = total * 0.5 + float(x[0])
                visits += 1
            values = [total, visits, visits * 16]
        elif name == "subtile_tail":
            values = (x[:n] + 1).tolist()
            values.append(sum((min(16, n-cursor) + 5)//6 for cursor in range(0, n, 16)))
        elif name == "outer_control":
            scalars.update(STEPS=steps, MODE=mode)
            total = 1.0
            for iteration in range(steps):
                for cursor in range(0, n, 16):
                    total = total * 0.5 + float(x[0]) + iteration
                    visits += 1
            values = [total, visits, mode]
        else:
            rows = mode
            scalars.update(STEPS=steps, ROWS=rows)
            for cursor in range(0, n, 16):
                local = np.ones((rows, 4), dtype=np.float64)
                panel = float(expected_x[0])
                for iteration in range(steps):
                    local = local * 2 + panel if iteration % 2 == 0 else local - 0.5
                    local[iteration % rows, iteration % 4] += panel
                for _ in range(2):
                    local = local * 0.5 + 1.0
                values.extend(local.ravel().tolist())
                expected_x[0] += 2
            values.append(len(values)//(rows*4))
        expected_y[:len(values)] = values
        results.append({"name": label, "symbol": symbol, "scalars": scalars,
                        "input": x, "expected_input": expected_x, "expected_output": expected_y,
                        "output_elements": len(values)})
    return results

"""Scan and atomic IO/oracles for the existing pointer native worker."""
from ..types import Case

SCANS = {
    "scan_rows": (1, False, 0), "scan_columns": (0, False, 0),
    "scan_reverse_rows": (1, True, 0), "scan_reverse_columns": (0, True, 0),
    "scan_tuple_reverse": (1, True, 1), "scan_affine_reverse": (1, True, 2),
    "scan_affine_reverse_columns": (0, True, 2),
}
ATOMICS = {"atomic_" + name: op for op, name in enumerate(("add", "max", "min", "and", "or", "xor", "xchg"))}
ATOMICS.update(atomic_cas=7, atomic_contention=8, atomic_cas_contention=9)


def catalog():
    cases = {}
    for operation, (axis, reverse, kind) in SCANS.items():
        name = "pointer_" + operation
        cases[name] = Case(name, "pointer_semantics", {"M": 4, "N": 8}, "complete-" + operation,
                           input_policy="small-signed-integers", dtype="i32",
                           data={"riscv": True, "operation": operation, "axis": axis, "reverse": reverse, "kind": kind})
    for operation, kind in ATOMICS.items():
        name = "pointer_" + operation
        shape = {"N": 53, "CAPACITY": 64} if kind == 8 else {"N": 16, "CAPACITY": 16} if kind == 9 else {"N": 27, "CAPACITY": 32}
        cases[name] = Case(name, "pointer_semantics", shape, "complete-" + operation,
                           input_policy="signed-state-and-values", dtype="i32",
                           data={"riscv": True, "operation": operation, "kind": kind})
    return cases


def ports(case):
    from ..workloads import TensorSpec
    kind = case.data["kind"]
    if case.data["operation"] in SCANS:
        shape = (case.shape["M"], case.shape["N"])
        result = [TensorSpec("X", "i32", shape), TensorSpec("Y", "i32", shape, "output")]
        if kind:
            result.append(TensorSpec("Z", "i32", shape, "output"))
        if kind == 2:
            result.append(TensorSpec("Values", "i32", shape))
        return result
    capacity = case.shape["CAPACITY"]
    state_size = 2 if kind == 8 else 1 if kind == 9 else capacity
    result = [TensorSpec("State", "i32", (state_size,), "inout"),
              TensorSpec("Old", "i32", (capacity,), "output")]
    if kind < 8:
        result.append(TensorSpec("Values", "i32", (capacity,)))
    if kind == 7:
        result.append(TensorSpec("Compare", "i32", (capacity,)))
    return result


def workload(case):
    from ..core import NativeWorkload
    kind = case.data["kind"]
    if case.data["operation"] in SCANS:
        return NativeWorkload("riscv_inputs.h", "RiscvScanInputs data(inputs);",
            {"X": "data.x.data()", "Values": "data.values.data()", "Y": "output.data()", "Z": "output.data()+data.size"},
            output_type="int32_t", defines=tuple("-DWB_SCAN_" + key + "=" + str(value) for key, value in
                {**case.shape, "AXIS": case.data["axis"], "REVERSE": int(case.data["reverse"]), "KIND": kind}.items()),
            reset="std::fill(output.begin(),output.end(),int32_t(-999));", correctness="exact-integer-prefix-scan",
            snapshots={"X": "data.x", "Values": "data.values"})
    return NativeWorkload("riscv_inputs.h", "RiscvAtomicInputs data(inputs);",
        {"State": "data.state.data()", "Values": "data.values.data()", "Compare": "data.compare.data()", "Old": "output.data()"},
        output_type="int32_t", defines=tuple("-DWB_ATOMIC_"+key+"="+str(value) for key,value in
            {**case.shape, "KIND": kind}.items()),
        reset="data.reset(); std::fill(output.begin(),output.end(),int32_t(-999));",
        correctness="exact-atomic-state-and-returned-old-values", snapshots={"State": "data.state", "initial-State": "data.initial"})

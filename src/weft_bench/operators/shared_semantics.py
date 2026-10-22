"""Small complete-call workloads for the shared Triton numerical frontend."""
from ..types import Case

OPERATIONS = {
    "arange": {"N": 128},
    "sum": {"M": 4, "N": 128},
    "max": {"M": 4, "N": 128},
    "softmax": {"M": 4, "N": 128},
    "dot_acc": {"M": 16, "N": 16, "K": 32},
    "batch_dot_acc": {"B": 2, "M": 16, "N": 16, "K": 32},
    "for_while": {"N": 128},
    "nested_level": {"M": 4, "N": 128},
}


def catalog():
    return {"shared_" + operation: Case("shared_" + operation, "shared_semantics", dict(shape),
            "complete-" + operation, input_policy="finite-uniform", data={"operation": operation})
            for operation, shape in OPERATIONS.items()}


def ports(case):
    from ..workloads import TensorSpec

    shape, operation = case.shape, case.data["operation"]
    m, n, k = shape.get("M", 1), shape["N"], shape.get("K", 1)
    if operation in ("dot_acc", "batch_dot_acc"):
        prefix = (shape["B"],) if operation == "batch_dot_acc" else ()
        return [TensorSpec("A", "f32", (*prefix, m, k)), TensorSpec("B", "f32", (*prefix, k, n)),
                TensorSpec("Acc", "f32", (*prefix, m, n)), TensorSpec("Y", "f32", (*prefix, m, n), "output")]
    input_shape = (m, n) if "M" in shape else (n,)
    output_shape = (m,) if operation in ("sum", "max") else input_shape
    return [TensorSpec("X", "f32", input_shape), TensorSpec("Y", "f32", output_shape, "output")]


def workload(case):
    from ..core import NativeWorkload

    kind = list(OPERATIONS).index(case.data["operation"]) + 1
    defines = (f"-DWB_SHARED_KIND={kind}", *(f"-DWB_SHARED_{key}={case.shape.get(key, 1)}" for key in ("B", "M", "N", "K")))
    return NativeWorkload("shared_semantics_inputs.h", "SharedSemanticsInputs data(inputs);",
        {"X": "data.x.data()", "A": "data.a.data()", "B": "data.b.data()", "Acc": "data.acc.data()", "Y": "output.data()"},
        defines=defines, data_type="SharedSemanticsInputs")

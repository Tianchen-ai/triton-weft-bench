"""Workload inputs independent of pointer-based Triton source implementations."""
from dataclasses import replace
from ..types import Case

OPERATIONS = {
    "add": {"N": 128, "SX": 1, "SZ": 1, "SY": 1},
    "tail": {"N": 131, "SX": 1, "SZ": 1, "SY": 1},
    "strided": {"N": 131, "SX": 2, "SZ": 3, "SY": 2},
    "sum": {"M": 4, "N": 117},
    "argmax": {"M": 4, "N": 117},
    "product": {"M": 4, "N": 117},
    "gather_join_split": {"M": 4, "N": 8, "G0": 2, "G1": 4},
    "softmax": {"M": 4, "N": 117},
    "dot": {"M": 16, "N": 16, "K": 32},
    "control": {"N": 128},
    "alias": {"N": 117},
    "mask_guard": {"N": 117},
    "carry": {"N": 32, "STEPS": 3},
    "select": {"N": 128},
    "bool": {"N": 131},
}
CAST_MODES = {"cast_fp16_rtne": 0, "cast_fp16_rtz": 1, "cast_fp64": 2,
              "cast_bool": 3, "cast_int32": 4, "cast_bitcast": 5, "cast_fp16_int32": 6,
              "cast_int8": 10, "cast_uint8": 11, "cast_uint32": 12,
              "cast_signed_unsigned_widen": 13, "cast_uint16": 14, "cast_bool_uint8": 15,
              "cast_unsigned_signed_narrow": 16, "cast_signed_unsigned_narrow": 17}
OPERATIONS.update({name: {"N": 37} for name in CAST_MODES})
GUARD = 128


def catalog():
    policies = {"bool": "boolean-0-or-1", "argmax": "finite-with-argmax-ties",
                "product": "near-unit-product-with-sign-and-zero",
                "gather_join_split": "finite-with-repeated-nonmonotonic-indices"}
    policies.update({name: "integer-conversion-boundaries" if mode in (16, 17) else
                     "finite-fractional-conversion" if mode in (4, 6, 10, 11, 12, 13, 14) else
                     "floating-conversion-boundaries-with-nan-inf-signed-zero"
                     for name, mode in CAST_MODES.items()})
    cases = {"pointer_" + operation: Case("pointer_" + operation, "pointer_semantics", dict(shape),
            "complete-" + operation, input_policy=policies.get(operation, "finite-uniform"),
            dtype="i1" if operation == "bool" else "f32", data={"operation": operation})
            for operation, shape in OPERATIONS.items()}
    from .riscv_semantics import catalog as riscv_catalog
    cases.update(riscv_catalog())
    return cases


def _shared(case):
    operation = {"dot": "dot_acc", "control": "for_while"}.get(case.data["operation"], case.data["operation"])
    return replace(case, family="shared_semantics", data={"operation": operation})


def ports(case):
    from ..workloads import TensorSpec
    from .shared_semantics import ports as shared_ports

    if case.data.get("riscv"):
        from .riscv_semantics import ports as riscv_ports
        return riscv_ports(case)

    if case.data["operation"] in ("sum", "softmax", "dot", "control"):
        return shared_ports(_shared(case))
    if case.data["operation"] == "gather_join_split":
        m, n, g0, g1 = (case.shape[key] for key in ("M", "N", "G0", "G1"))
        return [TensorSpec("X", "f32", (m, n)), TensorSpec("I0", "i32", (g0, n)),
                TensorSpec("I1", "i32", (m, g1)), TensorSpec("Y0", "f32", (g0, n), "output"),
                TensorSpec("Y1", "f32", (m, g1), "output"), TensorSpec("YNeg", "f32", (m, g1), "output")]
    if case.data["operation"] in ("argmax", "product"):
        m, n = case.shape["M"], case.shape["N"]
        result = [TensorSpec("X", "f32", (m, n)), TensorSpec("Y", "f32", (m,), "output")]
        if case.data["operation"] == "argmax":
            result.append(TensorSpec("Index", "i32", (m,), "output"))
        return result
    if case.data["operation"] == "alias":
        # Ports describe the logical span at each pointer. The native fixture
        # owns the larger shared allocation and checks its surrounding guards.
        size = case.shape["N"]
        return [TensorSpec("X", "f32", (size,), layout="shared-storage-offset-1"),
                TensorSpec("Y", "f32", (size,), "output", "same-storage-offset-2")]
    if case.data["operation"] == "mask_guard":
        return [TensorSpec(name, "f32", (case.shape["N"],), access, "ends-at-protected-page")
                for name, access in (("X", "input"), ("Z", "input"), ("Y", "output"))]
    if case.data["operation"] == "carry":
        return [TensorSpec("X", "f32", (case.shape["STEPS"], case.shape["N"])),
                TensorSpec("Y", "f32", (case.shape["N"],), "output")]
    if case.data["operation"] == "select":
        return [TensorSpec(name, "f32", (case.shape["N"],), access)
                for name, access in (("X", "input"), ("Z", "input"), ("Y", "output"))]
    if case.data["operation"] == "bool":
        return [TensorSpec("X", "i1", (case.shape["N"] + GUARD + 1,), layout="offset-1-with-guards"),
                TensorSpec("Y", "i1", (case.shape["N"] + GUARD + 1,), "output", "offset-1-with-guards")]
    if case.data["operation"] in CAST_MODES:
        return [TensorSpec("X", "f32", (case.shape["N"],)),
                TensorSpec("Y", "f32", (case.shape["N"],), "output")]
    # Physical storage includes a prefix and tail guard; N is the logical length.
    n = case.shape["N"]
    return [TensorSpec(name, "f32", (1 + (n + GUARD) * case.shape[stride],), access,
                       "offset-1-strided-with-guards")
            for name, stride, access in (("X", "SX", "input"), ("Z", "SZ", "input"), ("Y", "SY", "output"))]


def workload(case):
    from ..core import NativeWorkload
    from .shared_semantics import workload as shared_workload

    if case.data.get("riscv"):
        from .riscv_semantics import workload as riscv_workload
        return riscv_workload(case)

    scalars = {key: str(value) for key, value in case.shape.items()}
    if case.data["operation"] in CAST_MODES:
        return NativeWorkload("numeric_cast_inputs.h", "NumericCastInputs data(inputs);",
            {"X": "data.x.data()", "Y": "output.data()"}, scalars=scalars,
            defines=(f"-DWB_CAST_N={case.shape['N']}", f"-DWB_CAST_MODE={CAST_MODES[case.data['operation']]}"),
            data_type="NumericCastInputs", correctness="declared-conversion-rounding-and-special-values")
    if case.data["operation"] in ("sum", "softmax", "dot", "control"):
        return replace(shared_workload(_shared(case)), scalars=scalars)
    if case.data["operation"] == "gather_join_split":
        return NativeWorkload("pointer_state_inputs.h", "PointerGatherInputs data(inputs);",
            {"X": "data.x.data()", "I0": "data.indices0.data()", "I1": "data.indices1.data()",
             "Y0": "output.data()", "Y1": "output.data()+data.row_size",
             "YNeg": "output.data()+data.row_size+data.column_size"},
            defines=tuple(f"-DWB_POINTER_{key}={value}" for key, value in {**case.shape, "GUARD": GUARD}.items()),
            scalars=scalars, correctness="within-tolerance-gather-and-join-split")
    if case.data["operation"] in ("argmax", "product"):
        argmax = case.data["operation"] == "argmax"
        return NativeWorkload("pointer_state_inputs.h",
            ("PointerArgmaxInputs" if argmax else "PointerProductInputs") + " data(inputs);",
            {"X": "data.x.data()", "Y": "output.data()", **({"Index": "data.indices.data()"} if argmax else {})},
            defines=(f"-DWB_POINTER_M={case.shape['M']}", f"-DWB_POINTER_N={case.shape['N']}",
                     f"-DWB_POINTER_GUARD={GUARD}"), scalars=scalars,
            reset="std::fill(data.indices.begin(),data.indices.end(),int32_t(-1));" if argmax else "",
            correctness="exact-max-and-first-index" if argmax else "within-tolerance")
    if case.data["operation"] == "alias":
        return NativeWorkload("pointer_state_inputs.h", "PointerAliasInputs data(inputs);",
            {"X": "data.state.data()+1", "Y": "data.state.data()+2"},
            defines=(f"-DWB_POINTER_N={case.shape['N']}", f"-DWB_POINTER_GUARD={GUARD}"), scalars=scalars,
            reset="data.reset();", collect="output=data.state;",
            correctness="within-tolerance-and-preserved-alias-storage")
    if case.data["operation"] == "mask_guard":
        return NativeWorkload("pointer_state_inputs.h", "PointerGuardInputs data(inputs);",
            {"X": "data.x.data()", "Z": "data.z.data()", "Y": "data.y.data()"},
            defines=(f"-DWB_POINTER_N={case.shape['N']}", f"-DWB_POINTER_GUARD={GUARD}"), scalars=scalars,
            collect="std::copy_n(data.y.data(),data.output_size,output.begin());",
            correctness="within-tolerance-and-protected-mask-pages")
    if case.data["operation"] == "carry":
        return NativeWorkload("pointer_state_inputs.h", "PointerCarryInputs data(inputs);",
            {"X": "data.x.data()", "Y": "output.data()"},
            defines=(f"-DWB_POINTER_N={case.shape['N']}", f"-DWB_POINTER_GUARD={GUARD}",
                     f"-DWB_POINTER_STEPS={case.shape['STEPS']}"), scalars=scalars)
    if case.data["operation"] == "select":
        return NativeWorkload("pointer_state_inputs.h", "PointerSelectInputs data(inputs);",
            {"X": "data.x.data()", "Z": "data.z.data()", "Y": "output.data()"},
            defines=(f"-DWB_POINTER_N={case.shape['N']}", f"-DWB_POINTER_GUARD={GUARD}"),
            scalars={**scalars, "SELECT_X": "data.select_x"})
    if case.data["operation"] == "bool":
        return NativeWorkload("pointer_state_inputs.h", "PointerBooleanInputs data(inputs);",
            {"X": "data.x.data()+1", "Y": "output.data()+1"}, output_type="uint8_t",
            defines=(f"-DWB_POINTER_N={case.shape['N']}", f"-DWB_POINTER_GUARD={GUARD}"), scalars=scalars,
            reset="std::fill(output.begin(),output.end(),uint8_t(0xa5));",
            correctness="exact-boolean-and-unchanged-padding")
    return NativeWorkload("vector_add_inputs.h", "VectorAddInputs data(inputs);",
        {"X": "data.x.data()+1", "Z": "data.z.data()+1", "Y": "output.data()+1"},
        defines=tuple(f"-DWB_VECTOR_{key}={value}" for key, value in {**case.shape, "GUARD": GUARD}.items()),
        scalars=scalars,
        correctness="within-tolerance-and-unchanged-padding")

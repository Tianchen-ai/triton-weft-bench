"""Typed dense matmul contracts, independent of CPU libraries and compilers."""

from ..core import NativeWorkload
from ..types import Case


def catalog():
    return {
        name: Case(name, "matmul", {"M": 128, "N": 4096, "K": 4096}, "preallocated-gemm",
                   dtype=source + "*" + source + "->" + output,
                   data={"input_dtype": source, "output_dtype": output})
        for name, source, output in (("gemm_bf16_f32", "bf16", "f32"),
                                     ("gemm_s8_s32", "i8", "i32"))
    }


def ports(case):
    from ..workloads import TensorSpec

    s = case.shape
    return [TensorSpec("X", case.data["input_dtype"], (s["M"], s["K"])),
            TensorSpec("W", case.data["input_dtype"], (s["N"], s["K"])),
            TensorSpec("Y", case.data["output_dtype"], (s["M"], s["N"]), "output")]


def workload(case):
    integer = case.data["input_dtype"] == "i8"
    sampled = case.shape["M"] * case.shape["N"] > 4096
    return NativeWorkload(
        "matmul_inputs.h", "MatmulInputs data(inputs);",
        {"X": "data.input.data()", "W": "data.weight.data()", "Y": "output.data()"},
        output_type="int32_t" if integer else "float", data_type="MatmulInputs",
        correctness=("sampled-64-" if sampled else "") + ("exact" if integer else "within-tolerance"),
        defines=tuple(f"-DWB_{key}={value}" for key, value in case.shape.items()) +
                ("-DWB_MATMUL_INTEGER=" + str(int(integer)),),
    )

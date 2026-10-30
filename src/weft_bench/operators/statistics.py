"""Complete count/mean/M2 workload with an independent wide reference."""
from ..core import NativeWorkload
from ..types import Case


def catalog():
    return {"welford": Case("welford", "statistics", {"M": 128, "N": 4096},
                           "complete-streaming-moments", dtype="f32->count,mean,M2",
                           input_policy="finite-uniform-fixed-seed",
                           data={"zero_extent_axes": ["N"]})}


def ports(case):
    from ..workloads import TensorSpec
    return [TensorSpec("X", "f32", (case.shape["M"], case.shape["N"])),
            TensorSpec("Y", "f32", (case.shape["M"], 3), "output")]


def workload(case):
    if case.shape["M"] <= 0 or not 0 <= case.shape["N"] < (1 << 24):
        raise ValueError("statistics requires positive M and 0 <= N < 2**24")
    return NativeWorkload("statistics_inputs.h", "StatisticsInputs data(inputs);",
                          {"X": "data.input.data()", "Y": "output.data()"},
                          defines=tuple(f"-DWB_{key}={value}" for key, value in case.shape.items()),
                          data_type="StatisticsInputs", correctness="exact-count-and-full-wide-reference-rtol1e-4-atol1e-5")

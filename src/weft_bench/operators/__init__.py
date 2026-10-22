from dataclasses import dataclass, field, replace

from ..loading import load_object
from ..types import Case

FORMATS = (
    "q1_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_s",
    "iq2_xs",
    "iq2_xxs",
    "iq3_s",
    "iq3_xxs",
    "iq4_nl",
    "iq4_xs",
    "tq1_0",
    "tq2_0",
    "mxfp4",
    "nvfp4",
)
PARTNERS = (
    "q8_0",
    "q8_0",
    "q8_1",
    "q8_0",
    "q8_1",
    "q8_0",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_0",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_0",
    "q8_0",
)
DECODE_DEFINES = dict(
    zip(
        (
            "q1_0",
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q8_0",
            "q2_k",
            "q3_k",
            "q4_k",
            "q5_k",
            "q6_k",
            "iq1_s",
            "iq1_m",
            "iq2_xxs",
            "iq4_nl",
            "iq4_xs",
            "tq1_0",
            "tq2_0",
            "mxfp4",
            "nvfp4",
        ),
        (
            "Q10",
            "Q40",
            "Q41",
            "Q50",
            "Q51",
            "Q80",
            "Q2K",
            "Q3K",
            "Q4K",
            "Q5K",
            "Q6K",
            "IQ1S",
            "IQ1M",
            "IQ2_XXS",
            "IQ4_NL",
            "IQ4_XS",
            "TQ10",
            "TQ20",
            "MXFP4",
            "NVFP4",
        ),
    )
)


def catalog():
    cases = {}
    for index, (fmt, partner) in enumerate(zip(FORMATS, PARTNERS)):
        name = fmt + "_dequantize"
        cases[name] = Case(
            name,
            "dequantize",
            {"rows": 1024, "K": 4096},
            "row-dequantization",
            input_policy="finite-random-record-replicated",
            dtype=f"{fmt}->f32",
        )
        for suffix, rows in (("dot", 1), ("dot_rows", 4096)):
            name = f"{fmt}_{suffix}"
            cases[name] = Case(
                name,
                "vec_dot",
                {"rows": rows, "K": 4096},
                "vec-dot-leaf" if rows == 1 else "independent-dots-shared-activation",
                input_policy="independent-quantized-rows-seed410",
                dtype=f"{fmt}*{partner}",
            )
    for kind, fmt in enumerate(("q8_0", "q8_1", "q8_k")):
        name = fmt + "_quantize"
        cases[name] = Case(
            name,
            "quantize",
            {"rows": 1, "K": 4096},
            "activation-quantization",
            input_policy="fixed-float-mod31",
            dtype=f"f32->{fmt}",
        )
    cases["gemm_f32"] = Case(
        "gemm_f32",
        "dense",
        {"M": 128, "N": 4096, "K": 4096},
        "preallocated-gemm",
    )
    cases["affine"] = Case("affine", "dense", {"N": 4096}, "elementwise-scale-bias")
    cases["causal_attention"] = Case(
        "causal_attention", "attention", {"B": 1, "T": 4, "S": 16, "H": 4, "G": 2, "D": 32},
        "causal-grouped-query-attention", input_policy="finite-random-seed410",
    )
    for name in ("rms_norm", "layer_norm"):
        cases[name] = Case(name, "dense", {"M": 128, "N": 4096}, name + "-affine")
    cases["rope_split"] = Case(
        "rope_split",
        "dense",
        {"T": 128, "H": 32, "P": 64},
        "rope-precomputed-cos-sin",
    )
    cases["rope_full"] = replace(
        cases["rope_split"], name="rope_full", scope="rope-neox-includes-coefficients"
    )
    cases["softmax"] = Case("softmax", "dense", {"M": 4096, "N": 128}, "row-softmax")
    for name, affine in (("ime_centered", False), ("ime_affine", True)):
        cases[name] = Case(
            name,
            "ime",
            {"M": 4, "N": 16, "K": 4096},
            "packed-ime-leaf",
            input_policy="distinct-columns-mod61-activation-mod17",
            dtype="packed-i4*i8",
            data={"affine": affine},
        )
        cases[name + "_decode"] = replace(
            cases[name], name=name + "_decode", shape={"M": 1, "N": 16, "K": 4096},
            phase="decode", data={"affine": affine, "tile_rows": 1},
        )
    for rows, phase in ((1, "decode"), (4, "prefill")):
        name = "ime2_i8_" + phase
        cases[name] = Case(name, "ime2", {"M": rows, "N": 32, "K": 4096},
                           "packed-ime2-i8-block32", dtype="packed-i8*i8->f32",
                           phase=phase, data={"tile_rows": rows})
    for index, fmt in enumerate(("f16", *FORMATS)):
        for phase, m in (("decode", 1), ("prefill", 128)):
            name = fmt + "_projection_" + phase
            cases[name] = Case(
                name,
                "projection",
                {"M": m, "N": 4096, "K": 4096},
                "projection-includes-activation-quantization",
                phase=phase,
                dtype=fmt + "*f32",
                input_policy="valid-quantized-record-replicated",
                data={"format": fmt},
            )
    for phase, m in (("decode", 1), ("prefill", 128)):
        name = "f16_f32_projection_" + phase
        cases[name] = Case(
            name, "projection", {"M": m, "N": 4096, "K": 4096},
            "projection-preserves-f32-activation", phase=phase, dtype="f16*f32",
            input_policy="stored-f16-weights-original-f32-activation",
            data={"format": "f16", "activation_dtype": "f32"},
        )
    from .shared_semantics import catalog as shared_catalog
    from .pointer_semantics import catalog as pointer_catalog
    from .matmul import catalog as matmul_catalog

    cases.update(shared_catalog())
    cases.update(pointer_catalog())
    cases.update(matmul_catalog())
    cases["amx_q4_0_tile"] = Case(
        "amx_q4_0_tile", "amx_tile", {"M": 32, "N": 32, "K": 4096},
        "native-amx-q4_0-tile/prequantized/persistent-model-default",
        dtype="q4_0*q8_0", data={"format": "q4_0"},
    )
    cases["amx_q4_0_prefill_strided"] = Case(
        "amx_q4_0_prefill_strided", "amx_tile", {"M": 32, "N": 32, "K": 4096, "LDC": 4096},
        "native-amx-q4_0-tile/strided-output/prequantized/persistent-model-default",
        dtype="q4_0*q8_0", data={"format": "q4_0"},
    )
    cases["amx_q4_k_prefill_strided"] = Case(
        "amx_q4_k_prefill_strided", "amx_tile", {"M": 32, "N": 32, "K": 4096, "LDC": 4096},
        "native-amx-q4_k-tile/strided-output/prequantized/persistent-model-default",
        dtype="q4_k*q8_k", data={"format": "q4_k"},
    )
    cases["amx_q6_k_prefill_strided"] = Case(
        "amx_q6_k_prefill_strided", "amx_tile", {"M": 32, "N": 32, "K": 4096, "LDC": 4096},
        "native-amx-q6_k-tile/strided-output/prequantized/persistent-model-default",
        dtype="q6_k*q8_k", data={"format": "q6_k"},
    )
    cases["amx_q4_0_decode_tile"] = Case(
        "amx_q4_0_decode_tile", "amx_tile", {"M": 1, "N": 64, "K": 4096},
        "native-vnni-q4_0-tile/prequantized/persistent-model-default",
        dtype="q4_0*q8_0", data={"format": "q4_0", "decode": True},
    )
    cases["amx_q4_k_decode_tile"] = Case(
        "amx_q4_k_decode_tile", "amx_tile", {"M": 1, "N": 512, "K": 4096},
        "native-vnni-q4_k-tile/prequantized/persistent-model-default",
        dtype="q4_k*q8_k", data={"format": "q4_k", "decode": True},
    )
    cases["amx_q6_k_decode_tile"] = Case(
        "amx_q6_k_decode_tile", "amx_tile", {"M": 1, "N": 512, "K": 14336},
        "native-vnni-q6_k-tile/prequantized/persistent-model-default",
        dtype="q6_k*q8_k", data={"format": "q6_k", "decode": True},
    )
    return cases


SUITES = {
    "matmul": ("gemm_f32", "gemm_bf16_f32", "gemm_s8_s32"),
    "pointer-semantics": tuple("pointer_" + name for name in ("add", "tail", "strided", "sum", "softmax", "dot", "control", "alias", "mask_guard", "carry", "select", "bool")),
    "shared-semantics": tuple("shared_" + name for name in ("arange", "sum", "max", "softmax", "dot_acc", "batch_dot_acc", "for_while", "nested_level")),
    "quick": (
        "q4_0_dot",
        "q4_k_dot",
        "q8_k_quantize",
        "gemm_f32",
        "rms_norm",
        "layer_norm",
        "rope_split",
        "softmax",
    ),
    "llama-hot": (
        "q4_0_dot_rows",
        "q4_k_dot_rows",
        "q6_k_dot_rows",
        "q4_k_projection_decode",
        "q4_k_projection_prefill",
        "ime_centered",
        "ime_affine",
    ),
}


@dataclass(frozen=True)
class OperatorSelection:
    name: str
    shape: dict[str, int] = field(default_factory=dict)

    def describe(self):
        dimensions = ",".join(f"{key}={value}" for key, value in self.shape.items())
        return f"{self.name}({dimensions})" if dimensions else self.name


def configured_suites(settings=None):
    """Named workload selections are external data, independent of the runner."""
    configured = (settings or {}).get("operator_suites", {})
    if not isinstance(configured, dict):
        raise ValueError("operator_suites must map names to case lists")
    result = {}
    for name, cases in {**SUITES, **configured}.items():
        if not isinstance(name, str) or not name or not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError("each operator suite needs a name and a nonempty list of cases")
        selections = []
        for item in cases:
            if isinstance(item, str) and item:
                selected = OperatorSelection(item)
            elif isinstance(item, dict) and not item.keys() - {"case", "shape"}:
                case_name, shape = item.get("case"), item.get("shape", {})
                if not isinstance(case_name, str) or not case_name or not isinstance(shape, dict):
                    raise ValueError("operator suite entries require a case name and a shape mapping")
                selected = OperatorSelection(case_name, dict(shape))
            else:
                raise ValueError("operator suite entries must be case names or {case, shape} mappings")
            # Validate the entire suite before compilation or device work starts.
            get(selected.name, selected.shape)
            selections.append(selected)
        if len(selections) != len({case.name for case in selections}):
            raise ValueError("duplicate workload in operator suite " + name)
        result[name] = tuple(selections)
    return result


def get(name, shape=None):
    if ":" in name:
        case = load_object(name)
        if callable(case):
            case = case()
    else:
        try:
            case = catalog()[name]
        except KeyError:
            raise ValueError(f"unknown case {name!r}; use weft-bench list") from None
    if not isinstance(case, Case):
        raise TypeError("custom case must be a Case or a factory returning one")
    if shape:
        unknown = shape.keys() - case.shape.keys()
        if unknown or any(type(v) is not int or v <= 0 for v in shape.values()):
            raise ValueError(f"invalid shape override: {shape}")
        case = replace(case, shape={**case.shape, **shape})
    return case

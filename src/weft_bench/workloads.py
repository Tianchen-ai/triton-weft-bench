"""The shipped GGML workloads: semantics/IO, never compiler or machine selection."""

from dataclasses import dataclass, replace
from .core import NativeWorkload
from .adapters.dense import dense_defines


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    access: str = "input"
    layout: str = "contiguous"
    strides: tuple[int, ...] = ()


def ggml_type(fmt):
    return "GGML_TYPE_" + fmt.upper()


def parameter_defaults(case):
    if case.name == "pointer_select":
        return {"select_x": 1.0}
    if case.name in ("rms_norm", "layer_norm"):
        return {"epsilon": 1e-5}
    if case.name == "affine":
        return {"scale": 1.25, "bias": -0.5}
    if case.family == "attention":
        return {"scale": case.shape["D"] ** -0.5, "start": case.shape["S"] - case.shape["T"]}
    return {}


def ports(case):
    if case.family == "ime2":
        m, n, k = (case.shape[x] for x in ("M", "N", "K"))
        rows = case.data["tile_rows"]
        if m <= 0 or m % rows or n <= 0 or n % 32 or k <= 0 or k % 32:
            raise ValueError("IME2 packed rows require complete M tiles, N32 and K32 blocks")
        return [TensorSpec("W_codes", "i8", (n, k), layout="packed-before-timing"),
                TensorSpec("X_codes", "i8", (m, k), layout="packed-before-timing"),
                TensorSpec("Y", "f32", (m, n), "output")]
    if case.family == "matmul":
        from .operators.matmul import ports as matmul_ports

        return matmul_ports(case)
    if case.family == "pointer_semantics":
        from .operators.pointer_semantics import ports as pointer_ports

        return pointer_ports(case)
    if case.family == "shared_semantics":
        from .operators.shared_semantics import ports as shared_ports

        return shared_ports(case)
    s = case.shape
    family = case.family
    if family == "attention":
        if s["H"] % s["G"] or s["T"] > s["S"]:
            raise ValueError("attention requires H divisible by G and T <= S")
        return [TensorSpec("Q", "f32", (s["B"], s["T"], s["H"], s["D"])),
                TensorSpec("K", "f32", (s["B"], s["S"], s["G"], s["D"])),
                TensorSpec("V", "f32", (s["B"], s["S"], s["G"], s["D"])),
                TensorSpec("Y", "f32", (s["B"], s["T"], s["H"], s["D"]), "output")]
    if family == "dense":
        if case.name == "affine":
            return [TensorSpec("X", "f32", (s["N"],)),
                    TensorSpec("Y", "f32", (s["N"],), "output")]
        if case.name.startswith("rope"):
            result = [TensorSpec("X", "f32", (s["T"], s["H"], 2, s["P"]))]
            if case.name == "rope_split":
                result += [
                    TensorSpec(k, "f32", (s["T"], s["P"])) for k in ("Cos", "Sin")
                ]
            return result + [TensorSpec("Y", "f32", result[0].shape, "output")]
        result = [TensorSpec("X", "f32", (s["M"], s.get("K", s["N"])))]
        if case.name == "gemm_f32":
            result.append(TensorSpec("W", "f32", (s["N"], s["K"])))
        if case.name in ("rms_norm", "layer_norm"):
            result.append(TensorSpec("Weight", "f32", (s["N"],)))
        if case.name == "layer_norm":
            result.append(TensorSpec("Bias", "f32", (s["N"],)))
        return result + [TensorSpec("Y", "f32", (s["M"], s["N"]), "output")]
    if family == "vec_dot":
        w, x = case.dtype.split("*")
        return [
            TensorSpec("W", w, (s["rows"], s["K"]), layout="ggml-blocks"),
            TensorSpec("X", x, (s["K"],), layout="ggml-blocks"),
            TensorSpec("Y", "f32", (s["rows"],), "output"),
        ]
    if family in ("quantize", "dequantize"):
        before, after = case.dtype.split("->")
        return [
            TensorSpec(
                "X" if family == "quantize" else "W",
                before,
                (s["rows"], s["K"]),
                layout="contiguous" if before == "f32" else "ggml-blocks",
            ),
            TensorSpec(
                "Y",
                after,
                (s["rows"], s["K"]),
                "output",
                "contiguous" if after == "f32" else "ggml-blocks",
            ),
        ]
    if family == "amx_tile":
        decode = case.data.get("decode", False)
        weight = case.data["format"]
        block, activation = {"q4_0": (32, "q8_0"), "q4_k": (256, "q8_k"), "q6_k": (256, "q8_k")}[weight]
        if (decode and (s["M"] != 1 or s["N"] <= 0 or s["N"] % 64)) or (not decode and (s["N"] <= 0 or s["N"] % 32 or s["M"] < 1)) or s["K"] <= 0 or s["K"] % block:
            raise ValueError(f"AMX {weight} requires complete K{block} blocks; decode uses M1/N64 groups and prefill positive M/N32 groups")
        ldc = s.get("LDC", s["N"])
        if ldc < s["N"] or (decode and ldc != s["N"]):
            raise ValueError("AMX output row stride must cover N; decode uses its contiguous row ABI")
        return [TensorSpec("W", weight, (s["N"], s["K"]), layout="ggml-model-default-AMX"),
                TensorSpec("Xq", activation, (s["K"],) if decode else (s["M"], s["K"]), layout="ggml-blocks"),
                TensorSpec("Y", "f32", (s["N"],) if decode else (s["M"], s["N"]), "output",
                           "strided" if ldc != s["N"] else "contiguous", (ldc, 1) if ldc != s["N"] else ())]
    if family == "projection":
        return [
            TensorSpec(
                "W", case.data["format"], (s["N"], s["K"]), layout="ggml-blocks"
            ),
            TensorSpec("X", "f32", (s["M"], s["K"])),
            TensorSpec("Y", "f32", (s["M"], s["N"]), "output"),
        ]
    if family == "ime":
        return [
            TensorSpec("W_f32", "f32", (s["N"], s["K"]), layout="packed-before-timing"),
            TensorSpec("X_f32", "f32", (s["M"], s["K"]), layout="packed-before-timing"),
            TensorSpec("Y", "f32", (s["M"], s["N"]), "output"),
        ]
    raise ValueError(f"no GGML workload for {case.name}")


def configured_workload(case, settings):
    """The same declared numerical contract for native and language-runtime calls."""
    workload = native_workload(case)
    defines = []
    if case.family == "projection" and "-DWB_ACTIVATION_TYPE=GGML_TYPE_Q8_K" in workload.defines:
        contract = settings.get("projection_activation_contracts", {}).get("q8_k")
        if contract:
            if contract not in ("extrema-lo-rtne", "first-absolute-max-rtne"):
                raise ValueError("unsupported projection Q8_K storage contract")
            defines.append("-DWB_PROJECTION_Q8K_CONTRACT=1")
            if contract == "first-absolute-max-rtne":
                defines.append("-DWB_Q8K_FIRST_EXTREME=1")
    if case.family == "quantize":
        fmt = case.dtype.split("->")[1]
        contract = settings.get("quantization_contracts", {}).get(fmt)
        if contract:
            if fmt != "q8_k" or contract not in ("extrema-lo-rtne", "first-absolute-max-rtne"):
                raise ValueError("unsupported explicit quantization storage contract")
            if contract == "first-absolute-max-rtne":
                defines.append("-DWB_Q8K_FIRST_EXTREME=1")
    return replace(workload, defines=(*workload.defines, *defines))


def native_workload(case):
    if case.family == "ime2":
        ports(case)
        return NativeWorkload(
            "ime2_inputs.h", "IME2Inputs data(inputs);",
            {"W": "data.weights.data()", "Xq": "data.activation.data()+m*(data.K/32)*38",
             "Y": "output.data()+m*data.N"},
            defines=tuple(f"-DWB_{k}={v}" for k, v in case.shape.items()) +
                    (f"-DWB_TILE_ROWS={case.data['tile_rows']}",),
            before_call="for(size_t m=0;m<data.M;m+=WB_TILE_ROWS) {", after_call="}")
    if case.family == "amx_tile":
        ports(case)
        weight, activation, block = {"q4_0": ("q4_0", "q8_0", 32), "q4_k": ("q4_K", "q8_K", 256),
                                     "q6_k": ("q6_K", "q8_K", 256)}[case.data["format"]]
        return NativeWorkload(
            "amx_tile_inputs.h", "AMXTileInputs data(inputs);",
            {"W": "data.weights.data()", "Xq": "data.quantized.data()", "Y": "data.output(output)"},
            output_size="data.output_size + (data.ldc != data.N ? 16 : 0)",
            defines=tuple(f"-DWB_{k}={v}" for k, v in case.shape.items()) + (
                f"-DWB_ROWS={case.shape['N']}", "-DWB_QUANT_KIND=0", f"-DWB_WEIGHT_TYPE=GGML_TYPE_{weight.upper()}",
                f"-DWB_ACTIVATION_TYPE=GGML_TYPE_{activation.upper()}", "-DWB_GGML_MODEL_WEIGHTS=1",
                f"-DWB_AMX_WEIGHT_BLOCK=block_{weight}", f"-DWB_AMX_ACTIVATION_BLOCK=block_{activation}",
                f"-DWB_AMX_BLOCK_SIZE={block}",
                f"-DWB_AMX_DECODE={int(case.data.get('decode', False))}") +
                (() if "LDC" in case.shape else (f"-DWB_LDC={case.shape['N']}",)),
            snapshots={"W": "data.weights", "Xq": "data.quantized", "reference-Y": "data.expected"},
            correctness="within-tolerance-full-tile",
            strides={"Y": (case.shape["LDC"], 1)} if "LDC" in case.shape and case.shape["LDC"] != case.shape["N"] else {},
        )
    if case.family == "matmul":
        from .operators.matmul import workload

        return workload(case)
    if case.family == "pointer_semantics":
        from .operators.pointer_semantics import workload

        return workload(case)
    if case.family == "shared_semantics":
        from .operators.shared_semantics import workload

        return workload(case)
    s = case.shape
    family = case.family
    if family == "attention":
        return NativeWorkload(
            "attention_inputs.h", "AttentionInputs data(inputs);",
            {"Q": "data.query.data()", "K": "data.keys.data()",
             "V": "data.values.data()", "Y": "output.data()"},
            defines=tuple(f"-DWB_{key}={value}" for key, value in s.items()),
            data_type="AttentionInputs", scalars={"start": "data.start", "scale": "data.scale"},
        )
    if family == "dense":
        return NativeWorkload(
            "dense_inputs.h",
            "DenseInputs data(inputs);",
            {
                "X": "data.input.data()",
                "W": "data.weight.data()",
                "Weight": "data.weight.data()",
                "Bias": "data.bias.data()",
                "Cos": "data.cosine.data()",
                "Sin": "data.sine.data()",
                "Y": "output.data()",
            },
            defines=tuple(dense_defines(case)),
            data_type="DenseInputs",
            scalars={"epsilon": "data.epsilon", "scale": "data.scale", "bias": "data.scalar_bias"},
            correctness=(
                "sampled-64-within-tolerance"
                if case.name == "gemm_f32"
                else "within-tolerance"
            ),
        )
    tables = {
        k: "data." + k + ".data()"
        for k in ("grid", "signs", "codebook", "f16_bits", "scale")
    }
    tables.update(
        powers="data.powers",
        e8m0_scale="data.scale.data()",
        ue4m3_scale="data.scale.data()",
    )
    if family in ("vec_dot", "dequantize", "quantize"):
        kind = {"vec_dot": 0, "quantize": 1, "dequantize": 2}[family]
        if family == "vec_dot":
            fmt, partner = case.dtype.split("*")
        else:
            fmt = case.dtype.split("->")[1 if family == "quantize" else 0]
            partner = "q8_0"
        buffers = {
            **tables,
            "W": "data.weights.data()+row*data.weight_bytes",
            "Y": "output.data()+row",
        }
        if family == "vec_dot":
            buffers["X"] = "data.activation.data()"
        elif family == "quantize":
            buffers.update(
                X="data.source.data()+row*data.K",
                Y="output.data()+row*data.weight_bytes",
            )
        else:
            buffers["Y"] = "output.data()+row*data.K"
        defines = (
            f"-DWB_ROWS={s['rows']}",
            f"-DWB_K={s['K']}",
            f"-DWB_QUANT_KIND={kind}",
            "-DWB_WEIGHT_TYPE=" + ggml_type(fmt),
            "-DWB_ACTIVATION_TYPE=" + ggml_type(partner),
        )
        return NativeWorkload(
            "quantized_inputs.h",
            "QuantizedInputs data(inputs);",
            buffers,
            output_type="uint8_t" if family == "quantize" else "float",
            defines=defines,
            before_call="for(size_t row=0;row<data.rows;++row) {",
            after_call="}",
            batch_buffers=({**tables, "W": "data.weights.data()", "X": "data.activation.data()",
                            "Y": "output.data()"} if family in ("vec_dot", "dequantize") else {}),
            correctness=("bit-exact" if family == "quantize" else
                         "within-tolerance-packed-affine-dot" if family == "vec_dot" and fmt in ("q4_1", "q5_1") else
                         "within-tolerance"),
        )
    if family == "projection":
        from .operators import FORMATS, PARTNERS

        fmt = case.data["format"]
        partner = case.data.get("activation_dtype", "f16" if fmt == "f16" else PARTNERS[FORMATS.index(fmt)])
        preserve_f32 = partner == "f32"
        if preserve_f32 and fmt != "f16":
            raise ValueError("F32-preserving projection currently requires F16 weights")
        buffers = {k: v.replace("data.", "data.tables.") for k, v in tables.items()}
        buffers.update(
            W="data.weights.data()",
            X="data.activation.data()",
            Xq="workspace.data()",
            Xh="reinterpret_cast<_Float16*>(workspace.data())",
            Y="output.data()",
        )
        if fmt == "f16":
            buffers["W"] = "reinterpret_cast<const _Float16*>(data.weights.data())"
        if preserve_f32:
            buffers.pop("Xq")
            buffers.pop("Xh")
        defines = tuple(f"-DWB_{k}={v}" for k, v in s.items()) + (
            f"-DWB_ROWS={s['N']}",
            "-DWB_QUANT_KIND=0",
            "-DWB_WEIGHT_TYPE=" + ggml_type(fmt),
            "-DWB_ACTIVATION_TYPE=" + ggml_type(partner),
        )
        if preserve_f32:
            defines += ("-DWB_PROJECTION_KEEP_F32=1",)
        snapshots = {"X": "data.activation", "W": "data.weights"}
        if not preserve_f32:
            snapshots["reference-Xq"] = "data.quantized"
        snapshots.update({"reference-Y": "data.expected", "reference-Y-indices": "data.checked_indices",
                          "table-grid": "data.tables.grid", "table-signs": "data.tables.signs"})
        return NativeWorkload(
            "projection_inputs.h",
            "ProjectionInputs data(inputs);",
            buffers,
            defines=defines,
            snapshots=snapshots,
            correctness="within-tolerance-full-decode" if s["M"]==1 else "within-tolerance-sampled-64-prefill-and-finite-outputs",
        )
    if family == "ime":
        return NativeWorkload(
            "ime_inputs.h",
            "IMEInputs data(inputs);",
            {
                "W": "data.weights.data()",
                "Xq": "data.activation.data()+m*(data.K/32)*36",
                "Y": "output.data()+m*data.N",
            },
            defines=tuple(f"-DWB_{k}={v}" for k, v in s.items())
            + (f"-DWB_AFFINE={int(case.data['affine'])}",
               f"-DWB_TILE_ROWS={case.data.get('tile_rows', 4)}"),
            before_call="for(size_t m=0;m<data.M;m+=WB_TILE_ROWS) {",
            after_call="}",
        )
    raise ValueError(f"no native workload for {case.name}")

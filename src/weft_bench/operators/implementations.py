"""Default Weft bindings; measured target choices live in external files."""

from ..types import Implementation


def default_binding(case):
    name, family = case.name, case.family
    if family == "vec_dot":
        fmt, partner = case.dtype.split("*")
        return Implementation(
            f"examples/kernels/vec_dot/{fmt}_{partner}.py",
            f"quantized_vec_dot_{fmt}_{partner}",
        )
    if family == "dequantize":
        fmt = case.dtype.split("->")[0]
        return Implementation(
            f"examples/kernels/dequantize/{fmt}.py", f"row_dequantize_{fmt}"
        )
    if family == "quantize":
        fmt = case.dtype.split("->")[1]
        return Implementation(
            f"examples/kernels/quantize/{fmt}.py",
            "q8_K_quantize" if fmt == "q8_k" else name,
        )
    if name == "gemm_f32":
        return Implementation(
            "examples/kernels/dense/gemm_f32.py",
            name,
            {"meta": {"MC": 64, "NC": 16, "MR": 4, "NR": 4}},
        )
    if name in ("rms_norm", "layer_norm"):
        return Implementation(
            "source/weft/programs/normalization.py",
            name,
            {"meta": {"BLOCK": 32}, "lmul_eighths": 32},
        )
    if name == "softmax":
        return Implementation(
            "source/weft/programs/attention.py",
            name,
            {
                "meta": {"N": case.shape["N"]},
                "lmul_eighths": 16,
                "private_stack_bytes": 262144,
            },
        )
    if name in ("rope_split", "rope_full"):
        return Implementation(
            "source/weft/programs/attention.py", "rope_split", {"meta": {"BLOCK": 32}}
        )
    if family == "ime":
        rows = case.data.get("tile_rows", 4)
        symbol = "q4_" + ("affine" if case.data["affine"] else "centered")
        symbol += "_packed_decode" if rows == 1 else "_packed_prefill"
        return Implementation(
            "source/weft/ggml/mul_mat/ime1/kernels.py",
            symbol,
            {
                "meta": {"MR": rows, "NR": 16},
                "matrix_extension": "spacemit-ime1",
                "require_extensions": ["spacemit-ime1"],
            },
        )
    return Implementation()


def resolve_binding(case, target, options):
    from dataclasses import asdict

    result = asdict(default_binding(case))
    for layer in (
        target.settings.get("compile_options", {}),
        options.get("compile_options", {}),
    ):
        result["options"].update(layer)
    for binding in (
        target.settings.get("bindings", {}).get(case.name, {}),
        options.get("bindings", {}).get(case.name, {}),
    ):
        result["options"].update(binding.get("options", {}))
        result.update({k: v for k, v in binding.items() if k != "options"})
    for key in ("source", "symbol", "arguments"):
        if key in options:
            result[key] = options[key]
    return result

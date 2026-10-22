"""Map observed functions to reusable workloads, never infer an engine from a phase."""

import re
import shlex
import json
from pathlib import Path


def link_hotspots(
    rows, *, target, phase, call_sites=None, operator_rows=(), model_info=None
):
    sites = call_sites or {}
    linked = []
    for hot in rows[:3]:
        row = dict(hot)
        symbol = hot["symbol"]
        case = None
        engine = None
        quant = re.search(
            r"(?:quantized_vec_dot|ggml_vec_dot)_(q\d+_[0kK])_q8_[0kK1]", symbol
        )
        if quant:
            case = quant[1].lower() + "_dot_rows"
            engine = "rvv"
        elif "gemm_kernel_i8i4" in symbol or "packed_prefill" in symbol or "packed_decode" in symbol:
            case = "ime_affine" if "affine" in symbol else "ime_centered"
            if "packed_decode" in symbol:
                case += "_decode"
            engine = "ime"
            if "gemm_kernel_i8i4" in symbol:
                case = sites.get(symbol, {}).get(
                    "case"
                )  # native symbol serves both packed formats
        elif "rms_norm" in symbol:
            case = "rms_norm"
            engine = "native"
        elif "soft_max" in symbol or "softmax" in symbol:
            case = "softmax"
            engine = "native"
        elif "rope" in symbol:
            case = "rope_full"
            engine = "native"
        elif "quantize_row_q8_K" in symbol or "q8_K_quantize" in symbol:
            case = "q8_k_quantize"
            engine = "rvv"
        site = sites.get(symbol, sites.get(case, {})) if case else sites.get(symbol, {})
        case = site.get("case", case)
        row.update(
            phase=phase,
            engine=site.get("engine", engine),
            case=case,
            function=site.get("function", symbol),
        )
        if site.get("origin"):
            row["mapping_evidence"] = site["origin"]
        if case:
            shape = site.get("shape")
            model_shapes = []
            if model_info:
                kind = {"q4_0": 2, "q4_k": 12, "q6_k": 14}.get(
                    case.removesuffix("_dot_rows")
                )
                if kind is not None:
                    model_shapes = [
                        {"rows": item["shape"][1], "K": item["shape"][0]}
                        for item in model_info.get("tensor_shapes", [])
                        if item["ggml_type"] == kind
                    ]
            if not shape and model_shapes:
                shape = model_shapes[0]
            command = [
                "weft-bench",
                "op",
                case,
                "--target",
                target,
                "--impl",
                "weft,ggml",
            ]
            if shape:
                command += ["--shape", ",".join(f"{k}={v}" for k, v in shape.items())]
            origin = (
                site.get("origin", "configured")
                if site.get("shape")
                else "model-tensor-metadata" if model_shapes else "not_observed"
            )
            mapping = (
                "configured_call"
                if site.get("shape")
                else "model_tensor_case" if model_shapes else "representative_case"
            )
            row.update(
                shape=shape,
                shape_origin=origin,
                model_shapes=model_shapes,
                replay=shlex.join(command),
                mapping=mapping,
            )
            if model_shapes and not site.get("shape"):
                row["mapping_note"] = (
                    "Tensor dimensions suggest workloads; sampling did not observe native tile sizes or call frequency."
                )
            row["operator_results"] = [
                {
                    "implementation": r.implementation,
                    "ms": r.median_ms,
                    "status": r.status,
                }
                for r in operator_rows
                if r.case == case
                and r.target == target
                and (not shape or r.shape == shape)
            ]
        else:
            row.update(
                mapping="unmapped",
                mapping_note="No established operator mapping for this function",
            )
        linked.append(row)
    return linked


def attach_saved_profiles(operator_rows, target):
    """Optional saved model evidence is reused during check, without model execution."""
    settings = target.settings.get("llama", {})
    for model, phases in settings.get("saved_profiles", {}).items():
        info_path = settings.get("model_info", {}).get(model)
        info = json.loads(Path(info_path).read_text()) if info_path else None
        for phase, path in phases.items():
            data = json.loads(Path(path).read_text())
            symbols = (
                data.get("symbols", data.get("hotspots", []))
                if isinstance(data, dict)
                else data
            )
            linked = link_hotspots(
                symbols,
                target=target.name,
                phase=phase,
                call_sites=settings.get("call_sites", {}).get(model, {}).get(phase, {}),
                model_info=info,
                operator_rows=operator_rows,
            )
            for row in operator_rows:
                for hot in linked:
                    related = hot.get("case") == row.case
                    leaf = (
                        row.case.endswith("_dot")
                        and hot.get("case") == row.case + "_rows"
                    )
                    if related or leaf:
                        row.details.setdefault("model_paths", []).append(
                            {
                                "model": model,
                                "profile": str(path),
                                **hot,
                                "scope_relation": (
                                    "same-workload"
                                    if related
                                    else "leaf-of-multi-row-workload"
                                ),
                            }
                        )

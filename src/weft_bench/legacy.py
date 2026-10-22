"""Explicit import of old observations, never silently relabelled as new timing."""

import csv
import json
from pathlib import Path
import re

from .types import Result


def import_history(source, *, measurement=None):
    """Keep both sides and every unmapped observation; never select a best result."""
    from dataclasses import replace

    path = Path(source)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    else:
        records = json.loads(path.read_text())
    if isinstance(records, dict) and "results" in records:
        return [Result(**r) for r in records["results"]]
    rows = []
    for index, old in enumerate(records):
        try:
            config = json.loads(old.get("configuration") or "{}")
        except ValueError:
            config = {}
        if measurement is not None and config.get("measurement") != measurement:
            continue
        # Reuse the established semantic import, preserving original fields.
        reason = "No established workload/timing mapping; original record retained"
        try:
            reference = _operator_record(old, config, path)
        except (ValueError, KeyError, TypeError) as error:
            reference = None
            reason = f"Incomplete historical scenario: {error}"
        if reference is None:
            row = Result(
                old.get("kernel", f"row-{index}"),
                "historical",
                old.get("hardware", "unknown"),
                {},
                old.get("timing_scope", "unknown"),
                "unmapped",
                details={
                    "original": old,
                    "imported_from": str(path),
                    "mapping_note": reason,
                },
            )
            rows.append(row)
            continue
        rows.append(reference)
        value = old.get("median_ms")
        if value:
            details = {
                **reference.details,
                "build": {
                    "source": config.get("entry"),
                    "options": config.get("options", {}),
                },
                "native_build": {
                    **config.get("llvm_toolchain", {}),
                    "flags": old.get("compiler_flags"),
                    "compiler_version": old.get("compiler"),
                },
            }
            raw = config.get("raw_record")
            if raw:
                artifact = Path(raw).parent / "artifact"
                details["artifacts"] = {"weft": str(artifact)}
            rows.append(
                replace(
                    reference,
                    implementation="weft",
                    median_ms=float(value),
                    details=details,
                )
            )
    if not rows:
        raise ValueError("no records match the selected historical measurement")
    return rows


def _operator_record(old, config, path):
    fmt = re.match(r"quantized_vec_dot_(.+)_q8_[0k1]$", old.get("kernel", ""))
    shape = {
        k: int(v) for k, v in re.findall(r"([A-Za-z]+)=(\d+)", old.get("shape", ""))
    }
    march = re.search(r"-march=([^;\s]+)", old.get("compiler_flags", ""))
    vlen = re.search(r"VLEN(\d+)", old.get("isa", ""))
    if not (
        old.get("family") == "vec_dot"
        and fmt
        and march
        and vlen
        and old.get("comparison_median_ms")
        and config.get("input_policy") == "independent-quantized-rows-seed410"
    ):
        return None
    from .operators import get

    case = get(fmt[1] + "_dot_rows", {"rows": shape["N"], "K": shape["K"]})
    target = {"SG2044": "sg2044", "K1/X60": "k1"}.get(old["hardware"], old["hardware"])
    return Result(
        case.name,
        "ggml",
        target,
        case.shape,
        case.scope,
        "ok",
        float(old["comparison_median_ms"]),
        correctness=old.get("correctness", "not_checked"),
        details={
            "imported_from": str(path),
            "original": old,
            "measurement": {
                "cache": "cold",
                "eviction_bytes": 67108864,
                "warmup": 1,
                "samples": 10,
                "batch_ms": 10,
            },
            "input_policy": case.input_policy,
            "dtype": case.dtype,
            "phase": "",
            "execution": {
                "cpus": config["native_cpu"],
                "threads": 1,
                "env": {},
                "isa": {"march": march[1], "abi": "lp64d", "vlen_bits": int(vlen[1])},
            },
        },
    )


def import_baseline(source, *, kind="operators", target=None, profile=False):
    path = Path(source)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            data = list(csv.DictReader(stream))
    else:
        data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return [Result(**row) for row in data["results"]]
    rows = []
    for old in data:
        if kind == "llama":
            if not target:
                raise ValueError("historical llama JSON needs --target")
            phase = "prefill" if old["n_prompt"] else "decode"
            row = Result(
                Path(old["model_filename"]).stem + "/" + phase,
                "normal",
                target,
                {
                    "prompt": old["n_prompt"],
                    "generate": old["n_gen"],
                    "depth": old.get("n_depth", 0),
                },
                "llama-" + phase,
                "ok",
                float(old["avg_ns"]) / 1e6,
                [float(v) / 1e6 for v in old.get("samples_ns", [])],
            )
            row.details = {
                "measurement": {"profile": profile, "protocol": "historical-llama"},
                "native": old,
            }
        else:
            timing = old.get("comparison_median_ms")
            if timing is None or timing == "":
                continue
            shape = {
                key: int(value)
                for key, value in re.findall(r"([A-Za-z]+)=(\d+)", old.get("shape", ""))
            }
            row = Result(
                old.get("kernel", "unknown"),
                old.get("comparison_kind", "ggml"),
                target or old.get("hardware", "unknown"),
                shape,
                old.get("timing_scope", "historical"),
                "ok",
                float(timing),
                correctness=old.get("correctness", "not_checked"),
            )
            row.details = {
                "measurement": {"cache": "cold", "protocol": "historical-driver"},
                "native": old,
            }
            # This established driver has a known equivalent in the new runner.
            # Other historical scopes remain archival until an explicit adapter exists.
            if old.get("family") == "vec_dot":
                config = json.loads(old.get("configuration") or "{}")
                fmt = re.match(r"quantized_vec_dot_(.+)_q8_[0k1]$", row.case)
                march = re.search(r"-march=([^;\s]+)", old.get("compiler_flags", ""))
                vlen = re.search(r"VLEN(\d+)", old.get("isa", ""))
                if (
                    fmt
                    and march
                    and vlen
                    and config.get("input_policy")
                    == "independent-quantized-rows-seed410"
                ):
                    row.case = fmt[1] + "_dot_rows"
                    row.target = target or {"SG2044": "sg2044", "K1/X60": "k1"}.get(
                        row.target, row.target
                    )
                    row.shape = {"rows": shape["N"], "K": shape["K"]}
                    row.scope = "independent-dots-shared-activation"
                    from .operators import catalog

                    case = catalog()[row.case]
                    row.details.update(
                        measurement={
                            "cache": "cold",
                            "eviction_bytes": 67108864,
                            "samples": int(old.get("repetitions", 10)),
                            "warmup": 1,
                            "batch_ms": 10,
                        },
                        input_policy=config["input_policy"],
                        dtype=case.dtype,
                        phase="",
                        execution={
                            "cpus": config["native_cpu"],
                            "threads": 1,
                            "env": {},
                            "isa": {
                                "march": march[1],
                                "abi": "lp64d",
                                "vlen_bits": int(vlen[1]),
                            },
                        },
                    )
        row.details.update(
            imported_from=str(path.resolve()),
            comparison_note="Historical scope retained; select the same protocol to reuse",
        )
        rows.append(row)
    if not rows:
        raise ValueError("no baseline timings found in this file")
    return rows

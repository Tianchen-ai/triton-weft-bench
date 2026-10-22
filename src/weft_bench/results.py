from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .types import Result
from .models.environment import backend_policy


PROJECTION_SCOPE = "projection-includes-activation-quantization"
MIXED_PROJECTION_SCOPE = "projection-preserves-f32-activation"
MODEL_WEIGHTS_SCOPE = PROJECTION_SCOPE + "/persistent-model-default-weights"


def execution_contract(execution):
    """Resolve an existing local launcher alias, preserving recorded evidence."""
    numa = execution.get("numa", {})
    command = numa.get("command")
    if isinstance(command, str) and Path(command).is_absolute() and Path(command).is_file():
        return {**execution, "numa": {**numa, "command": str(Path(command).resolve())}}
    return execution


def input_identity(policy):
    """Resolve a live local model-file alias without changing recorded paths."""
    if isinstance(policy, str) and Path(policy).is_absolute() and Path(policy).is_file():
        return str(Path(policy).resolve())
    return policy


def projection_scope(scope):
    for base in (PROJECTION_SCOPE, MIXED_PROJECTION_SCOPE):
        if scope in (base, base + "/persistent-model-default-weights"):
            return base
    return None


def contract(row):
    """Readable comparison key; compiler build may change between candidates."""
    d = row.details
    measurement = d.get("measurement", {})
    timing = {
        k: measurement[k]
        for k in ("cache", "eviction_bytes", "profile", "protocol")
        if k in measurement
    }
    if timing.get("cache") == "reuse":
        timing.pop("eviction_bytes", None)
    return (
        row.case,
        row.target,
        json.dumps(row.shape, sort_keys=True),
        row.scope,
        json.dumps(timing, sort_keys=True),
        input_identity(d.get("input_policy")),
        d.get("dtype"),
        d.get("phase"),
        json.dumps(execution_contract(d.get("execution", {})), sort_keys=True),
        json.dumps(d.get("model_build_policy"), sort_keys=True),
        json.dumps(backend_policy(d.get("runtime_environment", {})), sort_keys=True),
    )


def execution_issue(row):
    """Reject explicit contradictions without inferring external parallelism."""
    execution = row.details.get("execution", {})
    entry = row.details.get("entry", execution.get("entry"))
    threads = execution.get("threads")
    if entry in ("single", "single-instance") and threads != 1:
        return f"single-instance entry cannot represent execution.threads={threads!r}"
    if entry == "grid":
        launch_threads = row.details.get("launch", {}).get("threads")
        if launch_threads is not None and launch_threads != threads:
            return f"grid launch threads={launch_threads!r} differ from execution.threads={threads!r}"
    return None


def frequency_policy_differences(row, reference):
    """Compare observed policy, not instantaneous frequency or system load."""
    def policies(result):
        return {item["cpu"]: item["frequency_policy"]
                for item in result.details.get("observed_environment", {}).get("affinity", [])
                if item.get("frequency_policy")}

    current, previous = policies(row), policies(reference)
    fields = ("driver", "governor", "energy_preference", "minimum_khz", "maximum_khz")
    differences = {}
    # Historical observations without these fields remain unknown; never infer
    # their policy from today's device configuration.
    for cpu in current.keys() & previous.keys():
        changed = {key: {"current": current[cpu][key], "reference": previous[cpu][key]}
                   for key in fields if key in current[cpu] and key in previous[cpu]
                   and current[cpu][key] != previous[cpu][key]}
        if changed:
            differences[str(cpu)] = changed
    return differences


def _projection_evidence(row):
    """Recognize recorded complete-operator boundaries, never a suffix alone."""
    base_scope = projection_scope(row.scope)
    if base_scope is None:
        return None
    d = row.details
    if d.get("measurement", {}).get("protocol") != "native-workload-v3":
        return None
    if any(type(row.shape.get(k)) is not int or row.shape[k] < 1 for k in ("M", "N", "K")):
        return None
    m, n, k = (row.shape[key] for key in ("M", "N", "K"))
    ports = {p["name"]: p for p in d.get("ports", [])}
    dtype = d.get("dtype")
    if not isinstance(dtype, str) or not dtype.endswith("*f32"):
        return None
    weight_type = dtype.removesuffix("*f32")
    for name, dtype, shape, access in (
        ("W", weight_type, [n, k], "input"),
        ("X", "f32", [m, k], "input"),
        ("Y", "f32", [m, n], "output"),
    ):
        port = ports.get(name, {})
        if port.get("dtype") != dtype or list(port.get("shape", [])) != shape or port.get("access") != access:
            return None
    boundary = d.get("projection_boundary", {})
    if boundary:
        # An explicit incompatible boundary must not fall back to legacy evidence.
        required = {
            "schema": 1,
            "scope": base_scope,
            "weights": "persistent-before-timing",
            "activation": "preserved-f32-in-call" if base_scope == MIXED_PROJECTION_SCOPE else "preparation-in-call",
            "output": "complete-projection-in-call",
        }
        evidence = boundary.get("evidence", {})
        if base_scope == MIXED_PROJECTION_SCOPE and evidence.get("provider") == "ggml":
            dispatch = d.get("reference_dispatch", {})
            if (dispatch.get("activation_precision") != "f32-preserved"
                    or not dispatch.get("resolved_compute_symbol")
                    or d.get("reference_precision", {}).get("contract") != "stored-f16-weights-original-f32-activation"):
                return None
        if (all(boundary.get(key) == value for key, value in required.items())
                and isinstance(evidence.get("provider"), str) and evidence["provider"]
                and all(evidence.get(key) for key in (
                    "worker_source", "input_setup", "weight_input", "activation_input",
                    "preparation", "invocation",
                ))):
            return {"kind": "native-worker", "scope": base_scope, "worker_source": evidence["worker_source"]}
        return None
    # Existing fixed references already observed the selected model weight buffer.
    # Read that evidence in place; do not rewrite old results or infer it from a name.
    dispatch = d.get("reference_dispatch", {})
    if (row.scope == MODEL_WEIGHTS_SCOPE
            and d.get("reference_boundary") == "persistent model weights prepared outside timing; activation quantization and graph compute inside"
            and dispatch.get("entry") == "ggml_backend_graph_plan_compute"
            and dispatch.get("weights_policy") == "model-default"
            and dispatch.get("weight_preparation") == "persistent tensor_set/repack before timing"
            and dispatch.get("weight_buffer_usage") == "WEIGHTS"
            and dispatch.get("weight_buffer_type")
            and dispatch.get("projection_rows") == m
            and dispatch.get("weight_type", "").lower() == weight_type.lower()):
        return {"kind": "observed-ggml-model-weights", "scope": base_scope, "weight_buffer_type": dispatch["weight_buffer_type"]}
    return None


def comparison_basis(row, reference):
    """Explain the sole supported scope equivalence; all other fields stay exact."""
    if execution_issue(row) or execution_issue(reference) or frequency_policy_differences(row, reference):
        return None
    current, previous = contract(row), contract(reference)
    if current[3] == previous[3] or current[:3] + current[4:] != previous[:3] + previous[4:]:
        return None
    current_evidence, reference_evidence = _projection_evidence(row), _projection_evidence(reference)
    if (current_evidence is None or reference_evidence is None
            or current_evidence["scope"] != reference_evidence["scope"]):
        return None
    return {
        "rule": "complete-projection-with-persistent-weights",
        "current_scope": row.scope,
        "reference_scope": reference.scope,
        "current_evidence": current_evidence,
        "reference_evidence": reference_evidence,
    }


def contracts_match(row, reference):
    # Preserve exact historical comparisons, including cases without new metadata.
    if execution_issue(row) or execution_issue(reference) or frequency_policy_differences(row, reference):
        return False
    return contract(row) == contract(reference) or comparison_basis(row, reference) is not None


def apply_baselines(rows, baselines, name=None):
    for row in rows:
        row.speedup = None
        row.baseline = None
        row.details.pop("comparison_note", None)
        row.details.pop("comparison_differences", None)
        row.details.pop("comparison_basis", None)
        row.details.pop("baseline_median_ms", None)
        row.details.pop("baseline_scope", None)
        row.details.pop("baseline_measurement", None)
        issue = execution_issue(row)
        if issue:
            row.details["comparison_note"] = issue
            continue
        matches = [
            b
            for b in baselines
            if contracts_match(row, b)
            and b.status == "ok"
            and b.median_ms is not None
            and (name is None or b.implementation == name)
        ]
        if len(matches) == 1 and row.status == "ok" and row.median_ms:
            row.speedup = matches[0].median_ms / row.median_ms
            row.baseline = matches[0].implementation
            row.details["baseline_median_ms"] = matches[0].median_ms
            row.details["baseline_scope"] = matches[0].scope
            row.details["baseline_measurement"] = dict(matches[0].details.get("measurement", {}))
            basis = comparison_basis(row, matches[0])
            if basis:
                row.details["comparison_basis"] = basis
        elif len(matches) > 1:
            row.details["comparison_note"] = (
                "multiple matching baselines; select one implementation"
            )
        else:
            row.details["comparison_note"] = (
                "no baseline with matching execution/timing scope"
            )
            fields = ("case", "target", "shape", "scope", "measurement", "input_policy", "dtype", "phase", "execution", "model_build_policy", "backend_environment")
            row.details["comparison_differences"] = [
                {"implementation": b.implementation, "differences": {
                    **{key: {"current": current, "reference": reference}
                       for key, current, reference in zip(fields, contract(row), contract(b))
                       if current != reference},
                    **({"execution_entry": {"reference_error": execution_issue(b)}} if execution_issue(b) else {}),
                    **({"frequency_policy": frequency_policy_differences(row, b)}
                       if frequency_policy_differences(row, b) else {}),
                }}
                for b in baselines if b.case == row.case and b.target == row.target
                and (name is None or b.implementation == name)
            ]
    return rows


def save(rows, destination):
    path = Path(destination)
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "results.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "results": [r.to_dict() for r in rows]}, indent=2
        )
        + "\n"
    )
    fields = [
        "case",
        "shape",
        "implementation",
        "target",
        "scope",
        "median_ms",
        "calling_thread_median_ms",
        "cpu_time_scope",
        "observed_frequency_before_khz",
        "observed_frequency_after_khz",
        "baseline",
        "baseline_median_ms",
        "speedup",
        "warmup",
        "samples",
        "samples_collected",
        "batch_ms",
        "measurement_protocol",
        "cache",
        "baseline_scope",
        "baseline_warmup",
        "baseline_samples",
        "baseline_batch_ms",
        "comparison_note",
        "status",
        "correctness",
        "tokens_per_second",
        "model_policy",
        "replacement",
        "replacement_counter_scope",
        "vec_dot_calls",
        "quantize_calls",
        "ime_calls",
        "ime2_calls",
        "amx_calls",
        "candidate_origin",
        "jit_status",
        "jit_cold_compile_and_call_ms",
        "jit_first_cache_hit_ms",
        "jit_hot_call_ms",
        "jit_disk_cache_hit",
        "jit_launches_per_workload",
        "error",
    ]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            data = row.to_dict()
            data["shape"] = ";".join(f"{k}={v}" for k, v in row.shape.items())
            data["tokens_per_second"] = row.details.get("tokens_per_second", "")
            data["model_policy"] = row.details.get("model_policy")
            data["replacement"] = row.details.get("replacement")
            data["candidate_origin"] = row.details.get("candidate_origin")
            jit = row.details.get("jit", row.details if row.details.get("entry") == "python-jit-grid" else {})
            runtime = jit.get("runtime", {})
            data["jit_status"] = runtime.get("status")
            data["jit_cold_compile_and_call_ms"] = runtime.get("cold_compile_and_launch_ms")
            data["jit_first_cache_hit_ms"] = runtime.get("first_cache_hit_launch_ms")
            data["jit_hot_call_ms"] = jit.get("python_jit", {}).get("median_ms")
            events = jit.get("disk_cache", {}).get("compilations", [])
            data["jit_disk_cache_hit"] = all(event["cache_hit"] for event in events) if events else None
            data["jit_launches_per_workload"] = runtime.get("launches_per_workload")
            counts = row.details.get("replacement_counts", {})
            data["replacement_counter_scope"] = counts.get("scope")
            for kind in ("vec_dot", "quantize", "ime", "ime2", "amx"):
                observed = counts.get(kind + "_observed" if kind in ("ime2", "amx") else "observed")
                data[kind + "_calls"] = counts.get(kind) if observed else None
            native = row.details.get("native", {})
            cpu_samples = native.get("calling_thread_samples_ms", [])
            data["calling_thread_median_ms"] = statistics.median(cpu_samples) if cpu_samples else None
            data["cpu_time_scope"] = native.get("cpu_time_scope")
            frequency = native.get("observed_frequency_khz", {})
            data["observed_frequency_before_khz"] = frequency.get("before")
            data["observed_frequency_after_khz"] = frequency.get("after")
            measurement = row.details.get("measurement", {})
            data.update(warmup=measurement.get("warmup"), samples=measurement.get("samples"),
                        samples_collected=len(row.samples_ms), measurement_protocol=measurement.get("protocol"),
                        batch_ms=measurement.get("batch_ms"), cache=measurement.get("cache"))
            # Only comparison paths that selected a unique successful reference
            # record this value. Never reconstruct a reference time from a ratio.
            compared = row.status == "ok" and row.baseline is not None and row.speedup is not None
            data["baseline_median_ms"] = row.details.get("baseline_median_ms") if compared else None
            data["baseline_scope"] = row.details.get("baseline_scope") if compared else None
            reference_measurement = row.details.get("baseline_measurement", {}) if compared else {}
            for key in ("warmup", "samples", "batch_ms"):
                data["baseline_" + key] = reference_measurement.get(key)
            data["speedup"] = row.speedup if compared else None
            notes = [row.details["comparison_note"]] if row.details.get("comparison_note") else []
            basis = row.details.get("comparison_basis", {})
            if basis.get("rule"):
                notes.append(basis["rule"])
            for item in row.details.get("comparison_differences", []):
                if item.get("differences"):
                    notes.append(item["implementation"] + ": " + ", ".join(item["differences"]))
            if compared and data["baseline_median_ms"] is None:
                notes.append("reference time absent in saved result; compare with the reference run to populate it")
            data["comparison_note"] = "; ".join(notes)
            writer.writerow({key: data.get(key) for key in fields})
    return path


def read(path, *, _visited=()):
    path = Path(path)
    if path.is_dir():
        path = path / ("index.json" if (path / "index.json").is_file() else "results.json")
    identity = path.resolve()
    if identity in _visited:
        raise ValueError("cyclic result index: " + str(path))
    data = json.loads(path.read_text())
    if "sources" in data:
        rows = []
        for source in data["sources"]:
            location = path.parent / source["path"]
            selected = read(location, _visited=(*_visited, identity))
            rows.extend(row for row in selected
                        if ("cases" not in source or row.case in source["cases"])
                        and ("implementation" not in source or row.implementation == source["implementation"])
                        and ("target" not in source or row.target == source["target"]))
        return rows
    rows = [Result(**row) for row in data["results"]]
    # A copied run can replay its own captured canonical IR on a different host.
    for row in rows:
        local = path.parent / "artifacts" / row.case / row.implementation
        if (local / "replay.json").exists():
            row.details["replay"] = str((local / "replay.json").resolve())
            row.details["diagnostics"] = str(local.resolve())
            for key, relative in (
                ("canonical", "canonical.mlir"),
                ("weft", "artifact"),
            ):
                if (local / relative).exists():
                    row.details.setdefault("artifacts", {})[key] = str(
                        (local / relative).resolve()
                    )
    return rows


def table(rows):
    header = ["case", "shape", "implementation", "target", "ms", "speedup", "status"]
    models = any("tokens_per_second" in r.details for r in rows)
    if models:
        header += ["tokens/s", "replacement"]
    cells = [header]
    for r in rows:
        line = [
            r.case,
            ",".join(f"{k}={v}" for k, v in r.shape.items()),
            r.implementation,
            r.target,
            f"{r.median_ms:.6f}" if r.median_ms is not None else "—",
            f"{r.speedup:.3f}x" if r.speedup is not None else "—",
            r.status,
        ]
        if models:
            tokens = r.details.get("tokens_per_second")
            line += [
                f"{tokens:.3f}" if tokens else "—",
                r.details.get("replacement", "—"),
            ]
        cells.append(line)
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in cells
    )

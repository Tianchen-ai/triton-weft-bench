"""Verification selections are data; targets and implementations remain external."""
from copy import deepcopy
import fnmatch
import json
from pathlib import Path
import re

from ..operators import get
from ..loading import load_object


def compilations(settings, target):
    """Consume a producer's current manifest, with explicit deduplicated bindings."""
    configuration = settings.get("compilations")
    if not configuration:
        return {}
    rows = json.loads(Path(configuration["results"]).read_text())
    records = {}
    for row in rows:
        key = (row["target"], row["case"])
        if key in records:
            raise ValueError("duplicate compilation record " + repr(key))
        records[key] = row
    bindings = (json.loads(Path(configuration["bindings"]).read_text()) if configuration.get("bindings") else
                [{"target": row["target"], "case": row["case"], "compiled_case": row["case"]} for row in rows])
    scenarios = {}
    for binding in bindings:
        if binding["target"] != target.name:
            continue
        key = (target.name, binding["compiled_case"])
        record = deepcopy(records.get(key, {"status": "unavailable", "error": "producer has no compilation record for " + repr(key)}))
        if configuration.get("root"):
            directory = Path(configuration["root"]) / target.name / binding["compiled_case"]
            record.setdefault("diagnostics", str(directory))
            record.setdefault("request", str(directory / "triton-compile.json"))
            record.setdefault("source_snapshot", str(directory / "author-source.py"))
        if record["status"] == "compiled":
            scenarios[binding["case"]] = {"artifact": record["artifact"], "compilation": record}
        else:
            scenarios[binding["case"]] = {"compilation_failure": record}
    return scenarios


def plan(target, *, groups=None, cases=None, use_prepared=True):
    settings = target.settings.get("verification", {})
    implementation = settings.get("source_implementation", "triton_source")
    configured = target.implementations.get(implementation)
    if configured is None:
        raise ValueError(f"verification source implementation {implementation!r} is not configured")
    entries = []
    prepared = compilations(settings, target) if use_prepared else {}
    for name, binding in configured.get("bindings", {}).items():
        case = get(name)
        scenario = {**prepared.get(name, {}), **settings.get("operators", {}).get(name, {})}
        options = {**configured, **binding}
        options.pop("bindings", None)
        if not options.get("source"):
            raise ValueError(f"verification requires source for {name}, not a cached AOT")
        options.pop("artifact", None)
        options.pop("path", None)
        entry = {"id": "operators/" + name, "group": "operators", "kind": "native",
                 "case": name, "shape": {**case.shape, **target.settings.get("case_catalog", {}).get(name, {}).get("shape", {}),
                                         **scenario.get("shape", {})},
                 "inputs": scenario.get("inputs", {}), "options": options,
                 "source": options["source"], "symbol": options["symbol"]}
        if use_prepared and scenario.get("artifact"):
            entry["artifact"] = options["artifact"] = scenario["artifact"]
            options.pop("source", None)
            entry["compilation"] = scenario.get("compilation", {})
        elif use_prepared and scenario.get("compilation_failure"):
            entry.update(kind="compilation_failed", compilation=scenario["compilation_failure"])
        entries.append(entry)
    for configured_entry in settings.get("entries", []):
        entry = deepcopy(configured_entry)
        if entry["kind"] in ("native", "compile_rejection") or (entry["kind"] == "program" and entry.get("source")):
            base = target.implementations.get(entry.get("implementation", implementation), {})
            binding = base.get("bindings", {}).get(entry["case"], {})
            options = {**base, **binding, **entry.get("options", {})}
            options.pop("bindings", None)
            if entry.get("source"):
                options.update(source=entry["source"], symbol=entry["symbol"])
                options.pop("artifact", None)
                options.pop("path", None)
            entry["options"] = options
            if entry["kind"] != "program":
                entry.setdefault("shape", get(entry["case"]).shape)
        entries.append(entry)
    if use_prepared and settings.get("prepared_entries"):
        paths = settings["prepared_entries"]
        paths = [paths] if isinstance(paths, str) else paths
        by_id = {}
        for path in paths:
            for row in json.loads(Path(path).read_text()):
                if row["target"] != target.name:
                    continue
                if row["case"] in by_id:
                    raise ValueError("duplicate prepared entry " + row["case"])
                by_id[row["case"]] = row
        for entry in entries:
            record = by_id.get(entry["id"])
            if record is None:
                continue
            entry["compilation"] = record
            if entry["kind"] == "compile_rejection":
                continue
            if record["status"] == "compiled":
                entry["artifact"] = record["artifact"]
                entry["source_origin"] = entry.get("source")
                entry["source"] = record.get("source_snapshot", entry.get("source"))
                if entry["kind"] == "native":
                    entry["options"].pop("source", None)
                    entry["options"]["artifact"] = record["artifact"]
            else:
                entry["kind"] = "compilation_failed"
    names = set()
    for entry in entries:
        identity = entry["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", identity) or any(
                piece in (".", "..") for piece in identity.split("/")):
            raise ValueError("invalid verification id " + identity)
        if identity in names:
            raise ValueError("duplicate verification id " + identity)
        names.add(identity)
        entry["target"] = target.name
        factory = entry.get("options", {}).get("factory")
        if factory and getattr(load_object(factory), "execution_mode", None) == "jit":
            entry["execution_mode"] = "jit"
            if entry.get("artifact") or entry["kind"] == "compilation_failed":
                raise ValueError(identity + ": JIT verification requires source execution on the target; "
                                 "prepared compilation records belong to a separate native verification configuration")
    selected = [entry for entry in entries if
                (not groups or entry["group"] in groups) and
                (not cases or any(fnmatch.fnmatchcase(entry["id"], pattern) for pattern in cases))]
    missing_cases = [pattern for pattern in (cases or [])
                     if not any(fnmatch.fnmatchcase(entry["id"], pattern) for entry in selected)]
    missing_groups = [group for group in (groups or [])
                      if not any(entry["group"] == group for entry in selected)]
    if missing_cases or missing_groups:
        raise ValueError("verification selection contains unmatched selectors: " +
                         ", ".join(["case=" + value for value in missing_cases] +
                                   ["group=" + value for value in missing_groups]))
    if (groups or cases) and not selected:
        raise ValueError("verification selection matched no entries")
    return selected

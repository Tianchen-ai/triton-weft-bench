"""Compile a selected verification plan using the existing source adapter."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import re
import traceback

from ..adapters.triton_native import compile_source
from ..diagnostics import failure_stage
from .catalog import plan
from .provenance import snapshot
from .runner import _prepare_current, _write


def prepare(target, *, output, groups=None, cases=None, workers=1, resume=False, rerun=None):
    if workers < 1:
        raise ValueError("preparation workers must be positive")
    entries = plan(target, groups=groups, cases=cases, use_prepared=False)
    output = Path(output).resolve()
    manifest = output / "preparation.json"
    results = {row["case"]: row for row in json.loads(manifest.read_text())} if manifest.exists() else {}
    if any(row["target"] != target.name for row in results.values()):
        raise ValueError("preparation output belongs to a different target")
    pending = []
    for entry in entries:
        previous = results.get(entry["id"])
        if previous and not resume and rerun is None:
            raise ValueError("preparation already has results; use --resume or --rerun")
        if rerun == "failed" and (previous is None or previous["status"] == "compiled"):
            continue
        if previous and resume and rerun is None:
            recorded = Path(previous["provenance"])
            if not recorded.exists() or json.loads(recorded.read_text()) != snapshot(entry, target):
                raise ValueError(entry["id"] + ": preparation identity changed; explicitly select --rerun all")
            continue
        pending.append(entry)
    _write(output / "plan.json", {"target": target.name, "entries": entries})

    def compile_entry(entry):
        directory = _prepare_current(output / "cases" / entry["id"])
        before = snapshot(entry, target)
        _write(directory / "provenance.json", before)
        options = entry.get("options", {})
        row = {"target": target.name, "case": entry["id"], "source": options.get("source"),
               "symbol": options.get("symbol"), "options": options,
               "diagnostics": str(directory), "source_snapshot": str(directory / "author-source.py"),
               "provenance": str(directory / "provenance.json")}
        try:
            if not options.get("source"):
                raise ValueError("source preparation requires a configured source entry")
            artifact = compile_source(None, target, options, directory)
            row.update(status="compiled", artifact=str(artifact))
            if entry["kind"] == "compile_rejection":
                row.update(status="invalid_source_accepted", error="compiler accepted the configured invalid source")
            if snapshot(entry, target) != before:
                row.update(status="stale", error="compiler/source changed during compilation; rebuild against a stable environment")
        except Exception as error:
            row.update(status="compile_failed", error=str(error), compiler_stage=failure_stage(str(error)),
                       traceback=traceback.format_exc())
            if entry["kind"] == "compile_rejection" and entry.get("diagnostic_pattern") and re.search(entry["diagnostic_pattern"], str(error)):
                row.update(status="rejected", expectation="compile-rejection")
        _write(directory / "verification.json", row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compile_entry, entry) for entry in pending]
        for future in as_completed(futures):
            row = future.result()
            results[row["case"]] = row
            _write(manifest, sorted(results.values(), key=lambda r: r["case"]))
            print(row["case"] + ": " + row["status"], flush=True)
    return [results[entry["id"]] for entry in entries if entry["id"] in results]

"""A ledger around existing adapters: one device run at a time, no baseline timing."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

from ..api import benchmark
from ..inputs import resolve_inputs
from ..operators import get
from ..results import save
from ..types import Measurement
from .catalog import plan
from .provenance import snapshot, observe_runtime


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def native(entry, target, directory):
    case = replace(get(entry["case"]), shape=entry["shape"])
    implementation = entry.get("implementation", "candidate")
    configured = replace(target, implementations={implementation: entry["options"]})
    rows = benchmark(case, [implementation], configured, Measurement(check_only=True),
                     inputs=resolve_inputs(entry.get("inputs")), save_outputs=True,
                     output=directory / "artifacts")
    save(rows, directory)
    row = rows[0]
    detail = row.details
    return {"status": "passed" if row.status == "ok" else row.status,
            "result": str((directory / "results.json").resolve()), "error": row.error,
            "stage": detail.get("failure_stage"), "compiler_stage": detail.get("compiler_stage"),
            "correctness": row.correctness, "checked": row.status == "ok",
            "oracle": detail.get("oracle"),
            "source": entry.get("source", entry["options"].get("source")),
            "symbol": entry.get("symbol", entry["options"].get("symbol")),
            "diagnostics": detail.get("diagnostics"), "replay": detail.get("replay"),
            "outputs": detail.get("outputs"), "native": detail.get("native"),
            "jit": detail.get("jit"), "candidate_origin": detail.get("candidate_origin"),
            "compilation": entry.get("compilation"), "artifacts": detail.get("artifacts")}


def _retain_failure(root, current):
    """Keep the first diagnostic and most recent diagnostic, never every attempt."""
    first = root / "first-failure"
    destination = first if not first.exists() else root / "latest-failure"
    if destination.exists():
        shutil.rmtree(destination)
    current.rename(destination)


def _prepare_current(root):
    root.mkdir(parents=True, exist_ok=True)
    # Upgrade the initial development ledger to bounded, stable storage.
    for old in sorted(root.glob("attempt-*")):
        result = old / "verification.json"
        if result.is_file() and json.loads(result.read_text()).get("status") not in ("passed", "compiled", "rejected"):
            _retain_failure(root, old)
        elif old.is_dir():
            shutil.rmtree(old)
    current = root / "current"
    if current.exists():
        result = current / "verification.json"
        if result.is_file() and json.loads(result.read_text()).get("status") not in ("passed", "compiled", "rejected"):
            _retain_failure(root, current)
        else:
            shutil.rmtree(current)
    current.mkdir()
    return current


def verify(target, *, output, groups=None, cases=None, resume=False, rerun=None):
    output = Path(output).expanduser().resolve()
    entries = plan(target, groups=groups, cases=cases)
    identity = observe_runtime(target, output)
    target = replace(target, settings={**target.settings, "_verification_runtime_identity": identity})
    ledger_file = output / "verification.json"
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {
        "schema": 1, "target": target.name, "entries": {}}
    if ledger["target"] != target.name:
        raise ValueError("verification output belongs to a different target")
    ledger.pop("stopped_after_current", None)
    ledger["selection"] = [entry["id"] for entry in entries]
    _write(output / "plan.json", {"target": target.name, "entries": entries})
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        existing = ledger["entries"].get(entry["id"])
        if existing and not resume and rerun is None:
            raise ValueError("output already has results; use --resume or --rerun")
        if existing and resume and rerun is None and existing["status"] not in ("pending", "running"):
            if existing.get("provenance") != snapshot(entry, target):
                raise ValueError(entry["id"] + ": plan, source or compiler identity changed; select --case and --rerun all")
        ledger["entries"].setdefault(entry["id"], {**entry, "status": "pending", "attempts": []})

    def flush():
        ledger["counts"] = dict(Counter(row["status"] for row in ledger["entries"].values()))
        ledger["selection_counts"] = dict(Counter(ledger["entries"][key]["status"] for key in ledger["selection"]))
        ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write(ledger_file, ledger)
        failures = [row for row in ledger["entries"].values()
                    if row["status"] not in ("passed", "pending", "running", "not_applicable")]
        _write(output / "failures.json", {"target": target.name, "entries": failures})

    flush()
    executed = []
    for entry in entries:
        stop_file = output / ".stop-after-current"
        if stop_file.exists():
            stop_file.unlink()
            ledger["stopped_after_current"] = True
            flush()
            break
        record = ledger["entries"][entry["id"]]
        if resume and record["status"] == "passed" and rerun != "all":
            continue
        if rerun == "failed" and record["status"] in ("passed", "pending", "running"):
            continue
        if resume and record["status"] not in ("pending", "running") and rerun is None:
            continue
        index = len(record["attempts"]) + 1
        directory = _prepare_current(output / "cases" / entry["id"])
        provenance = snapshot(entry, target)
        attempts = record["attempts"]
        record.clear()
        record.update({**entry, "status": "running", "provenance": provenance, "attempts": attempts})
        flush()
        print("Verifying " + entry["id"] + " on " + target.name, flush=True)
        try:
            if entry["kind"] == "native":
                result = native(entry, target, directory)
            else:
                from .workers import run
                result = run(entry, target, directory)
        except Exception as error:
            result = {"status": "unsupported" if isinstance(error, NotImplementedError) else "verification_failed",
                      "error": str(error), "traceback": traceback.format_exc(), "checked": False}
        if snapshot(entry, target) != provenance:
            result["environment_changed_during_run"] = True
            if result["status"] == "passed":
                result.update(status="stale", checked=False,
                              error="source or compiler changed during execution; rerun against a stable environment")
        result.update(attempt=index, directory=str(directory))
        _write(directory / "verification.json", result)
        record.update(result)
        executed.append(entry["id"])
        record["attempts"] = [{"attempt": item["attempt"], "status": item["status"]}
                              for item in record["attempts"]]
        record["attempts"].append({"attempt": index, "status": result["status"]})
        flush()
        print(entry["id"] + ": " + result["status"] +
              (" — " + str(result.get("error", "")).splitlines()[0][:180] if result.get("error") else ""), flush=True)
    if executed and observe_runtime(target, output) != identity:
        for key in executed:
            record = ledger["entries"][key]
            if record["status"] == "passed":
                record.update(status="stale", checked=False,
                              error="installed runtime changed during verification; rerun against a stable installation")
                record["attempts"][-1]["status"] = "stale"
                path = Path(record["directory"]) / "verification.json"
                detail = json.loads(path.read_text())
                detail.update(status=record["status"], checked=False, error=record["error"])
                _write(path, detail)
        flush()
    return ledger

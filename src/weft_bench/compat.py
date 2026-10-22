from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import tempfile

from .execution import ExecutionError, Executor
from .loading import load_object
from .types import Measurement, Result


def adapter_for(name, target):
    defaults = {
        "weft": "weft_bench.adapters.native:WeftAdapter",
        "ggml": "weft_bench.adapters.native:GGMLAdapter",
        "artifact": "weft_bench.adapters.native:ArtifactAdapter",
    }
    options = dict(target.implementations.get(name, {}))
    factory = options.pop("factory", defaults.get(name))
    if not factory:
        raise ValueError(
            f"implementation {name!r} needs a module:factory in external configuration"
        )
    return load_object(factory)(options)


def benchmark(case, implementations, target, measurement=None, *, output=None):
    measurement = measurement or Measurement()
    root = Path(output) if output else Path(tempfile.mkdtemp(prefix="weft-bench-"))
    rows = []
    for name in implementations:
        directory = root / case.name / name
        executor = Executor(target, directory)
        row = Result(case.name, name, target.name, case.shape, case.scope, "pending")
        row.details = {
            "measurement": asdict(measurement),
            "input_policy": case.input_policy,
            "dtype": case.dtype,
            "phase": case.phase,
            "execution": {
                "cpus": executor.cpus,
                "threads": target.settings.get("threads", 1),
                "isa": target.settings.get("target", {}),
                "env": executor.env,
            },
        }
        from .diagnostics import save_request, failure_stage

        row.details["replay"] = save_request(case, name, target, measurement, directory)
        stage = "prepare"
        try:
            prepared = adapter_for(name, target).prepare(case, target, executor)
            row.details.update(prepared.contract)
            row.details["artifacts"] = prepared.artifacts
            stage = "run"
            completed = executor.command(
                prepared.argv,
                cwd=prepared.cwd,
                env={**prepared.env, **measurement.env()},
                affinity=True,
                check=False,
            )
            (directory / "stdout.log").write_text(completed.stdout)
            (directory / "stderr.log").write_text(completed.stderr)
            if completed.returncode:
                row.status = (
                    "check_failed"
                    if any(
                        x in completed.stderr.lower()
                        for x in ("mismatch", "numeric", "tolerance")
                    )
                    else "run_failed"
                )
                row.error = (completed.stderr or completed.stdout)[-6000:]
            else:
                records = [
                    json.loads(line.removeprefix("WB_RESULT "))
                    for line in completed.stdout.splitlines()
                    if line.startswith("WB_RESULT ")
                ]
                if len(records) != 1:
                    raise ValueError(
                        "native driver must emit exactly one WB_RESULT JSON record"
                    )
                record = records[0]
                samples = record["samples_ms"]
                if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError(
                        "native timing must contain finite positive milliseconds"
                    )
                row.samples_ms = samples
                row.median_ms = statistics.median(samples)
                row.correctness = record.get("correctness", "not_checked")
                row.status = "check_failed" if row.correctness == "failed" else "ok"
                row.details["native"] = record
        except NotImplementedError as error:
            row.status, row.error = "unsupported", str(error)
        except ExecutionError as error:
            row.status, row.error = error.stage + "_failed", str(error)
            row.details["failure_stage"] = error.stage
        except Exception as error:
            row.status, row.error = (
                stage + "_failed",
                f"{type(error).__name__}: {error}",
            )
            row.details["failure_stage"] = stage
        row.details["compiler_stage"] = failure_stage(row.error)
        if (directory / "compile.json").exists():
            row.details["build"] = json.loads((directory / "compile.json").read_text())
            if (directory / "effective-options.json").exists():
                row.details["build"]["effective_options"] = json.loads(
                    (directory / "effective-options.json").read_text()
                )
        row.details["native_build"] = target.settings.get("toolchain", {})
        if name == "ggml":
            row.details["reference_build"] = target.settings.get("ggml", {})
        row.details["diagnostics"] = str(directory.resolve())
        if (directory / "canonical.mlir").exists():
            row.details.setdefault("artifacts", {})["canonical"] = str(
                (directory / "canonical.mlir").resolve()
            )
        (directory / "commands.json").write_text(
            json.dumps(executor.commands, indent=2)
        )
        rows.append(row)
    return rows

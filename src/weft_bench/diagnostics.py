"""Saved cases are directly reusable by the mainline compiler developer."""

from dataclasses import asdict, replace
import difflib
import json
from pathlib import Path
import re

from .types import Case, Measurement, Target


def save_request(case, name, target, measurement, directory):
    # Targets contain only user-supplied execution configuration, never vault credentials.
    payload = {
        "case": asdict(case),
        "implementation": name,
        "target": asdict(target),
        "measurement": asdict(measurement),
    }
    path = Path(directory) / "replay.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path.resolve())


def failure_stage(error):
    match = re.search(
        r"(?:compiler-error|invalid-input|unsupported-target) at ([^\n]+)", error or ""
    )
    return match[1] if match else None


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(a.keys() | b.keys()):
            rows += differences(
                a.get(key), b.get(key), prefix + ("." if prefix else "") + key
            )
        return rows
    return [] if a == b else [{"field": prefix, "before": a, "after": b}]


def load_request(row):
    path = row.details.get("replay")
    if not path:
        raise ValueError(
            "result predates replay support; use its documented original case"
        )
    return Path(path), json.loads(Path(path).read_text())


def explain(row):
    result = {
        "case": row.case,
        "status": row.status,
        "compiler_stage": row.details.get("compiler_stage"),
        "failure_stage": row.details.get("failure_stage"),
        "comparison": row.details.get("comparison", {}),
        "comparison_note": row.details.get("comparison_note"),
        "comparison_differences": row.details.get("comparison_differences", []),
        "error": row.error,
        "replay": row.details.get("replay"),
        "artifacts": row.details.get("artifacts", {}),
        "model_paths": row.details.get("model_paths", []),
        "scenario": {
            "shape": row.shape,
            "scope": row.scope,
            **{
                k: row.details.get(k)
                for k in ("measurement", "execution", "input_policy")
            },
        },
    }
    relevant = lambda build: {
        k: v for k, v in build.items() if k not in ("output", "canonical")
    }
    current = relevant(row.details.get("build", {}))
    old = row.details.get("comparison", {}).get("previous", {})
    if old:
        result["changes"] = differences(
            relevant(old.get("build", {})), current, "build"
        )
        result["measurement_changes"] = differences(
            old.get("measurement", {}), row.details.get("measurement", {})
        )
        result["native_build_changes"] = differences(
            old.get("native_build", {}), row.details.get("native_build", {})
        )
        for filename in ("canonical.mlir", "kernel.c", "kernel.s"):
            old_artifact = old.get("artifact")
            new_artifact = row.details.get("artifacts", {}).get("weft")
            if not old_artifact or not new_artifact:
                continue
            before = Path(old_artifact) / filename
            after = Path(new_artifact) / filename
            if filename == "canonical.mlir":
                before, after = (
                    before.parent.parent / filename,
                    after.parent.parent / filename,
                )
            if before.is_file() and after.is_file():
                a, b = before.read_text(), after.read_text()
                result[filename] = {
                    "identical": a == b,
                    "diff": "".join(
                        difflib.unified_diff(
                            a.splitlines(True),
                            b.splitlines(True),
                            fromfile=str(before),
                            tofile=str(after),
                        )
                    ),
                }
    return result


def replay(row, *, compiler=None, source=None, output):
    from .api import benchmark

    path, request = load_request(row)
    inputs = request["case"].get("data", {}).get("inputs", {})
    for role, info in inputs.get("buffers", {}).items():
        for folder in ("worker-" + request["implementation"], "worker"):
            local = path.parent.parent / folder / "inputs" / (role + ".bin")
            if local.is_file():
                info["path"] = str(local.resolve())
                break
    target = Target(**request["target"])
    name = request["implementation"]
    options = dict(target.implementations.get(name, {}))
    from .runner import mode

    if mode(name, target) == "triton_jit":
        if compiler:
            raise ValueError("native Triton JIT replay uses its configured target Python/toolchain; --compiler selects the legacy Weft CLI")
        options = {**options, **options.get("bindings", {}).get(request["case"]["name"], {})}
        options.pop("bindings", None)
        saved_source = path.with_name("author-source.py")
        options["source"] = source or (str(saved_source) if saved_source.is_file() else options["source"])
        target = replace(target, implementations={**target.implementations, name: options})
        return benchmark(Case.from_dict(request["case"]), [name], target,
                         Measurement(**request["measurement"]), output=output)
    if mode(name, target) == "triton":
        from .adapters.triton_native import replay_options

        options = replay_options(options, request["case"]["name"], path.parent, source=source, compiler=compiler)
        target = replace(target, implementations={**target.implementations, name: options})
        return benchmark(Case.from_dict(request["case"]), [name], target,
                         Measurement(**request["measurement"]), output=output)
    compile_file = path.with_name("compile.json")
    if compile_file.exists():
        build = json.loads(compile_file.read_text())
        options.update(
            root=build["root"],
            compiler=compiler or build["compiler"],
            bindings={
                request["case"]["name"]: {
                    "source": build["source"],
                    "symbol": build["symbol"],
                    "options": build["options"],
                    "arguments": build.get("arguments", {}),
                }
            },
        )
        effective = path.with_name("effective-options.json")
        if effective.exists():
            options["bindings"][request["case"]["name"]]["options"] = json.loads(
                effective.read_text()
            )
            options.pop("selection", None)
        canonical = path.with_name("canonical.mlir")
        if canonical.exists() and source is None:
            options["canonical"] = str(canonical)
        if source:
            options["source"] = source
            options.pop("canonical", None)
    elif compiler:
        options["compiler"] = compiler
    saved_artifact = path.parent / "artifact"
    if not compile_file.exists() and (saved_artifact / "manifest.json").is_file():
        options["path"] = str(saved_artifact.resolve())
        options["artifacts"] = {
            **options.get("artifacts", {}),
            request["case"]["name"]: str(saved_artifact.resolve()),
        }
        if compiler or source:
            manifest = json.loads((saved_artifact / "manifest.json").read_text())
            options["factory"] = "weft_bench.adapters.native:WeftAdapter"
            options["bindings"] = {
                request["case"]["name"]: {
                    "source": source or "",
                    "symbol": manifest["kernel"]["symbol"],
                    "options": manifest["options"],
                }
            }
            if source:
                options["source"] = source
            else:
                options["canonical"] = str(
                    (saved_artifact / "canonical.mlir").resolve()
                )
    target = replace(target, implementations={**target.implementations, name: options})
    return benchmark(
        Case.from_dict(request["case"]),
        [name],
        target,
        Measurement(**request["measurement"]),
        output=output,
    )

"""Transport for correctness workers; product APIs own loading and execution."""
import json
from pathlib import Path
import tarfile
import shutil
import re

from ..execution import Executor


def _upload_artifact(executor, source):
    source = Path(source).resolve()
    metadata = json.loads((source / "manifest.json").read_text())
    if metadata.get("producer") != "triton" or not metadata.get("native_library"):
        raise ValueError("program verification requires an exported Triton native artifact")
    archive = executor.local / "artifact.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=True) as package:
        package.add(source, arcname="artifact")
    uploaded = executor.upload(archive, "artifact.tar.gz")
    executor.command(["tar", "xzf", uploaded, "--no-same-owner", "--no-same-permissions"],
                     cwd=executor.workspace, stage="artifact_deploy")
    return str(Path(executor.workspace) / "artifact")


def run(entry, target, directory):
    if entry["kind"] == "compile_rejection":
        from ..adapters.triton_native import compile_source
        from ..diagnostics import failure_stage
        from ..execution import ExecutionError
        from ..operators import get

        expected = entry.get("diagnostic_pattern")
        if not expected:
            raise ValueError("expected compiler rejection requires a specific diagnostic_pattern")
        recorded = entry.get("compilation")
        if recorded:
            accepted = recorded["status"] in ("compiled", "invalid_source_accepted")
            diagnostic = recorded.get("error", "")
            matched = not accepted and bool(re.search(expected, diagnostic))
            return {"status": "passed" if matched else "invalid_source_accepted" if accepted else "unexpected_rejection",
                    "checked": False, "rejected": not accepted, "expected_diagnostic_observed": matched,
                    "diagnostic": diagnostic, "error": None if matched else diagnostic,
                    "compiler_stage": recorded.get("compiler_stage"), "diagnostics": recorded.get("diagnostics"),
                    "expectation": "compile-rejection"}
        try:
            compile_source(get(entry["case"]), target, entry["options"], directory)
        except ExecutionError as error:
            diagnostic = str(error)
            matched = bool(re.search(expected, diagnostic))
            return {"status": "passed" if matched else "unexpected_rejection", "checked": False,
                    "rejected": True, "expected_diagnostic_observed": matched, "diagnostic": diagnostic,
                    "error": None if matched else diagnostic, "compiler_stage": failure_stage(diagnostic),
                    "diagnostics": str(directory), "expectation": "compile-rejection"}
        return {"status": "invalid_source_accepted", "checked": False, "rejected": False,
                "error": "compiler accepted the configured invalid source", "diagnostics": str(directory)}
    if entry["kind"] == "usage":
        return run_usage(entry, target, directory)
    if entry["kind"] == "compilation_failed":
        failure = entry["compilation"]
        return {"status": "unsupported" if failure.get("status") == "unavailable" else "compile_failed", "checked": False, "stage": "compile",
                "error": failure["error"], "diagnostics": failure.get("diagnostics"),
                "compilation": failure}
    if entry["kind"] in ("framework", "runtime"):
        return run_probe(entry, target, directory)
    if entry["kind"] == "not_applicable":
        if not entry.get("reason"):
            raise ValueError("inapplicable entries require a concrete capability reason")
        return {"status": "not_applicable", "checked": False, "reason": entry["reason"]}
    if entry["kind"] != "program":
        raise NotImplementedError("unconfigured verification worker " + entry["kind"])
    configuration = target.settings.get("verification", {}).get("runtime", {})
    if not configuration.get("python"):
        raise ValueError("verification.runtime.python must select the target's installed product")
    artifact = entry.get("artifact")
    if not artifact:
        from ..adapters.triton_native import compile_source
        from ..execution import ExecutionError
        from ..diagnostics import failure_stage
        try:
            artifact = str(compile_source(None, target, entry["options"], directory))
        except ExecutionError as error:
            return {"status": "compile_failed", "checked": False, "error": str(error),
                    "stage": "compile", "compiler_stage": failure_stage(str(error)), "diagnostics": str(directory)}
    executor = Executor(target, directory / "worker")
    here = Path(__file__).parent
    files = [(here / "program_process.py", "program_process.py")]
    files.extend((here.parent / "operators" / name, name) for name in
                 ("program_oracles.py", "dense_oracles.py", "encoded_oracles.py", "example_oracles.py"))
    for namespace, root in entry.get("imports", {}).items():
        if not namespace.isidentifier():
            raise ValueError("import root must be a Python package name")
        root = Path(root)
        files.extend((path, str(Path(namespace) / path.relative_to(root))) for path in root.rglob("*.py"))
    deployed = executor.upload_many(files)
    command = deployed["program_process.py"]
    request = {"case": entry["case"], "artifact": _upload_artifact(executor, artifact),
               "bindings": entry.get("bindings", {}), "seed": entry.get("seed", 410),
               "output": str(Path(executor.workspace) / "output"), "grid": entry.get("grid", [1])}
    if entry.get("source"):
        request["encoding_source"] = executor.upload(entry["source"], "encoding_source.py")
    request_path = executor.write("request.json", json.dumps(request, indent=2))
    environment = {**configuration.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = executor.command([configuration["python"], command, request_path], affinity=True,
                                 cwd=executor.workspace, env=environment, check=False)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2) + "\n")
    (directory / "worker.log").write_text(completed.stdout + completed.stderr)
    messages = [json.loads(line.removeprefix("WB_RESULT ")) for line in completed.stdout.splitlines()
                if line.startswith("WB_RESULT ")]
    if not messages:
        return {"status": "runtime_failed", "checked": False, "stage": "worker_process",
                "exit_code": completed.returncode, "error": completed.stderr or completed.stdout,
                "diagnostics": str(directory)}
    result = messages[-1]
    if completed.returncode and result["status"] == "passed":
        result.update(status="runtime_failed", checked=False, error="worker exited after its result", exit_code=completed.returncode)
    executor.download(Path(request["output"]) / "result.json", directory / "result.json")
    if result.get("inputs"):
        executor.download(Path(request["output"]) / "inputs.npz", directory / "inputs.npz")
    for call in range(1, result.get("calls", 1) + 1):
        if result.get("checks"):
            executor.download(Path(request["output"]) / f"outputs-{call}.npz", directory / f"outputs-{call}.npz")
    return {**result, "artifact": artifact, "diagnostics": str(directory),
            "result": str(directory / "result.json")}


def _download_output(executor, source, directory):
    archive = str(Path(executor.workspace) / "outputs.tar.gz")
    executor.command(["tar", "czf", archive, "-C", source, "."], stage="collect")
    local = executor.download(archive, directory / "outputs.tar.gz")
    try:
        with tarfile.open(local) as package:
            for member in package:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                    raise ValueError("invalid worker output archive entry")
                destination = directory / "output" / path
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with package.extractfile(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    finally:
        local.unlink()


def run_probe(entry, target, directory):
    kind = entry["kind"]
    configuration = target.settings.get("verification", {}).get("runtime", {})
    if not configuration.get("python"):
        raise ValueError("verification.runtime.python is required")
    configuration = {**configuration, **entry.get("runtime", {})}
    executor = Executor(target, directory / "worker")
    here = Path(__file__).parent
    probes = here / "probes"
    worker_source = "cpu_distribution.py" if kind == "runtime" and entry["case"] == "distribution" else kind + "_process.py"
    files = [(probes / worker_source, "probe.py"),
             (here.parent / "adapters/compile_triton.py", "compile_triton.py"),
             (here.parent / "adapters/cpu_target.py", "cpu_target.py")]
    if kind == "framework":
        files.extend((path, str(Path("kernels") / path.relative_to(probes / "framework_kernels")))
                     for path in (probes / "framework_kernels").rglob("*.py"))
    else:
        files.append((probes / "runtime_kernels.py", "kernels.py"))
        files.append((probes / "numerical_kernels.py", "numerical_kernels.py"))
        files.append((probes / "numerical_primitive_kernels.py", "numerical_primitive_kernels.py"))
        files.append((probes / "numerical_primitives.py", "numerical_primitives.py"))
        files.append((probes / "packed_layout_kernels.py", "packed_layout_kernels.py"))
        files.append((probes / "packed_layouts.py", "packed_layouts.py"))
        files.append((probes / "control_kernels.py", "control_kernels.py"))
        files.append((probes / "helper_native.py", "helper_native.py"))
        files.append((probes / "memory_kernels.py", "memory_kernels.py"))
        files.append((probes / "riscv_kernels.py", "riscv_kernels.py"))
        files.append((probes / "level_kernels.py", "level_kernels.py"))
        files.append((probes / "level_scenarios.py", "level_scenarios.py"))
        files.append((probes / "bundled_toolchain.py", "bundled_toolchain.py"))
        files.append((probes / "dense_view_kernels.py", "dense_view_kernels.py"))
        files.append((probes / "dense_view_probe.py", "dense_view_probe.py"))
        files.append((probes / "source_auto_probe.py", "source_auto_probe.py"))
        files.append((probes / "source_auto_kernels.py", "source_auto_kernels.py"))
        if entry["case"] == "dynamic_rounding":
            files.extend((probes / name, name) for name in
                         ("dynamic_rounding.py", "dynamic_rounding_kernels.py"))
            files.append((here.parent / "native/verification/fenv_bridge.c", "fenv_bridge.c"))
    from .source_inputs import source_files
    extra = source_files(configuration)
    reserved = {name.split("/", 1)[0] for _, name in files} | {"request.json", "toolchain.json", "output"}
    if any(name.split("/", 1)[0] in reserved for _, name in extra):
        raise ValueError("runtime source destination conflicts with a worker file")
    files.extend(extra)
    executor.upload_many(files)
    request = {"target": target.name, **entry.get("request", {})}
    request.setdefault("toolchain", configuration.get("toolchain", {}))
    executor.write("request.json", json.dumps(request, indent=2))
    executor.write("toolchain.json", json.dumps(request["toolchain"], indent=2))
    output = str(Path(executor.workspace) / "output")
    executor.command(["mkdir", "-p", output])
    environment = {**configuration.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1",
                   "WEFT_BENCH_FRAMEWORK_OUTPUT": output, "WEFT_BENCH_VERIFY_OUTPUT": output,
                   "TRITON_CACHE_DIR": output + "/triton-cache", "TRITON_DUMP_DIR": output + "/triton-dump",
                   "TORCHINDUCTOR_CACHE_DIR": output + "/inductor-cache"}
    completed = executor.command([configuration["python"], "probe.py", entry["case"]],
                                 cwd=executor.workspace, env=environment, affinity=True, check=False,
                                 unset_env=configuration.get("unset_environment", ()))
    (directory / "worker.log").write_text(completed.stdout + completed.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2) + "\n")
    _download_output(executor, output, directory)
    rows = [json.loads(line[len("WB_RESULT "):]) for line in completed.stdout.splitlines()
            if line.startswith("WB_RESULT ")]
    result = rows[-1] if rows else {"status": "runtime_failed", "error": completed.stderr or completed.stdout,
                                   "exit_code": completed.returncode, "stage": "worker_process"}
    status = result["status"]
    if status == "ok" and not completed.returncode:
        status = "passed"
    elif status in ("ok", "failed"):
        status = "numerical_failed" if "numerical mismatch" in result.get("error", "") else kind + "_failed"
    return {**result, "status": status, "checked": status == "passed", "diagnostics": str(directory),
            "runtime_configuration": configuration}


def run_usage(entry, target, directory):
    configuration = target.settings.get("verification", {}).get("runtime", {})
    configuration = {**configuration, **entry.get("runtime", {})}
    if not configuration.get("python"):
        raise ValueError("verification.runtime.python is required")
    executor = Executor(target, directory / "worker")
    here = Path(__file__).parent
    author_path = "author/" + Path(entry["source"]).name
    files = [(here / "usage_process.py", "usage_process.py"),
             (here.parent / "adapters/compile_triton.py", "compile_triton.py"),
             (Path(entry["source"]), author_path)]
    for namespace, root in entry.get("imports", {}).items():
        if not namespace.isidentifier():
            raise ValueError("usage import roots must have a Python package name")
        root = Path(root)
        if root.is_file():
            files.append((root, str(Path("author") / (namespace + ".py"))))
        elif root.is_dir():
            files.extend((path, str(Path("author") / namespace / path.relative_to(root))) for path in root.rglob("*.py"))
        else:
            raise FileNotFoundError(root)
    deployed = executor.upload_many(files)
    output = str(Path(executor.workspace) / "output")
    toolchain = executor.write("toolchain.json", json.dumps(configuration.get("toolchain", {})))
    request = {"source": deployed[author_path], "output": output, "toolchain_file": toolchain,
               "size": entry.get("size", 4097), "threads": entry.get("threads", 2),
               "contract": entry.get("contract", "use"), "operation": entry.get("operation", "affine")}
    request_path = executor.write("request.json", json.dumps(request, indent=2))
    environment = {**configuration.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1",
                   "TRITON_CACHE_DIR": output + "/triton-cache",
                   "TORCHINDUCTOR_CACHE_DIR": output + "/inductor-cache",
                   "TRITON_DUMP_DIR": output + "/triton-dump"}
    completed = executor.command([configuration["python"], deployed["usage_process.py"], request_path],
                                 cwd=executor.workspace, env=environment, affinity=True, check=False,
                                 unset_env=configuration.get("unset_environment", ()))
    (directory / "worker.log").write_text(completed.stdout + completed.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2) + "\n")
    _download_output(executor, output, directory)
    rows = [json.loads(line[len("WB_RESULT "):]) for line in completed.stdout.splitlines()
            if line.startswith("WB_RESULT ")]
    result = rows[-1] if rows else {"status": "runtime_failed", "checked": False,
                                   "error": completed.stderr or completed.stdout, "exit_code": completed.returncode}
    return {**result, "diagnostics": str(directory), "runtime_configuration": configuration}

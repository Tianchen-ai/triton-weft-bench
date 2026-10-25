"""Compile native deployment kernels, package and bind them before inference."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from ..adapters.triton_native import compile_source
from ..execution import Executor, ExecutionError
from ..libraries import WeftKernel


def retained_replacements(spec, bindings, package):
    """Explicit AOT inputs are copied into the package that survives replacement."""
    for role, entry in spec.items():
        if entry["artifact"]:
            group, symbol = role.split("/", 1)
            entry["artifact"] = str(package / "kernels" / bindings[group][symbol])
    return spec


def prepare_native_replacements(mapping, target, output):
    from .replacements import role_case

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    runtime = {"compilation_host": "target", **target.settings["llama"]["triton"]}
    if runtime["compilation_host"] not in ("host", "target"):
        raise ValueError("model compilation_host must be host or target")
    artifacts, remote_artifacts, bindings, spec = {}, {}, {}, {}
    for index, (role, candidate) in enumerate(mapping.items()):
        case = role_case(role)
        if isinstance(candidate, str):
            candidate = WeftKernel() if candidate == "default" else WeftKernel.parse(candidate)
        elif isinstance(candidate, dict):
            candidate = WeftKernel(**candidate)
        if not isinstance(candidate, WeftKernel):
            raise TypeError("replacement value must be a WeftKernel")
        name = "kernel_" + str(index)
        spec[role] = asdict(candidate)
        if candidate.artifact:
            artifact = Path(candidate.artifact).expanduser().resolve()
        else:
            options = {**runtime, **runtime.get("bindings", {}).get(role, {})}
            if candidate.source:
                options.update(source=candidate.source, symbol=candidate.symbol)
            if candidate.compiler:
                raise ValueError("Triton replacements select a toolchain, not a weft-compile binary")
            options["backend_options"] = {**options.get("backend_options", {}), **candidate.options}
            build = directory / name
            build.mkdir(exist_ok=True)
            artifact = compile_source(case, target, options, build)
            if runtime.get("compilation_host") == "target":
                remote_artifacts[name] = json.loads((build / "triton-compile.json").read_text())["output"]
        description = json.loads((artifact / "manifest.json").read_text())
        if description.get("producer") != "triton" or description.get("dispatch") != "triton_weft_instance_dispatch":
            raise ValueError("Triton model roles require exported native single-instance artifacts")
        artifacts[name] = str(artifact)
        group, entry = role.split("/", 1)
        bindings.setdefault(group, {})[entry] = name
    if runtime.get("compilation_host") == "target":
        from ..adapters.triton_source import collect_artifact
        executor = Executor(target, directory / "deployment")
        for name, artifact in artifacts.items():
            if name in remote_artifacts:
                continue
            archive = directory / (name + ".tar.gz")
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(artifact, arcname=name)
            remote = executor.upload(archive, name + ".tar.gz")
            executor.command(["tar", "-xf", remote, "-C", executor.workspace], stage="deploy")
            remote_artifacts[name] = executor.workspace + "/" + name
            archive.unlink()
        request = {"producer": "triton", "artifacts": remote_artifacts,
                   "bindings": bindings, "output": executor.workspace + "/package"}
        request_path = executor.write("package-request.json", json.dumps(request, indent=2))
        done = executor.command([runtime["python"], "-m", "weft_bench.models.package_triton", request_path],
                                affinity=True, check=False, unset_env=("PYTHONPATH",))
        (directory / "package.log").write_text(done.stdout + done.stderr)
        if done.returncode:
            raise ExecutionError("target_package", executor.commands[-1], done)
        if executor.host:
            collect_artifact(executor, "package", executor.local)
        package = directory / "native-package"
        if package.exists():
            shutil.rmtree(package)
        (executor.local / "package").rename(package)
        manifest = {"producer": "triton", "replacements": retained_replacements(spec, bindings, package), "bindings": bindings,
                    "package": request["output"] if executor.host else str(package.resolve()),
                    "source_package": str(package.resolve()), "compilation_host": "target",
                    "native_link": "compiled and packaged on the execution target before inference"}
        (directory / "replacements.json").write_text(json.dumps(manifest, indent=2))
        (executor.local / "commands.json").write_text(json.dumps(executor.commands, indent=2))
        return manifest
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if "pythonpath" in runtime:
        environment["PYTHONPATH"] = os.pathsep.join(runtime["pythonpath"])
    package = directory / "native-package"
    with tempfile.TemporaryDirectory(prefix=".native-package-", dir=directory) as temporary:
        staging = Path(temporary) / "package"
        request = {"producer": "triton", "artifacts": artifacts, "bindings": bindings, "output": str(staging.resolve())}
        request_path = directory / "package-request.json"
        request_path.write_text(json.dumps(request, indent=2))
        command = [runtime["python"], str(Path(__file__).with_name("package_triton.py")), str(request_path.resolve())]
        done = subprocess.run(command, env=environment, text=True, capture_output=True,
                              timeout=target.settings.get("timeout", 1800))
        (directory / "package.log").write_text(done.stdout + done.stderr)
        (directory / "package-command.json").write_text(json.dumps(command, indent=2))
        if done.returncode:
            raise ExecutionError("package", command, done)
        if package.exists():
            shutil.rmtree(package)
        staging.rename(package)
    executor = Executor(target, directory / "deployment")
    archive = directory / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(package, arcname="package")
    remote = executor.upload(archive, "package.tar.gz")
    executor.command(["tar", "-xf", remote, "-C", executor.workspace], stage="deploy")
    manifest = {"producer": "triton", "replacements": retained_replacements(spec, bindings, package), "bindings": bindings,
                "package": executor.workspace + "/package", "source_package": str(package.resolve()),
                "native_link": "already linked by Triton; no target compiler invocation"}
    (directory / "replacements.json").write_text(json.dumps(manifest, indent=2))
    (directory / "deployment/commands.json").write_text(json.dumps(executor.commands, indent=2))
    return manifest

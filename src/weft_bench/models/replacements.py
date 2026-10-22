"""Parameter-selected llama.cpp bindings, delegated to Weft's package authority."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from ..adapters.native import compile_artifact
from ..execution import Executor, ExecutionError
from ..libraries import WeftKernel
from ..operators import get


def role_case(role):
    group, separator, entry = role.partition("/")
    if not separator:
        raise ValueError("replacement must be group/role=kernel")
    if group == "ggml.vec_dot":
        return get(entry.lower() + "_dot")
    if group == "ggml.from_float":
        return get(entry.lower() + "_quantize")
    if group == "ggml.ime2" and entry in ("i8_decode", "i8_prefill"):
        return get("ime2_" + entry)
    if group == "ggml.amx" and entry in ("q4_0_decode", "q4_0_prefill"):
        return get("q4_0_projection_" + entry.removeprefix("q4_0_"))
    if group == "ggml.amx" and entry in ("q4_K_decode", "q4_K_prefill", "q6_K_decode", "q6_K_prefill"):
        storage, phase = entry.rsplit("_", 1)
        return get(storage.lower() + "_projection_" + phase)
    if group == "ggml.ime1":
        names = {
            "q4_centered_prefill": "ime_centered",
            "q4_affine_prefill": "ime_affine",
            "q4_centered_decode": "ime_centered_decode",
            "q4_affine_decode": "ime_affine_decode",
        }
        if entry in names:
            return get(names[entry])
    raise ValueError(
        f"unsupported llama.cpp replacement role {role!r}; select ggml.vec_dot, ggml.from_float, ggml.ime1, ggml.ime2 or ggml.amx"
    )


def parse_mapping(values):
    result = {}
    for value in values:
        role, separator, kernel = value.partition("=")
        if not separator or not kernel:
            raise ValueError(
                "replacement must be group/role=file.py:symbol or artifact-directory"
            )
        if role in result:
            raise ValueError("duplicate replacement " + role)
        role_case(role)
        result[role] = WeftKernel() if kernel == "default" else WeftKernel.parse(kernel)
    return result


def prepare_replacements(mapping, target, output):
    if not mapping:
        raise ValueError("provide at least one replacement")
    if target.settings.get("llama", {}).get("package_backend") == "triton":
        from .triton_package import prepare_native_replacements

        return prepare_native_replacements(mapping, target, output)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    bindings = {}
    spec = {}
    for index, (role, kernel) in enumerate(mapping.items()):
        case = role_case(role)
        if isinstance(kernel, str):
            kernel = WeftKernel() if kernel == "default" else WeftKernel.parse(kernel)
        elif isinstance(kernel, dict):
            kernel = WeftKernel(**kernel)
        if not isinstance(kernel, WeftKernel):
            raise TypeError("replacement value must be WeftKernel")
        spec[role] = asdict(kernel)
        name = "kernel_" + str(index)
        if kernel.artifact:
            artifact = Path(kernel.artifact).expanduser()
        else:
            options = {
                **target.implementations.get("weft", {}),
                **kernel.configuration(),
            }
            artifact = compile_artifact(
                case, target, options, Executor(target, directory / name)
            )
        artifacts[name] = str(artifact.resolve())
        group, entry = role.split("/", 1)
        bindings.setdefault(group, {})[entry] = name
    root = target.settings["weft_root"]
    python = target.settings.get("weft_python", sys.executable)
    helper = Path(__file__).with_name("package_weft.py")
    request = {
        "action": "create",
        "root": root,
        "artifacts": artifacts,
        "bindings": bindings,
        "output": str((directory / "source-package").resolve()),
    }
    request_path = directory / "package-request.json"
    request_path.write_text(json.dumps(request, indent=2))
    command = [python, str(helper), str(request_path.resolve())]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=target.settings.get("timeout", 1800),
    )
    (directory / "package.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise ExecutionError("package", command, completed)
    executor = Executor(target, directory / "deployment")
    archive = directory / "package.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(directory / "source-package", arcname="source-package")
    remote_archive = executor.upload(archive, "package.tar")
    executor.command(
        ["tar", "-xf", remote_archive, "-C", executor.workspace], stage="deploy"
    )
    remote_helper = executor.upload(helper, "package_weft.py")
    request = {
        "action": "link",
        "source": executor.workspace + "/source-package",
        "output": executor.workspace + "/package",
        "toolchain": target.settings["toolchain"],
    }
    deployment = target.settings.get("deployment", {})
    if deployment.get("weft_root"):
        request["root"] = deployment["weft_root"]
    request_remote = executor.write("link-request.json", json.dumps(request, indent=2))
    done = executor.command(
        [deployment.get("python", "python3"), remote_helper, request_remote],
        stage="package",
    )
    (directory / "link.log").write_text(done.stdout + done.stderr)
    manifest = {
        "replacements": spec,
        "bindings": bindings,
        "package": request["output"],
        "source_package": str((directory / "source-package").resolve()),
    }
    (directory / "replacements.json").write_text(json.dumps(manifest, indent=2))
    (directory / "deployment/commands.json").write_text(
        json.dumps(executor.commands, indent=2)
    )
    return manifest


def prepare_observer(config, target, output):
    """One-time harness preparation for a configured integrated llama.cpp build."""
    if not config.get("source") or not config.get("weft_include"):
        return {
            "prepared": False,
            "reason": "using supplied binary; source/include not configured for observer preparation",
        }
    build = config.get("build_directory")
    if not build:
        raise ValueError("configure build_directory when preparing the model observer")
    directory = Path(output)
    executor = Executor(target, directory)
    runtime = target.settings.get("triton_runtime", {})
    if target.settings.get("llama", {}).get("package_backend") == "triton" and runtime.get("library"):
        sdk = Path(runtime["library"]).parent.parent
        if Path(config["weft_include"]) != sdk / "share/llama.cpp/include":
            raise ValueError("model observer headers and selected native runtime must come from the same SDK")
        configured = executor.command(
            ["cmake", "-S", config["source"], "-B", build,
             "-DWeft_DIR=" + str(sdk / "lib/cmake/Weft")],
            stage="model_sdk_configure",
            check=False,
        )
        (directory / "configure.log").write_text(configured.stdout + configured.stderr)
        (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
        if configured.returncode:
            raise ExecutionError("model_sdk_configure", executor.commands[-1], configured)
    helper = executor.upload(Path(__file__).with_name("observe.py"), "observe.py")
    request = executor.write(
        "observe.json",
        json.dumps(
            {"source": config["source"], "weft_include": config["weft_include"]}
        ),
    )
    done = executor.command(
        [
            target.settings.get("deployment", {}).get("python", "python3"),
            helper,
            request,
        ],
        stage="model_prepare",
    )
    state = json.loads(done.stdout)
    compiled = executor.command(
        ["cmake", "--build", build, "--target", "llama-bench", "-j2"],
        stage="model_build",
        check=False,
    )
    (directory / "build.log").write_text(compiled.stdout + compiled.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
    if compiled.returncode:
        raise ExecutionError("model_build", executor.commands[-1], compiled)
    return {"prepared": True, **state}

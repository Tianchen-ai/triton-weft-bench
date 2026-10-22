"""Explicit Triton source/AOT provider for the shared native benchmark worker."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from ..core import NativeCall, PreparedNative
from ..execution import ExecutionError
from ..libraries import weft_call
from .cpu_target import selected_target


def replay_options(options, case_name, directory, *, source=None, compiler=None):
    if compiler:
        raise ValueError("Triton replay selects its Python/toolchain in the implementation configuration; --compiler is the legacy Weft CLI override")
    options = {**options, **options.get("bindings", {}).get(case_name, {})}
    options.pop("bindings", None)
    artifact = directory / "artifact"
    if source is None and (artifact / "triton.json").is_file():
        options.pop("source", None)
        options.pop("path", None)
        return {**options, "artifact": str(artifact.resolve())}
    request_path = directory / "triton-compile.json"
    if not request_path.is_file():
        raise NotImplementedError("source replay needs the original Triton compile request; this run contains only AOT input")
    request = json.loads(request_path.read_text())
    for key in ("symbol", "signature", "constexprs", "backend_options", "toolchain"):
        options[key] = request[key]
    options["compile_target"] = request["target"]
    options.pop("program", None)
    if request.get("program"):
        options["program"] = request["program"]
    options.pop("artifact", None)
    options.pop("path", None)
    options["source"] = source or str((directory / "author-source.py").resolve())
    options["pythonpath"] = [*options.get("pythonpath", os.environ.get("PYTHONPATH", "").split(os.pathsep)),
                             str(Path(request["source"]).parent)]
    return options


def grid_arguments(abi, workload, dimensions, aliases):
    """Pack the exported launch ABI; pointers borrow the existing workload buffers."""
    dimensions = {**abi.get("bindings", {}), **dimensions}
    statements, values = [], []
    strides = []
    for index, arg in enumerate(abi["arguments"]):
        role = aliases.get(arg["name"], arg["name"])
        if arg["name"].startswith(("__triton_pid_", "__triton_grid_")):
            values.append("nullptr")  # The exported launcher supplies these values.
        elif arg["kind"] in ("view", "pointer"):
            values.append("const_cast<void*>(static_cast<const void*>(" + workload.buffers[role] + "))")
            if arg["kind"] == "view" and arg["strided"]:
                shape = [int(x) if str(x).isdecimal() else dimensions[x] for x in arg["shape"]]
                selected = workload.strides.get(role, tuple(math.prod(shape[i + 1:]) for i in range(len(shape))))
                if len(selected) != len(shape):
                    raise ValueError("workload stride rank differs from native View " + role)
                strides.extend(selected)
        elif arg["kind"] == "scalar":
            variable = "launch_scalar_" + str(index)
            statements.append(arg["c_type"] + " " + variable + "=" + workload.scalars[role] + ";")
            values.append("&" + variable)
        else:
            raise ValueError("unsupported native launch argument " + arg["name"])
    extras = [dimensions[d] for d in abi["shape_parameters"]] + strides
    for index, value in enumerate(extras):
        variable = "launch_extra_" + str(index)
        statements.append("size_t " + variable + "=" + str(value) + ";")
        values.append("&" + variable)
    statements.append("void *launch_arguments[]={" + ",".join(values) + "};")
    return "".join(statements)


def compile_source(case, target, options, directory):
    from .compile_triton import clear_backend_failures

    clear_backend_failures(directory)
    source = Path(options["source"]).expanduser().resolve()
    if not options.get("python") or "signature" not in options:
        raise ValueError("Triton source needs its Python environment and ASTSource signature")
    request = {
        "engine": "triton", "source": str(source), "symbol": options["symbol"],
        "signature": options["signature"], "constexprs": options.get("constexprs", {}),
        "backend_options": options.get("backend_options", {}),
        "toolchain": options["toolchain"], "target": selected_target(options, target.settings["target"]),
        "output": str((directory / "artifact").resolve()),
    }
    if options.get("program"):
        request["program"] = options["program"]
    request_path = directory / "triton-compile.json"
    output = directory / "artifact"
    if output.exists():
        shutil.rmtree(output)
    request_path.write_text(json.dumps(request, indent=2))
    shutil.copy2(source, directory / "author-source.py")
    command = [options["python"], str(Path(__file__).with_name("compile_triton.py")), str(request_path.resolve())]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if "pythonpath" in options:
        if not isinstance(options["pythonpath"], list) or not all(isinstance(p, str) for p in options["pythonpath"]):
            raise ValueError("Triton pythonpath must be a list of import directories")
        environment["PYTHONPATH"] = os.pathsep.join(options["pythonpath"])
    (directory / "compile-commands.json").write_text(json.dumps([command], indent=2))
    result = subprocess.run(command, env=environment, text=True, capture_output=True,
                            timeout=target.settings.get("timeout", 1800))
    (directory / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ExecutionError("triton_compile", command, result)
    return Path(request["output"])


class TritonAdapter:
    @staticmethod
    def configure_candidate(case, options, candidate):
        binding = options.get("bindings", {}).get(case.name, {})
        original = {**options, **binding}
        changing_source = candidate.source and (candidate.source != original.get("source") or
                                               candidate.symbol != original.get("symbol"))
        # Source-specific constants, packing and workspaces describe the chosen
        # author program. A replacement inherits the target environment, not the
        # unrelated program's signature or tuning choice.
        options = {**options, "signature": {}, "constexprs": {}} if changing_source else original
        options.pop("bindings", None)
        if candidate.compiler:
            raise ValueError("Triton candidates use their configured Python/toolchain, not --compiler")
        if candidate.source:
            options.pop("artifact", None)
            options.pop("path", None)
            options.update(source=candidate.source, symbol=candidate.symbol)
        if candidate.artifact:
            options.pop("source", None)
            options.pop("path", None)
            options["artifact"] = candidate.artifact
        if candidate.options:
            options["backend_options"] = {**options.get("backend_options", {}), **candidate.options}
        if candidate.arguments:
            options["arguments"] = candidate.arguments
        return options

    def prepare(self, case, target, options, directory, worker, workload, name):
        options = {**options, **options.get("bindings", {}).get(case.name, {})}
        entry = options.get("entry", "single")
        if entry not in ("single", "grid"):
            raise ValueError("Triton entry must be single or grid")
        if entry == "single" and target.settings.get("threads", 1) != 1:
            raise NotImplementedError("Triton single-instance entry requires a one-thread target; use the explicit grid entry for parallel execution")
        if entry == "grid" and (workload.before_call or workload.after_call):
            raise NotImplementedError("grid requires whole-workload buffers, without an outer row/tile loop")
        if entry == "grid":
            grid = options.get("grid")
            threads = options.get("num_cpu_threads")
            if not isinstance(grid, list) or not 1 <= len(grid) <= 3 or any(type(x) is not int or x < 1 for x in grid):
                raise ValueError("grid must provide one to three positive integer dimensions")
            if type(threads) is not int or threads < 1:
                raise ValueError("grid measurement needs an explicit positive num_cpu_threads")
            grid = [*grid, *([1] * (3 - len(grid)))]
        configured = options.get("artifact") or options.get("path")
        if configured and options.get("source"):
            raise ValueError("select Triton source or an AOT artifact, not both")
        if configured:
            original = Path(configured).expanduser().resolve()
            artifact = directory / "artifact"
            if original != artifact.resolve():
                if artifact.exists():
                    shutil.rmtree(artifact)
                shutil.copytree(original, artifact)
        elif options.get("source"):
            artifact = compile_source(case, target, options, directory)
        else:
            raise ValueError("Triton adapter needs source/symbol or an exported artifact")
        manifest = json.loads((artifact / "manifest.json").read_text())
        metadata = json.loads((artifact / "triton.json").read_text())
        if manifest.get("producer") != "triton" or manifest.get("native_library") is not True:
            raise ValueError("candidate must be an exported Triton native artifact")
        if manifest.get("dispatch") != "triton_weft_instance_dispatch":
            raise ValueError("candidate does not expose the Triton single-instance dispatch ABI")
        expected_target = selected_target(options, target.settings["target"])
        for key, value in expected_target.items():
            if manifest["target"].get(key) != value:
                raise ValueError(f"Triton artifact target differs in {key}: "
                                 f"requested {value!r}, artifact {manifest['target'].get(key)!r}")
        abi = manifest["kernel"]
        if any(a["name"].startswith("__triton_") for a in abi["arguments"]):
            raise ValueError("single-instance measurement cannot use the internal grid ABI")
        if not (artifact / "kernel.so").is_file():
            raise FileNotFoundError("Triton artifact has no native kernel.so")
        aliases = options.get("arguments", {})
        from .workspaces import invocation_workspaces
        workload, scratch, scratch_details = invocation_workspaces(case, workload, abi, options.get("workspaces", []))
        if any(name in aliases or name in aliases.values() for name in options.get("workspaces", [])):
            raise ValueError("workspace arguments cannot alias workload inputs or outputs")
        from .persistent import persistent_inputs
        workload, packing, packed_bases, packed_details = persistent_inputs(
            case, workload, abi, options.get("persistent_inputs", []), aliases, options.get("persistent_providers"))
        # The same binder/checks as existing kernels, but invocation uses a loaded function pointer.
        call = weft_call(name, case, workload, {**abi, "symbol": "native_entry"}, aliases, packed_bases)
        runtime = options.get("native_runtime", target.settings.get("triton_runtime", {}))
        if not runtime.get("include") or not runtime.get("library"):
            raise NotImplementedError("configure target-side triton_runtime.include/library with open_native and check_worker")
        remote = {}
        for filename in ("kernel.so", "manifest.json"):
            remote[filename] = worker.upload(artifact / filename, name + "/" + filename)
        worker.upload(artifact / "kernel.h", name + "-kernel.h")
        for filename in metadata["runtime_files"]:
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts or path.parent == Path("."):
                raise ValueError("artifact runtime files must remain in a relative subdirectory")
            worker.upload(artifact / path, name + "/" + filename)
        launch_details = {}
        if entry == "grid":
            if "grid" not in manifest:
                raise NotImplementedError("re-export this Triton artifact with its grid manifest and launcher declaration")
            grid_manifest = manifest["grid"]
            abi = grid_manifest["kernel"]
            launcher = grid_manifest["launcher"]
            if not launcher.isascii() or not launcher.isidentifier():
                raise ValueError("invalid exported grid launcher symbol")
            remote["manifest.json"] = worker.write(name + "/grid-manifest.json", json.dumps(grid_manifest))
            launch_details = {"grid": grid, "threads": threads}
        entry_type = ""
        pointer_type = "decltype(&" + abi["symbol"] + ")"
        if entry == "single":
            declaration = abi["declaration"]
            prefix = "void " + abi["symbol"] + "("
            if not declaration.startswith(prefix) or not declaration.endswith(");"):
                raise ValueError("native artifact has an unsupported typed declaration")
            # Each loaded artifact owns its symbol namespace. Derive the pointer
            # type from its ABI instead of declaring competing global C symbols.
            parameters = re.sub(r"\brestrict\b", "__restrict__", declaration[len(prefix):])
            entry_type = "using native_entry_type = void (*)(" + parameters
            pointer_type = "native_entry_type"
        setup = (
            entry_type
            + "wb::TritonNative native(" + json.dumps(remote["kernel.so"]) + ","
            + json.dumps(remote["manifest.json"]) + "); native.check_worker();"
            + "auto native_entry=native.entry<" + pointer_type + ">();"
        )
        setup += scratch + packing
        if entry == "grid":
            setup += "auto native_launcher=native.launcher<decltype(&" + launcher + ")>(" + json.dumps(remote["kernel.so"]) + "," + json.dumps(launcher) + ");"
            setup += grid_arguments(abi, workload, case.shape, aliases)
            call = replace(call, invoke="native.grid(native_launcher,reinterpret_cast<weft_native_entry>(native_entry),launch_arguments,"
                           + ",".join(str(x) for x in grid) + "," + str(threads) + ");")
        includes = call.includes if entry == "grid" else tuple(h for h in call.includes if h != name + "-kernel.h")
        call = replace(call, prepare=call.prepare + setup,
                       includes=(*includes, "triton_native.h"))
        framework_weights = any(p["provider"] == "ggml-model-default/AMX" for p in packed_details)
        if framework_weights:
            call = replace(call, includes=(*call.includes, "projection_reference.h"))
        flags = ["-I" + runtime["include"], runtime["library"],
                 "-Wl,-rpath," + str(Path(runtime["library"]).parent), *runtime.get("ldflags", [])]
        return PreparedNative(call, flags, {
            "producer": "triton", "entry": "single-instance" if entry == "single" else "grid",
            "native_call_boundary": "whole-workload" if call.whole_workload else "workload-traversal",
            "compile_target": expected_target,
            "capability_check": ("SDK check_worker before the bounded native measurement scope" if entry == "single"
                                 else "SDK check_worker in every native launch worker before writes"),
            "launch": launch_details,
            "invocation_workspaces": scratch_details,
            "persistent_inputs": packed_details,
            "artifacts": {"weft": str(artifact.resolve())},
            "candidate_native_build": metadata["native_build"],
            "compiler_decisions": metadata["compilation"],
            "physical_program": {key: abi[key] for key in ("physical_resources", "physical_operations") if key in abi},
            "required_extensions": abi["required_extensions"],
            "used_extensions": abi["used_extensions"],
            "native_runtime": runtime,
        }, include_flags=["-DWB_GGML_MODEL_WEIGHTS=1"] if framework_weights else [])

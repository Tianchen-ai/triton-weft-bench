"""Outer build/deployment orchestration for the transport-free native bench core."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile

from .core import worker_source
from .diagnostics import save_request, failure_stage
from .execution import Executor, ExecutionError
from .inputs import Inputs, native_inputs
from .libraries import ggml_call, weft_call
from .types import Measurement, Result
from .workloads import configured_workload, parameter_defaults, ports
from .adapters.native import NATIVE, compile_artifact, flags_for, ggml_flags
from .operators.implementations import resolve_binding
from .results import MIXED_PROJECTION_SCOPE, projection_scope

BUILTINS = {
    "weft_bench.adapters.native:WeftAdapter": "weft",
    "weft_bench.adapters.native:ArtifactAdapter": "artifact",
    "weft_bench.adapters.native:GGMLAdapter": "ggml",
    "weft_bench.adapters.triton_native:TritonAdapter": "triton",
    "weft_bench.adapters.triton_jit:TritonJitAdapter": "triton_jit",
}


def mode(name, target):
    if target.implementations.get(name, {}).get("kind") == "native":
        return "native"
    factory = target.implementations.get(name, {}).get("factory")
    return (
        BUILTINS.get(factory)
        if factory
        else {"weft": "weft", "ggml": "ggml", "artifact": "artifact"}.get(name)
    )


def _failed(row, error, stage="prepare"):
    stage = error.stage if isinstance(error, ExecutionError) else stage
    row.status = (
        "unsupported" if isinstance(error, NotImplementedError) else stage + "_failed"
    )
    row.error = str(error)
    row.details["failure_stage"] = stage
    row.details["compiler_stage"] = failure_stage(row.error)


def _single_thread_boundary(row, requested_threads):
    """A direct instance does not acquire a team from the target environment."""
    entry = row.details.get("entry", row.details.get("execution", {}).get("entry"))
    if entry in ("single", "single-instance"):
        declared_threads = row.details["execution"].get("threads")
        row.details["execution"]["threads"] = 1
        if requested_threads != 1 or declared_threads != 1:
            row.details["requested_threads"] = requested_threads
            row.details["declared_threads"] = declared_threads
            raise NotImplementedError(
                "single-instance candidate executes on one thread; select a one-thread target "
                "or configure its existing grid/framework entry for parallel execution"
            )


def _compile_details(row, directory):
    is_triton = (directory / "triton-compile.json").exists() or (directory / "artifact/triton.json").exists()
    if (directory / "triton-compile.json").exists():
        row.details["build"] = json.loads((directory / "triton-compile.json").read_text())
        row.details["triton_cache"] = str((directory / "triton-cache").resolve())
    if (directory / "artifact/triton.json").is_file():
        row.details.setdefault("artifacts", {})["weft"] = str((directory / "artifact").resolve())
    if not is_triton and (directory / "compile.json").exists():
        row.details["build"] = json.loads((directory / "compile.json").read_text())
    if not is_triton and (directory / "canonical.mlir").exists():
        row.details.setdefault("artifacts", {})["canonical"] = str(
            (directory / "canonical.mlir").resolve()
        )
    if not is_triton and (directory / "effective-options.json").exists():
        row.details.setdefault("build", {})["effective_options"] = json.loads(
            (directory / "effective-options.json").read_text()
        )


def _record_projection_boundary(row, workload, call, provider, source):
    """Capture the prepared native call, without changing its code or timer."""
    if (provider not in ("ggml", "weft", "artifact", "triton")
            or workload.header != "projection_inputs.h"):
        return
    scope = projection_scope(row.scope)
    if scope is None:
        return
    row.details["projection_boundary"] = {
        "schema": 1,
        "scope": scope,
        "weights": "persistent-before-timing",
        "activation": "preserved-f32-in-call" if scope == MIXED_PROJECTION_SCOPE else "preparation-in-call",
        "output": "complete-projection-in-call",
        "evidence": {
            "provider": provider,
            "worker_source": str(Path(source).resolve()),
            "input_setup": workload.setup,
            "weight_input": workload.buffers["W"],
            "activation_input": workload.buffers["X"],
            "preparation": call.prepare,
            "invocation": workload.before_call + call.invoke + workload.after_call,
        },
    }


def run_operator(
    case,
    implementations,
    target,
    measurement=None,
    *,
    output=None,
    inputs=None,
    save_outputs=False,
    _worker_label=None,
):
    measurement = measurement or Measurement()
    root = Path(output or tempfile.mkdtemp(prefix="weft-bench-"))
    if len(set(implementations)) != len(implementations):
        raise ValueError("implementation names must be distinct")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in implementations):
        raise ValueError("implementation names must be identifiers")
    if any(mode(n, target) == "triton_jit" for n in implementations):
        if measurement.sample_order == "alternating":
            raise ValueError("alternating samples require prepared native artifacts in the shared worker")
        from .adapters.triton_jit import TritonJitAdapter

        rows = []
        for name in implementations:
            if mode(name, target) == "triton_jit":
                rows.extend(TritonJitAdapter().run(case, name, target, measurement, output=root,
                                                 inputs=inputs, save_outputs=save_outputs))
            else:
                rows.extend(run_operator(case, [name], target, measurement, output=root,
                                         inputs=inputs, save_outputs=save_outputs))
        return rows
    # Native Triton artifacts use independently loaded function pointers and
    # private per-call outputs/workspaces. Share the immutable workload input
    # preparation, while measuring and checking every candidate separately.
    shared_native = all(port.access != "inout" for port in ports(case)) and all(
        mode(n, target) == "ggml" or (
            mode(n, target) == "triton" and
            {**target.implementations[n],
             **target.implementations[n].get("bindings", {}).get(case.name, {})}.get("entry", "single") == "single"
        ) for n in implementations
    )
    if measurement.sample_order == "alternating" and (not shared_native or len(implementations) < 2):
        raise ValueError("alternating samples require at least two single-instance native implementations with immutable inputs")
    if sum(mode(n, target) != "ggml" for n in implementations) > 1 and not shared_native:
        return [
            r
            for n in implementations
            for r in run_operator(
                case,
                [n],
                target,
                measurement,
                output=root,
                inputs=inputs,
                save_outputs=save_outputs,
                _worker_label=n,
            )
        ]
    selected = inputs or Inputs()
    case = selected.resolve(case)
    worker = Executor(target, root / case.name / ("worker-" + _worker_label if _worker_label else "worker"))
    files = selected.materialize(worker.local / "inputs", ports(case))
    input_record = selected.describe(files)
    # Input content describes the scenario, not a build-hash approval gate.
    identity = {
        "generator": "typed-matmul-v1" if case.family == "matmul" else "ggml-workload-v3",
        "seed": selected.seed,
        "parameters": {**parameter_defaults(case), **selected.parameters},
    }
    identity["buffers"] = {
        k: {p: v for p, v in value.items() if p != "path"} for k, value in files.items()
    }
    quantization_contract = None
    projection_weights = target.settings.get("ggml", {}).get("projection_weights", "plain") if case.family == "projection" else None
    if projection_weights not in (None, "plain", "model-default"):
        raise ValueError("GGML projection_weights must be plain or model-default")
    if projection_weights == "model-default":
        identity["weight_preparation"] = "persistent-model-default"
    mixed_projection = case.family == "projection" and case.data.get("activation_dtype") == "f32"
    if mixed_projection:
        identity["numerical_contract"] = "stored-f16-weights-original-f32-activation"
    if case.family == "quantize":
        fmt = case.dtype.split("->")[1]
        quantization_contract = target.settings.get("quantization_contracts", {}).get(fmt)
        if quantization_contract:
            if fmt != "q8_k" or quantization_contract not in ("extrema-lo-rtne", "first-absolute-max-rtne"):
                raise ValueError("unsupported explicit quantization storage contract")
            identity["quantization_contract"] = quantization_contract
    case = replace(case, data={**case.data, "inputs": input_record})
    rows = []
    calls = []
    objects = []
    native_link_flags = []
    native_include_flags = []
    native_environment = {}
    for name in implementations:
        directory = root / case.name / name
        directory.mkdir(parents=True, exist_ok=True)
        row = Result(case.name, name, target.name, case.shape, case.scope, "pending")
        row.details = {
            "measurement": ({"check_only": True, "protocol": "native-workload-validation-v1"}
                            if measurement.check_only else {**{k: v for k, v in asdict(measurement).items() if k != "check_only"}, "protocol": "native-workload-v3"}),
            "input_policy": json.dumps(identity, sort_keys=True),
            "inputs": input_record,
            "ports": [asdict(p) for p in ports(case)],
            "dtype": case.dtype,
            "phase": case.phase,
            "execution": {
                "cpus": worker.cpus,
                "threads": target.settings.get("threads", 1),
                "isa": target.settings.get("target", {}),
                "env": worker.env,
            },
            "diagnostics": str(directory.resolve()),
            "worker": str(worker.local.resolve()),
            "native_build": target.settings.get("toolchain", {}),
        }
        row.details["replay"] = save_request(case, name, target, measurement, directory)
        if mixed_projection:
            row.details["oracle"] = {
                "independent": True,
                "kind": "stored-f16-weights-original-f32-activation",
                "accumulation": "f64-reference-f32-output",
            }
        if worker.numa:
            row.details["execution"]["numa"] = worker.numa
        rows.append(row)
    try:
        threads = target.settings.get("threads", 1)
        if type(threads) is not int or threads < 1:
            raise ValueError("operator threads must be a positive integer")
        workload = configured_workload(case, target.settings)
        if case.family == "projection" and "-DWB_ACTIVATION_TYPE=GGML_TYPE_Q8_K" in workload.defines:
            activation_contract = target.settings.get("projection_activation_contracts", {}).get("q8_k")
            if activation_contract:
                if activation_contract not in ("extrema-lo-rtne", "first-absolute-max-rtne"):
                    raise ValueError("unsupported projection Q8_K storage contract")
                for row in rows:
                    row.details["oracle"] = {
                        "independent": True, "kind": "decoded-weight-f64-projection",
                        "activation_quantization": activation_contract,
                        "native_activation_check": "all-packed-bytes-before-measurement",
                    }
        tools, flags = flags_for(target)
        needs_ggml = (case.family in ("vec_dot", "quantize", "dequantize", "projection", "ime", "amx_tile")
                      or any(mode(name, target) == "ggml" for name in implementations))
        includes, libraries = ggml_flags(target) if needs_ggml else ([], [])
        worker.upload_many((header, header.name) for header in NATIVE.glob("*.h"))
        for role, info in files.items():
            info["remote"] = worker.upload(info["path"], "inputs/" + role + ".bin")
    except Exception as error:
        for row in rows:
            _failed(row, error)
        return rows
    for row in rows:
        name = row.implementation
        options = dict(target.implementations.get(name, {}))
        directory = root / case.name / name
        try:
            selected_mode = mode(name, target)
            effective = {**options, **options.get("bindings", {}).get(case.name, {})}
            if selected_mode in ("weft", "artifact") or (
                selected_mode == "triton" and effective.get("entry", "single") == "single"
            ):
                row.details["entry"] = "single-instance"
                _single_thread_boundary(row, threads)
            # Packed input layout is portable. Native providers check their
            # actual artifact ISA; retain the existing intrinsic-path guard.
            if case.family == "ime" and selected_mode in ("ggml", "weft", "artifact") and (
                "spacemit-ime1" not in target.settings.get("capabilities", [])
                or len(worker.cpus) != 1
            ):
                raise NotImplementedError(
                    "GGML/legacy IME implementation requires spacemit-ime1 and one eligible CPU"
                )
            if threads != 1 and selected_mode not in ("ggml", "native", "triton"):
                raise NotImplementedError("this candidate uses a single native instance; select its explicit grid adapter for parallel execution")
            if mode(name, target) == "ggml":
                from .adapters.reference import inspect_ggml_compilation

                row.details["reference_compilation"] = inspect_ggml_compilation(
                    target, worker, require_optimized=not measurement.check_only)
                if case.family == "ime2":
                    from .adapters.reference import prepare_ime2_reference
                    if "spacemit-ime2" not in target.settings.get("capabilities", []) or len(worker.cpus) != 1:
                        raise NotImplementedError("IME2 reference requires one eligible A100 CPU")
                    extra, includes = prepare_ime2_reference(target, worker)
                    objects.extend(extra)
                    native_include_flags.extend(includes)
                    row.details["ime2_reference_build"] = str(worker.local / "ime2-reference-build.json")
                if threads != 1 and case.name != "gemm_f32" and case.family != "projection":
                    raise NotImplementedError("this GGML leaf has a single-call scope; select one thread or a complete GEMM/projection workload")
                if case.name == "rope_split":
                    from .adapters.reference import prepare_dense_reference

                    extra, _ = prepare_dense_reference(case, target, worker)
                    objects.extend(extra)
                call = ggml_call(name, case)
                if mixed_projection:
                    from .adapters.reference import prepare_mixed_projection_reference

                    row.details["reference_precision"] = prepare_mixed_projection_reference(target, worker)
                    native_include_flags.append("-DWB_GGML_AMX_F16_F32=1")
                if projection_weights == "model-default":
                    native_include_flags.append("-DWB_GGML_MODEL_WEIGHTS=1")
                    row.scope += "/persistent-model-default-weights"
                    row.details["reference_boundary"] = "persistent model weights prepared outside timing; activation quantization and graph compute inside"
                    if mixed_projection:
                        row.details["reference_boundary"] = "persistent model weights prepared outside timing; preserved F32 activation and graph compute inside"
                row.details["reference_build"] = target.settings["ggml"]
                if case.family == "dequantize":
                    row.details["oracle"] = {"independent": False, "kind": "library-reference-self-consistency",
                                             "function": "ggml_type_traits.to_float"}
            elif selected_mode == "native":
                from .loading import load_object

                adapter = load_object(options["factory"])()
                prepared = adapter.prepare_native(case, target, options, directory, worker, workload, name)
                call = prepared.call
                native_link_flags.extend(prepared.link_flags)
                native_include_flags.extend(prepared.include_flags)
                for key, value in prepared.environment.items():
                    if key in native_environment and native_environment[key] != value:
                        raise ValueError("native implementations require conflicting environment variable " + key)
                    native_environment[key] = value
                row.details.update(prepared.details)
                if prepared.details.get("timing_scope"):
                    row.scope = prepared.details["timing_scope"]
                _single_thread_boundary(row, threads)
            elif mode(name, target) == "triton":
                from .adapters.triton_native import TritonAdapter

                prepared = TritonAdapter().prepare(case, target, options, directory, worker, workload, name)
                call = prepared.call
                native_link_flags.extend(prepared.link_flags)
                native_include_flags.extend(prepared.include_flags)
                for key, value in prepared.environment.items():
                    if key in native_environment and native_environment[key] != value:
                        raise ValueError("native implementations require conflicting environment variable " + key)
                    native_environment[key] = value
                row.details.update(prepared.details)
                if prepared.details["entry"] == "grid":
                    row.scope += "/grid-launch"
                    row.details["execution"].update(entry="grid", **prepared.details["launch"])
            else:
                if mode(name, target) == "artifact":
                    configured = options.get("artifacts", {}).get(
                        case.name, options.get("path")
                    )
                    if not configured:
                        raise ValueError("Weft artifact path is required")
                    artifact = Path(configured).expanduser()
                    if not (artifact / "manifest.json").is_file():
                        raise ValueError(f"{artifact} is not a Weft artifact directory")
                    snapshot = directory / "artifact"
                    if artifact.resolve() != snapshot.resolve():
                        shutil.copytree(artifact, snapshot)
                        artifact = snapshot
                else:
                    local = Executor(target, directory)
                    artifact = compile_artifact(case, target, options, local)
                    (directory / "compile-commands.json").write_text(
                        json.dumps(local.commands, indent=2)
                    )
                manifest = json.loads((artifact / "manifest.json").read_text())
                if any(
                    manifest["target"].get(k) != v
                    for k, v in target.settings["target"].items()
                ):
                    raise ValueError("artifact target differs from execution target")
                metadata = manifest["kernel"]
                if case.family == "projection" and any(
                    a["name"] == "W" and "I[" in a.get("encoding", "")
                    for a in metadata["arguments"]
                ):
                    raise NotImplementedError(
                        "projection expects ordinary GGML weight blocks; select a matching source binding"
                    )
                source = worker.upload(artifact / "kernel.c", name + "/kernel.c")
                header = worker.upload(artifact / "kernel.h", name + "-kernel.h")
                worker.upload(artifact / "kernel.h", name + "/kernel.h")
                obj = worker.workspace + "/" + name + ".o"
                worker.command(
                    [*tools["cc"], *flags, "-std=c11", "-c", source, "-o", obj],
                    stage="native_compile",
                )
                objects.append(obj)
                binding = resolve_binding(case, target, options)
                call = weft_call(
                    name, case, workload, metadata, binding.get("arguments", {})
                )
                row.details["artifacts"] = {"weft": str(artifact.resolve())}
            calls.append(call)
        except Exception as error:
            _failed(row, error)
        finally:
            _compile_details(row, directory)
    if calls:
        def collect_saved_outputs(row):
            call = next(call for call in calls if call.name == row.implementation)
            names = ["Y", *{**workload.snapshots, **call.snapshots}]
            row.details["outputs"] = {}
            for name in names:
                path = root / case.name / row.implementation / (name + ".bin")
                worker.download(worker.workspace + "/" + row.implementation + "-" + name + ".bin", path)
                row.details["outputs"][name] = str(path.resolve())

        try:
            source = worker.write(
                "worker.cpp",
                worker_source(
                    workload,
                    calls,
                    native_inputs(selected, files),
                    save_outputs=save_outputs,
                    record_environment=target.settings.get("record_environment", False),
                    alternating=measurement.sample_order == "alternating",
                ),
            )
            for call in calls:
                row = next(row for row in rows if row.implementation == call.name)
                _record_projection_boundary(row, workload, call, mode(call.name, target),
                                            worker.local / "worker.cpp")
            binary = worker.workspace + "/worker"
            build_command = [
                    *tools["cxx"],
                    *flags,
                    "-std=c++17",
                    *workload.defines,
                    "-DWB_THREADS=" + str(threads),
                    *includes,
                    *native_include_flags,
                    "-I" + worker.workspace,
                    source,
                    *objects,
                    *tools.get("ldflags", []),
                    *libraries,
                    *native_link_flags,
                    "-lm",
                    "-ldl",
                    "-pthread",
                    "-o",
                    binary,
                ]
            for row in rows:
                if any(call.name == row.implementation for call in calls):
                    row.details["native_worker_build_command"] = list(build_command)
            worker.command(build_command, stage="native_compile")
            done = worker.command(
                [binary],
                cwd=worker.workspace,
                env={**native_environment, **measurement.env()},
                affinity=True,
                check=False,
            )
            (worker.local / "stdout.log").write_text(done.stdout)
            (worker.local / "stderr.log").write_text(done.stderr)
            records = {}
            evidence = {}
            environment = None
            for line in done.stdout.splitlines():
                if line.startswith("WB_RESULT "):
                    record = json.loads(line[10:])
                    name = record["implementation"]
                    if name in records:
                        raise ValueError("duplicate native result " + name)
                    records[name] = record
                elif line.startswith("WB_DETAILS "):
                    detail = json.loads(line[len("WB_DETAILS "):])
                    evidence[detail.pop("implementation")] = detail
                elif line.startswith("WB_ENVIRONMENT "):
                    environment = json.loads(line[len("WB_ENVIRONMENT "):])
            for row in rows:
                if native_environment:
                    row.details["execution"]["env"] = {**worker.env, **native_environment}
                if environment is not None:
                    row.details["observed_environment"] = environment
                if done.returncode:
                    row.details["worker_exit"] = {"returncode": done.returncode,
                                                  "stderr": done.stderr}
                row.details.update(evidence.get(row.implementation, {}))
                if row.status != "pending":
                    continue
                if row.implementation not in records:
                    error = (ExecutionError("run", [binary], done) if done.returncode else
                             ValueError("missing native result " + row.implementation))
                    _failed(row, error, "run")
                    continue
                record = records[row.implementation]
                row.details["native"] = record
                if record.get("status") in ("check_failed", "run_failed"):
                    row.status = record["status"]
                    row.error = record["error"]
                    row.details["failure_stage"] = record["stage"]
                    if mixed_projection and record["error"].startswith("projection precision contract mismatch:"):
                        row.status = "contract_mismatch"
                        row.details["failure_stage"] = "contract"
                    if save_outputs and record.get("outputs_available"):
                        collect_saved_outputs(row)
                    continue
                samples = record["samples_ms"]
                if measurement.check_only:
                    if record.get("status") != "checked" or samples:
                        raise ValueError("native worker did not honor check_only")
                else:
                    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                        raise ValueError("invalid native timing")
                    row.samples_ms = samples
                    row.median_ms = statistics.median(samples)
                row.correctness = record["correctness"]
                if row.details.get("oracle", {}).get("independent") is False:
                    row.correctness = row.details["oracle"]["kind"]
                row.status = "ok"
                if "reference_dispatch" in row.details:
                    dispatch = row.details["reference_dispatch"]
                    dispatch["executed_and_checked"] = True
                    if "amx_implementation" in dispatch:
                        dispatch["amx_executed"] = bool(dispatch["amx_implementation"])
                row.details.setdefault("artifacts", {})["executable"] = binary
                if save_outputs:
                    collect_saved_outputs(row)
        except Exception as error:
            for row in rows:
                if row.status == "pending":
                    _failed(row, error, "run")
    (worker.local / "commands.json").write_text(json.dumps(worker.commands, indent=2))
    return rows

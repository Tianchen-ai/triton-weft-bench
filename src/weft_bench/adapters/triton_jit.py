"""Target-native Triton JIT calls with the existing native workload fixture."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
import statistics

from ..core import fixture_source
from ..diagnostics import save_request
from ..execution import Executor, ExecutionError
from ..inputs import Inputs, native_inputs
from ..types import Result
from ..workloads import configured_workload, parameter_defaults, ports
from .native import NATIVE, flags_for, ggml_flags
from .cpu_target import compiler_target, target_architecture
from .triton_source import deploy_source, collect_artifact


class TritonJitAdapter:
    execution_mode = "jit"

    @staticmethod
    def configure_candidate(case, options, candidate):
        from .triton_native import TritonAdapter
        if candidate.artifact:
            raise ValueError("a JIT candidate requires DSL source, not a precompiled artifact")
        return TritonAdapter.configure_candidate(case, options, candidate)

    def run(self, case, name, target, measurement, *, output, inputs=None, save_outputs=False):
        selected = inputs or Inputs()
        case = selected.resolve(case)
        directory = Path(output) / case.name / name
        executor = Executor(target, directory)
        options = target.implementations[name]
        options = {**options, **options.get("bindings", {}).get(case.name, {})}
        row = Result(case.name, name, target.name, case.shape, case.scope, "pending")
        row.details = {
            "measurement": {**asdict(measurement), "protocol": "triton-python-grid-v1"},
            "dtype": case.dtype, "phase": case.phase, "producer": "triton",
            "entry": "python-jit-grid", "diagnostics": str(directory.resolve()),
            "execution": {"cpus": executor.cpus, "threads": options.get("num_cpu_threads"),
                          "grid": options.get("grid"), "isa": target.settings["target"]},
        }
        row.details["replay"] = save_request(case, name, target, measurement, directory)
        stage = "prepare"
        try:
            if options.get("artifact") or options.get("path"):
                raise ValueError("JIT requires DSL source; select the native adapter to validate an exported artifact")
            if measurement.cache != "reuse":
                raise NotImplementedError("JIT calls currently use prepared-data reuse; compiled-cache cold start is measured separately")
            declared = ports(case)
            workload = configured_workload(case, target.settings)
            files = selected.materialize(directory / "inputs", declared)
            row.details["inputs"] = selected.describe(files)
            row.details["ports"] = [asdict(p) for p in declared]
            row.details["input_policy"] = json.dumps({
                "generator": "ggml-workload-v3", "seed": selected.seed,
                "parameters": {**parameter_defaults(case), **selected.parameters},
                "buffers": {role: {key: value for key, value in item.items() if key != "path"}
                            for role, item in files.items()},
            }, sort_keys=True)
            case = replace(case, data={**case.data, "inputs": row.details["inputs"]})
            save_request(case, name, target, measurement, directory)
            for role, item in files.items():
                item["remote"] = executor.upload(item["path"], "inputs/" + role + ".bin")
            executor.upload_many((header, header.name) for header in NATIVE.glob("*.h"))
            tools, flags = flags_for(target)
            fixture = executor.workspace + "/fixture.so"
            source = Path(options["source"]).expanduser().resolve()
            source_request = deploy_source(executor, source)
            grid = options.get("grid", [1])
            threads = options.get("num_cpu_threads", target.settings.get("threads", 1))
            if not isinstance(grid, list) or not 1 <= len(grid) <= 3 or any(type(v) is not int or v < 1 for v in grid):
                raise ValueError("JIT grid requires one to three positive dimensions")
            if type(threads) is not int or threads < 1:
                raise ValueError("JIT num_cpu_threads must be positive")
            request = {
                **source_request,
                "symbol": options["symbol"], "fixture": fixture, "signature": options.get("signature", {}),
                "ports": [asdict(p) for p in declared], "shape": case.shape,
                "parameters": {**parameter_defaults(case), **selected.parameters},
                "arguments": options.get("arguments", {}), "constexprs": options.get("constexprs", {}),
                "toolchain": options.get("toolchain", {}), "backend_options": options.get("backend_options", {}),
                "num_cpu_threads": threads, "grid": grid, "measurement": asdict(measurement),
                "correctness": workload.correctness,
                "artifact": executor.workspace + "/artifact", "save_outputs": save_outputs,
                "check_python_aot": options.get("check_python_aot", False),
                "check_torch_compile": options.get("check_torch_compile", False),
                "profile_runtime": options.get("profile_runtime", False),
                "architecture": target_architecture(target.settings["target"]),
                "compile_target": compiler_target(options["compile_target"]) if "compile_target" in options else None,
            }
            environment = {**options.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1",
                           "TRITON_CACHE_DIR": executor.workspace + "/triton-cache",
                           "TORCHINDUCTOR_CACHE_DIR": executor.workspace + "/torch-cache",
                           "TRITON_DUMP_DIR": executor.workspace + "/triton-dump"}
            command = [options["python"], "-m", "weft_bench.adapters.jit_process"]
            remote_request = executor.write("jit-request.json", json.dumps(request, indent=2))
            stage = "interface"
            described = executor.command([*command, "--describe", remote_request], cwd=executor.workspace,
                                         env=environment, affinity=True, unset_env=("PYTHONPATH",))
            interface = [json.loads(line[10:]) for line in described.stdout.splitlines() if line.startswith("WB_RESULT ")]
            if len(interface) != 1:
                raise ValueError("target returned no unique Canonical interface")
            row.details["target_interface"] = interface[0]
            from .jit_fixture import prepare
            workload, call, metadata, invocation, preparation = prepare(case, workload, interface[0]["interface"], options)
            request.update(invocation=invocation, reset_each_call=bool(workload.reset or call.reset))
            row.details["preparation"] = preparation
            cpp = executor.write("fixture.cpp", fixture_source(workload, call, metadata, native_inputs(selected, files), save_outputs=save_outputs))
            needs_ggml = case.family in ("vec_dot", "quantize", "dequantize", "projection", "ime", "amx_tile")
            includes, libraries = ggml_flags(target) if needs_ggml else ([], [])
            executor.command([*tools["cxx"], *flags, "-std=c++17", "-shared", "-fPIC", *workload.defines,
                              *includes, "-I" + executor.workspace, cpp, *tools.get("ldflags", []), *libraries,
                              "-lm", "-pthread", "-o", fixture], stage="fixture_compile")
            remote_request = executor.write("jit-request.json", json.dumps(request, indent=2))
            stage = "run"
            done = executor.command([*command, remote_request], cwd=executor.workspace,
                                    env=environment, affinity=True, check=False, unset_env=("PYTHONPATH",))
            (directory / "stdout.log").write_text(done.stdout)
            (directory / "stderr.log").write_text(done.stderr)
            records = [json.loads(line[10:]) for line in done.stdout.splitlines() if line.startswith("WB_RESULT ")]
            if len(records) != 1:
                raise ExecutionError("jit_run", [*command, remote_request], done)
            record = records[0]
            row.details["runtime"] = record
            if record.get("compile_target"):
                row.details["compile_target"] = record["compile_target"]
            row.status = record["status"]
            if row.status != "ok":
                row.error = record["error"]
                row.details["failure_stage"] = record["stage"]
            else:
                samples = record["samples_ms"]
                if (not samples and not measurement.check_only) or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError("invalid native JIT timing")
                row.details["python_jit"] = {"samples_ms": samples, "median_ms": statistics.median(samples) if samples else None}
                row.correctness = record["correctness"]
                # A fresh interpreter must load the Triton disk cache. It gets
                # the same source and inputs, never the exported native library.
                cache_request = {**request, "export": False, "save_outputs": False,
                                 "check_python_aot": False, "check_torch_compile": False, "profile_runtime": False,
                                 "measurement": {**request["measurement"], "check_only": True}}
                cache_path = executor.write("jit-cache-request.json", json.dumps(cache_request, indent=2))
                reuse = executor.command([*command, cache_path], cwd=executor.workspace,
                                         env=environment, affinity=True, check=False, unset_env=("PYTHONPATH",))
                (directory / "cache-stdout.log").write_text(reuse.stdout)
                (directory / "cache-stderr.log").write_text(reuse.stderr)
                cache_records = [json.loads(line[10:]) for line in reuse.stdout.splitlines() if line.startswith("WB_RESULT ")]
                if len(cache_records) != 1 or cache_records[0]["status"] != "ok":
                    raise ExecutionError("jit_disk_cache", [*command, cache_path], reuse)
                reuse_record = cache_records[0]
                if not reuse_record["compilations"] or not all(event["cache_hit"] for event in reuse_record["compilations"]):
                    raise ValueError("fresh interpreter did not reuse the Triton compilation cache")
                row.details["disk_cache"] = reuse_record
                if (directory / "artifact").exists():
                    # Local execution already exported at the destination.
                    if executor.host:
                        shutil.rmtree(directory / "artifact")
                if executor.host:
                    collect_artifact(executor, "artifact", directory)
                row.details["artifacts"] = {"weft": str((directory / "artifact").resolve())}
                if save_outputs:
                    executor.download(executor.workspace + "/Y.bin", directory / "Y.bin")
                    row.details["outputs"] = {"Y": str((directory / "Y.bin").resolve())}
                # Compare only the native computation from this successful JIT
                # invocation, using the original worker/oracle/timing contract.
                from ..runner import run_operator
                native_options = {**options, "factory": "weft_bench.adapters.triton_native:TritonAdapter",
                                  "artifact": str((directory / "artifact").resolve()),
                                  "compile_target": record["compile_target"],
                                  "grid": grid, "num_cpu_threads": threads,
                                  "entry": options.get("entry", "grid" if math.prod(grid) > 1 else "single")}
                for key in ("source", "bindings", "path"):
                    native_options.pop(key, None)
                native_target = replace(target, implementations={**target.implementations, name: native_options})
                native_rows = run_operator(case, [name], native_target, measurement,
                                           output=directory / "native", inputs=selected, save_outputs=save_outputs)
                native = native_rows[0]
                native.details["jit"] = row.details
                native.details["replay"] = row.details["replay"]
                native.details["candidate_origin"] = "target-native-kernel-grid-JIT"
                row = native
        except Exception as error:
            from ..runner import _failed
            _failed(row, error, stage)
        if row.status != "ok":
            # Preserve emitted IR/native stages even when a later framework call fails.
            # The remote cache is disposable after the diagnostic archive is collected.
            try:
                archive = executor.workspace + "/diagnostics.tar.gz"
                collected = executor.command(["tar", "-czf", archive, "-C", executor.workspace,
                                              "triton-cache", "triton-dump", "artifact", "failure"],
                                             stage="collect", check=False)
                if collected.returncode in (0, 1, 2):
                    destination = directory / "diagnostics.tar.gz"
                    executor.download(archive, destination)
                    row.details["diagnostic_archive"] = str(destination.resolve())
            except Exception as error:
                row.details["collection_error"] = str(error)
        (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
        return [row]

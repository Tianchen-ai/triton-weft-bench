"""Target-native Triton JIT calls with the existing native workload fixture."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
import statistics
import tarfile

from ..core import fixture_source
from ..diagnostics import save_request
from ..execution import Executor, ExecutionError
from ..inputs import Inputs, native_inputs
from ..types import Result
from ..workloads import native_workload, parameter_defaults, ports
from .native import NATIVE, flags_for
from .cpu_target import compiler_target, target_architecture


class TritonJitAdapter:
    def run(self, case, name, target, measurement, *, output, inputs=None, save_outputs=False):
        selected = inputs or Inputs()
        case = selected.resolve(case)
        directory = Path(output) / case.name / name
        executor = Executor(target, directory)
        options = target.implementations[name]
        options = {**options, **options.get("bindings", {}).get(case.name, {})}
        row = Result(case.name, name, target.name, case.shape, case.scope + "/python-jit-grid", "pending")
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
            if measurement.cache != "reuse":
                raise NotImplementedError("JIT calls currently use prepared-data reuse; compiled-cache cold start is measured separately")
            declared = ports(case)
            if any(port.dtype != "f32" for port in declared) or sum(p.access == "output" for p in declared) != 1:
                raise NotImplementedError("Tensor JIT fixture currently supports float32 inputs and one output")
            workload = native_workload(case)
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
            for header in NATIVE.glob("*.h"):
                executor.upload(header, header.name)
            cpp = executor.write("fixture.cpp", fixture_source(workload, declared, native_inputs(selected, files)))
            tools, flags = flags_for(target)
            fixture = executor.workspace + "/fixture.so"
            executor.command([*tools["cxx"], *flags, "-std=c++17", "-shared", "-fPIC", *workload.defines,
                              "-I" + executor.workspace, cpp, *tools.get("ldflags", []), "-lm", "-pthread", "-o", fixture], stage="fixture_compile")
            source = Path(options["source"]).expanduser().resolve()
            remote_source = executor.upload(source, "author-source.py")
            helper = executor.upload(Path(__file__).with_name("jit_process.py"), "jit_process.py")
            executor.upload(Path(__file__).with_name("triton_source.py"), "triton_source.py")
            executor.upload(Path(__file__).with_name("cpu_target.py"), "cpu_target.py")
            if options.get("profile_runtime"):
                executor.upload(Path(__file__).with_name("runtime_profile.py"), "runtime_profile.py")
            grid = options["grid"]
            threads = options["num_cpu_threads"]
            if not isinstance(grid, list) or not 1 <= len(grid) <= 3 or any(type(v) is not int or v < 1 for v in grid):
                raise ValueError("JIT grid requires one to three positive dimensions")
            if type(threads) is not int or threads < 1:
                raise ValueError("JIT num_cpu_threads must be positive")
            request = {
                "source": remote_source, "symbol": options["symbol"], "fixture": fixture,
                "ports": [asdict(p) for p in declared], "shape": case.shape,
                "parameters": {**parameter_defaults(case), **selected.parameters},
                "arguments": options.get("arguments", {}), "constexprs": options.get("constexprs", {}),
                "toolchain": options["toolchain"], "backend_options": options.get("backend_options", {}),
                "num_cpu_threads": threads, "grid": grid, "measurement": asdict(measurement),
                "correctness": workload.correctness, "output_port": next(p.name for p in declared if p.access == "output"),
                "artifact": executor.workspace + "/artifact", "save_outputs": save_outputs,
                "check_python_aot": options.get("check_python_aot", False),
                "check_torch_compile": options.get("check_torch_compile", False),
                "profile_runtime": options.get("profile_runtime", False),
                "architecture": target_architecture(target.settings["target"]),
                "compile_target": compiler_target(options["compile_target"]) if "compile_target" in options else None,
            }
            remote_request = executor.write("jit-request.json", json.dumps(request, indent=2))
            environment = {**options["environment"], "PYTHONDONTWRITEBYTECODE": "1",
                           "TRITON_CACHE_DIR": executor.workspace + "/triton-cache",
                           "TRITON_DUMP_DIR": executor.workspace + "/triton-dump"}
            stage = "run"
            done = executor.command([options["python"], helper, remote_request], cwd=executor.workspace,
                                    env=environment, affinity=True, check=False)
            (directory / "stdout.log").write_text(done.stdout)
            (directory / "stderr.log").write_text(done.stderr)
            records = [json.loads(line[10:]) for line in done.stdout.splitlines() if line.startswith("WB_RESULT ")]
            if len(records) != 1:
                raise ExecutionError("jit_run", [options["python"], helper, remote_request], done)
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
                if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError("invalid native JIT timing")
                row.samples_ms, row.median_ms = samples, statistics.median(samples)
                row.correctness = record["correctness"]
                archive = executor.workspace + "/artifact.tar.gz"
                executor.command(["tar", "-czf", archive, "-C", executor.workspace, "artifact"], stage="collect")
                local_archive = directory / "artifact.tar.gz"
                executor.download(archive, local_archive)
                if (directory / "artifact").exists():
                    shutil.rmtree(directory / "artifact")
                with tarfile.open(local_archive) as bundle:
                    for member in bundle.getmembers():
                        path = Path(member.name)
                        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                            raise ValueError("unexpected native JIT export member")
                    bundle.extractall(directory)
                row.details["artifacts"] = {"weft": str((directory / "artifact").resolve())}
                if save_outputs:
                    executor.download(executor.workspace + "/Y.bin", directory / "Y.bin")
                    row.details["outputs"] = {"Y": str((directory / "Y.bin").resolve())}
        except Exception as error:
            from ..runner import _failed
            _failed(row, error, stage)
        if row.status != "ok":
            # Preserve emitted IR/native stages even when a later framework call fails.
            # The remote cache is disposable after the diagnostic archive is collected.
            try:
                archive = executor.workspace + "/diagnostics.tar.gz"
                collected = executor.command(["tar", "-czf", archive, "-C", executor.workspace,
                                              "triton-cache", "triton-dump", "artifact"],
                                             stage="collect", check=False)
                if collected.returncode in (0, 1, 2):
                    destination = directory / "diagnostics.tar.gz"
                    executor.download(archive, destination)
                    row.details["diagnostic_archive"] = str(destination.resolve())
            except Exception as error:
                row.details["collection_error"] = str(error)
        (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
        return [row]

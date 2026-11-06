"""Visible file/version identity for resumable verification, without content hashes."""
from pathlib import Path
import json

from ..execution import Executor
from ..loading import load_object


def observe_runtime(target, output):
    configuration = target.settings.get("verification", {}).get("runtime", {})
    if not configuration.get("python"):
        return {"status": "not_configured"}
    executor = Executor(target, Path(output) / "environment")
    implementation = target.settings.get("verification", {}).get(
        "source_implementation", target.settings.get("source_implementation", "triton_source"))
    factory = target.implementations.get(implementation, {}).get("factory")
    producer = getattr(load_object(factory), "producer", "triton") if factory else "triton"
    if producer not in ("triton", "tilelang"):
        raise ValueError("unknown verification runtime producer " + producer)
    script = """import contextlib, importlib, json, os, pathlib, sys
producer = PRODUCER
with contextlib.redirect_stdout(sys.stderr):
    package = importlib.import_module(producer)
    core = importlib.import_module('weft_core') if producer == 'tilelang' else None
roots = [pathlib.Path(package.__file__).parent]
if core is not None:
    roots.append(pathlib.Path(core.__file__).parent)
files = {}
for root in roots:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.so') and '__pycache__' not in path.parts:
            stat = path.stat()
            files[str(path.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
if producer == 'tilelang' and os.environ.get('WEFT_CORE_LIBRARY'):
    path = pathlib.Path(os.environ['WEFT_CORE_LIBRARY']).resolve()
    stat = path.stat()
    files[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
print(json.dumps({'status': 'observed', 'python': sys.executable, 'producer': producer,
                  producer + '_version': package.__version__,
                  producer + '_package': package.__file__, 'files': files}))
""".replace("PRODUCER", repr(producer))
    completed = executor.command([configuration["python"], "-c", script],
                                 env={**configuration.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
                                 unset_env=configuration.get("unset_environment", ()))
    try:
        identity = json.loads(completed.stdout) if not completed.returncode else {
            "status": "unavailable", "error": completed.stderr, "exit_code": completed.returncode}
    except ValueError:
        identity = {"status": "unavailable", "error": completed.stdout + completed.stderr}
    (executor.local / "observed-runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def snapshot(entry, target):
    paths = []
    if entry.get("kind") == "program":
        paths.extend((Path(__file__).with_name("program_process.py"),
                      Path(__file__).parent.parent / "operators/program_oracles.py",
                      Path(__file__).parent.parent / "operators/dense_oracles.py",
                      Path(__file__).parent.parent / "operators/encoded_oracles.py",
                      Path(__file__).parent.parent / "operators/example_oracles.py"))
        paths.extend(entry.get("imports", {}).values())
    if entry.get("kind") in ("framework", "runtime"):
        probes = Path(__file__).with_name("probes")
        paths.append(probes / (entry["kind"] + "_process.py"))
        paths.append(probes / ("framework_kernels" if entry["kind"] == "framework" else "runtime_kernels.py"))
        if entry["kind"] == "runtime":
            if entry["case"] == "distribution":
                paths.append(probes / "cpu_distribution.py")
            paths.extend(probes / name for name in
                         ("riscv_kernels.py", "memory_kernels.py", "numerical_kernels.py",
                          "numerical_primitive_kernels.py", "numerical_primitives.py", "control_kernels.py", "helper_native.py",
                          "packed_layout_kernels.py", "packed_layouts.py",
                          "level_kernels.py", "level_scenarios.py", "bundled_toolchain.py",
                          "dense_view_kernels.py", "dense_view_probe.py", "source_auto_kernels.py", "source_auto_probe.py"))
            if entry["case"] == "dynamic_rounding":
                paths.extend((probes / "dynamic_rounding.py", probes / "dynamic_rounding_kernels.py",
                              Path(__file__).parent.parent / "native/verification/fenv_bridge.c"))
        from .source_inputs import source_files
        configuration = {**target.settings.get("verification", {}).get("runtime", {}), **entry.get("runtime", {})}
        paths.extend(path for path, _ in source_files(configuration))
        paths.append(Path(__file__).parent.parent / "adapters/compile_triton.py")
        paths.append(Path(__file__).parent.parent / "adapters/cpu_target.py")
    if entry.get("kind") == "usage":
        paths.append(Path(__file__).with_name("usage_process.py"))
        paths.extend(entry.get("imports", {}).values())
    options = entry.get("options", {})
    if entry.get("artifact"):
        paths.append(entry["artifact"])
    else:
        paths.extend(target.settings.get("verification", {}).get("provenance_paths", []))
        paths.extend(options.get("pythonpath", []))
        if options.get("python"):
            paths.append(options["python"])
    source = entry.get("source", options.get("source"))
    if source and not entry.get("artifact"):
        paths.append(source)
    records = {}
    for item in paths:
        path = Path(item).expanduser().resolve()
        files = path.rglob("*") if path.is_dir() else (path,)
        for file in files:
            if "__pycache__" in file.parts or ".git" in file.parts or file.is_dir():
                continue
            if file.suffix in (".pyc", ".pyo"):
                continue
            try:
                stat = file.stat()
                records[str(file)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            except FileNotFoundError:
                records[str(file)] = {"missing": True}
    settings = target.settings.get("verification", {})
    reference = {}
    if entry.get("kind") == "native" and (options.get("kind") == "native" or options.get("factory") == "weft_bench.adapters.native:GGMLAdapter"):
        reference = {"native_reference_configuration": {key: target.settings.get(key) for key in
            ("target", "deployment", "toolchain", "threads", "env", "ggml", "references", "quantization_contracts")}}
    return {"entry": entry, "compiler_environment": settings.get("compiler_environment"),
            "observed_runtime": target.settings.get("_verification_runtime_identity"),
            "runtime": settings.get("runtime", {}), "files": records, **reference}

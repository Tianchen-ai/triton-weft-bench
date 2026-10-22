"""Run in the selected Triton environment; compile and export its native artifact."""

import json
import os
from pathlib import Path
import re
import sys


def clear_backend_failures(directory, prefix="compile-failure"):
    """Only current numbered diagnostic files belong to this capture namespace."""
    directory = Path(directory)
    if directory.is_dir():
        pattern = re.compile(re.escape(prefix) + r"-[0-9]+\.(json|mlir)$")
        for path in directory.iterdir():
            if path.is_file() and pattern.fullmatch(path.name):
                path.unlink()


def save_backend_failures(error, directory, prefix="compile-failure"):
    """Keep backend reports, including partial IR, across framework wrappers."""
    directory = Path(directory)
    clear_backend_failures(directory, prefix)
    pending, seen, seen_reports, records = [error], set(), set(), []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for child in (current.__cause__, current.__context__, getattr(current, "inner_exception", None)):
            if isinstance(child, BaseException):
                pending.append(child)
        report = getattr(current, "report", None)
        if not isinstance(report, dict) or id(report) in seen_reports:
            continue
        seen_reports.add(id(report))
        directory.mkdir(parents=True, exist_ok=True)
        saved = dict(report)
        partial = saved.pop("failed_ir", None)
        stem = prefix + "-" + str(len(records))
        if partial:
            path = directory / (stem + ".mlir")
            path.write_text(partial)
            saved["failed_ir"] = str(path.resolve())
        path = directory / (stem + ".json")
        path.write_text(json.dumps(saved, indent=2))
        records.append({"report": str(path.resolve()), "stage": saved.get("stage"),
                        "pass": saved.get("pass"), "category": saved.get("category"),
                        "failed_ir": saved.get("failed_ir")})
    return records


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(request["output"])
    cache = output.parent / "triton-cache"
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    os.environ["TRITON_DUMP_DIR"] = str(output.parent / "triton-dump")
    from triton import compile
    from triton.backends.compiler import CPUTarget
    from triton.compiler import ASTSource

    if __package__:
        from .triton_source import load_kernel
        from .cpu_target import compiler_target, check_jit_target
    else:
        from triton_source import load_kernel
        from cpu_target import compiler_target, check_jit_target
    source = request["source"]
    if request.get("source_module"):
        import importlib
        source = importlib.import_module(request["source_module"]).__file__
    kernel = load_kernel(source, request["symbol"])
    if request.get("target_native"):
        from triton.runtime.driver import driver
        target = driver.for_model("weft").get_current_target()
        check_jit_target(None, request["target"], vars(target))
    else:
        target = CPUTarget(**compiler_target(request["target"]))
    try:
        if request.get("program"):
            import importlib
            from triton import Config

            package = kernel.fn.__module__.rsplit(".", 1)[0]
            program = importlib.import_module(package + ".catalog").get(request["program"])
            if program.definition is not kernel:
                raise ValueError("selected Program does not own the requested source definition")
            program.signature(**request["constexprs"])
            configuration = Config(request["constexprs"], backend_options=request["backend_options"])
            compiled = program.compile(target=target, config=configuration, toolchain=request["toolchain"])
            (output.parent / "program-api.json").write_text(json.dumps({"program": program.name,
                "config_signature_compile": True, "source_parameters": request["constexprs"]}, indent=2))
        else:
            constants = {parameter.name: parameter.default for parameter in kernel.params
                         if parameter.is_constexpr and parameter.has_default}
            constants.update(request["constexprs"])
            compiled = compile(
                ASTSource(kernel, request["signature"], constants),
                target=target,
                options={**request["backend_options"], "toolchain": request["toolchain"]},
            )
    except Exception as error:
        save_backend_failures(error, output.parent)
        raise
    compiled.export(output)
    print(json.dumps({"engine": "triton", "artifact": str(output)}))


if __name__ == "__main__":
    main()

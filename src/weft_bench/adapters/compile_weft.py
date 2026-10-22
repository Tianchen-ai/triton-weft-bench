"""Runs in the configured Weft Python environment, never imported by GGML."""

import json
from pathlib import Path
import sys
import tempfile


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    root = Path(request["root"])
    sys.path[:0] = [str(root / "python"), str(root / "examples")]
    from weft.cli import load_definition
    from weft.runtime import Target, Toolchain, CompileOptions
    from weft.runtime.cache import ArtifactCache
    from weft.runtime.compiler import KernelSource

    options = CompileOptions(**request["options"])
    if request.get("selection"):
        from weft.runtime import TuningResult

        options = TuningResult.load(request["selection"]).winner.options
    if request.get("canonical"):
        captured = KernelSource.from_ir(
            Path(request["canonical"]).read_text(),
            toolchain=Toolchain(compiler=request["compiler"]),
        )
    else:
        source = Path(request["source"])
        if not source.is_absolute():
            source = root / source
        captured = KernelSource.capture(
            load_definition(str(source) + ":" + request["symbol"])
        )
    output = Path(request["output"])
    (output.parent / "canonical.mlir").write_text(captured.canonical)
    (output.parent / "effective-options.json").write_text(
        json.dumps(options.to_dict(), indent=2)
    )
    with tempfile.TemporaryDirectory(prefix=".compile-", dir=output.parent) as scratch:
        artifact = captured.build(
            target=Target(**request["target"]),
            options=options,
            native_library=False,
            toolchain=Toolchain(compiler=request["compiler"]),
            cache=ArtifactCache(Path(scratch)),
        )
        artifact.save(output)


if __name__ == "__main__":
    main()

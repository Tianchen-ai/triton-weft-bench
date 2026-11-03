"""Package already-compiled Triton CPU kernels through its public package API."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    producer = request["producer"]
    if producer == "triton":
        import triton
        from triton.integrations.llama_cpp import create_package
        load = triton.load
    elif producer == "tilelang":
        from weft_core.artifact import NativeArtifact
        from tilelang.weft.integrations.llama_cpp import create_package
        load = NativeArtifact.load
    else:
        raise ValueError(f"unknown native package producer {producer!r}")
    package = create_package(
        request["output"],
        {name: load(path) for name, path in request["artifacts"].items()},
        bindings=request["bindings"],
    )
    print(json.dumps({"producer": producer, "package": str(package.path), "bindings": package.bindings}))


if __name__ == "__main__":
    main()

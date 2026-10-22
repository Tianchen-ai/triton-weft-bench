"""Package already-compiled Triton CPU kernels through its public package API."""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    import triton
    from triton.integrations.llama_cpp import create_package

    package = create_package(
        request["output"],
        {name: triton.load(path) for name, path in request["artifacts"].items()},
        bindings=request["bindings"],
    )
    print(json.dumps({"producer": "triton", "package": str(package.path), "bindings": package.bindings}))


if __name__ == "__main__":
    main()

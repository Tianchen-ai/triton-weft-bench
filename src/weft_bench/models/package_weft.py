"""Run with the chosen public Weft Python runtime, on the build or target host."""

import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    if request.get("root"):
        sys.path.insert(0, str(Path(request["root"]) / "python"))
    from weft.runtime import Artifact, Package, Toolchain

    if request["action"] == "create":
        package = Package.create(
            request["output"],
            {name: Artifact(path) for name, path in request["artifacts"].items()},
            bindings=request["bindings"],
        )
    elif request["action"] == "link":
        tools = request["toolchain"]
        package = Package(request["source"]).link(
            request["output"],
            toolchain=Toolchain(
                cc=tools["cc"],
                cflags=tools.get("cflags", ()),
                ldflags=tools.get("ldflags", ()),
            ),
        )
    elif request["action"] == "install":
        from weft.integrations.llama_cpp import install_source

        print(json.dumps(install_source(request["source"])))
        return
    else:
        raise ValueError("unknown package operation")
    print(
        json.dumps({"package": str(package.path), "bindings": dict(package.bindings)})
    )


if __name__ == "__main__":
    main()

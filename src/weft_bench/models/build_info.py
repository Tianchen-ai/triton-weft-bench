"""Record the actual CPU compilation policy before timing a configured model."""

import json
from pathlib import Path
import re
import shlex


def inspect_build(executor, config):
    build = config.get("build_directory")
    if not build:
        return {"status": "unavailable", "reason": "binary has no configured build_directory"}
    database = config.get("compile_commands", str(Path(build) / "compile_commands.json"))
    entries = json.loads(executor.command(["cat", database], stage="model_build_inspection").stdout)
    units, policies, compilers = [], set(), {}
    for entry in entries:
        source = Path(entry["file"])
        backend = next((source.parts[i + 2] for i in range(len(source.parts) - 2)
                        if source.parts[i:i + 2] == ("ggml", "src")
                        and source.parts[i + 2].startswith("ggml-")), None)
        if backend is None or source.suffix not in (".c", ".cc", ".cpp", ".cxx"):
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        if any(arg.startswith("@") for arg in arguments):
            raise ValueError(f"model build inspection needs expanded compiler arguments: {source}")
        optimizations = [arg for arg in arguments if re.fullmatch(r"-O(?:[0-3sgz]|fast)?", arg)]
        optimization = optimizations[-1] if optimizations else "-O0"
        if optimization in ("-O0", "-Og"):
            raise ValueError(f"unoptimized model CPU source {source}: effective {optimization}; "
                             "configure matching optimized normal/candidate builds before timing")
        compiler = arguments[0]
        if compiler not in compilers:
            compilers[compiler] = executor.command([compiler, "--version"], stage="model_build_inspection").stdout.splitlines()[0]
        flags = [optimization]
        for index, arg in enumerate(arguments[1:], 1):
            if arg.startswith(("-march=", "-mcpu=", "-mtune=", "-mabi=", "--target=", "-target=",
                               "-ffp-", "-fno-fp-", "-ffast-math", "-fno-fast-math", "-funsafe-math",
                               "-fno-unsafe-math", "-fassociative-math", "-fno-associative-math")):
                flags.append(arg)
            elif arg in ("--target", "-target"):
                flags.append(arg + "=" + arguments[index + 1])
            elif arg == "-fno-integrated-as":
                flags.append(arg)
            elif arg == "-Xclang" and arguments[index:index + 3] == ["-Xclang", "-target-feature", "-Xclang"]:
                flags.append("target-feature=" + arguments[index + 3])
        policy = {"compiler": compilers[compiler], "flags": flags}
        policies.add(json.dumps(policy, sort_keys=True))
        units.append({"file": str(source), "backend": backend, "directory": entry["directory"], "arguments": arguments})
    if not units:
        raise ValueError(f"model compile database has no GGML backend translation units: {database}")
    return {"status": "observed", "database": database, "units": units,
            "policy": [json.loads(value) for value in sorted(policies)]}

"""Observe the configured execution context using the native worker's probe."""

import json
from pathlib import Path

from .adapters.native import NATIVE, flags_for
from .execution import Executor


def inspect_environment(target, output):
    directory = Path(output).resolve()
    executor = Executor(target, directory)
    executor.upload_many((NATIVE / name, name) for name in ("timer.h", "worker.h", "environment.h"))
    source = executor.write("environment.cpp", '#include "worker.h"\nint main(){wb::describe_environment();}\n')
    tools, flags = flags_for(target)
    executable = executor.workspace + "/environment"
    executor.command([*tools["cxx"], *flags, "-std=c++17", source,
                      *tools.get("ldflags", []), "-o", executable], stage="native_compile")
    completed = executor.command([executable], affinity=True)
    rows = [json.loads(line[len("WB_ENVIRONMENT "):]) for line in completed.stdout.splitlines()
            if line.startswith("WB_ENVIRONMENT ")]
    if len(rows) != 1:
        raise ValueError("native environment probe did not return one observation")
    result = {"target": target.name, "observed": rows[0],
              "requested": {"cpus": executor.cpus, "numa": executor.numa, "env": executor.env},
              "native_build": target.settings["toolchain"],
              "amx_execution": "not_tested", "commands": executor.commands}
    (directory / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ..operators.implementations import resolve_binding

NATIVE = Path(__file__).resolve().parents[1] / "native"


def compile_artifact(case, target, options, executor):
    root = options.get("root", target.settings.get("weft_root"))
    compiler = options.get("compiler", target.settings.get("compiler"))
    if not root or not compiler:
        raise ValueError(
            "Weft adapter needs weft_root and compiler in external configuration"
        )
    binding = resolve_binding(case, target, options)
    physical = binding["options"]
    output = executor.local / "artifact"
    request = {
        "root": root,
        "source": binding["source"],
        "symbol": binding["symbol"],
        "options": physical,
        "target": target.settings["target"],
        "compiler": compiler,
        "output": str(output.resolve()),
        "arguments": binding.get("arguments", {}),
    }
    if options.get("selection"):
        request["selection"] = options["selection"]
    if options.get("canonical"):
        request["canonical"] = options["canonical"]
    request_file = executor.local / "compile.json"
    request_file.write_text(json.dumps(request, indent=2))
    command = [
        options.get("python", target.settings.get("weft_python", sys.executable)),
        str(Path(__file__).with_name("compile_weft.py")),
        str(request_file.resolve()),
    ]
    executor.commands.append(command)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=target.settings.get("timeout", 1800),
    )
    (executor.local / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        from ..execution import ExecutionError

        raise ExecutionError("compile", command, result)
    return output


def flags_for(target):
    settings = target.settings
    tools = settings["toolchain"]
    flags = ["-O3", "-ffp-contract=fast", *tools.get("cflags", [])]
    isa = settings.get("target", {})
    if isa.get("march"):
        flags.append("-march=" + isa["march"])
        # Preserve legacy RVV configs, without applying a RISC-V ABI to x86.
        if isa.get("architecture", "").startswith("riscv") or isa["march"].startswith(("rv32", "rv64")):
            flags.append("-mabi=" + isa.get("abi", "lp64d"))
            if isa.get("vlen_bits"):
                width = isa["vlen_bits"]
                if type(width) is not int or width < 128 or width & (width - 1):
                    raise ValueError("RISC-V vlen_bits must be a power of two of at least 128")
                flags.append("-DWB_EXPECTED_VLEN_BITS=" + str(width))
        elif isa.get("abi"):
            flags.append("-mabi=" + isa["abi"])
    return tools, flags


def ggml_flags(target):
    ggml = target.settings.get("ggml")
    if not ggml:
        raise ValueError("case requires configured GGML source and libraries")
    includes = [
        "-I" + str(Path(ggml["source"]) / x)
        for x in ("ggml/include", "ggml/src", "ggml/src/ggml-cpu")
    ]
    libs = [
        "-L" + ggml["libraries"],
        "-Wl,-rpath," + ggml["libraries"],
        "-lggml",
        "-lggml-cpu",
        "-lggml-base",
    ]
    return includes, libs

"""Small, untimed model-logit comparison using the normal and selected llama libraries."""
from array import array
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import sys

from ..adapters.native import NATIVE, flags_for
from ..execution import Executor
from .observe import counter_capabilities
from .environment import BACKEND_ENVIRONMENT, backend_policy


def same_reference_contract(saved, current):
    """Accept a relocated local model/binary only when both names are the same file."""
    observed = dict(current)
    for key in ("model", "normal_binary"):
        before, after = saved.get(key), current.get(key)
        if before == after:
            continue
        if not isinstance(before, str) or not isinstance(after, str):
            return False
        old, new = Path(before), Path(after)
        if not (old.is_absolute() and new.is_absolute() and old.is_file() and new.is_file() and old.samefile(new)):
            return False
        observed[key] = before
    return saved == observed


def _run(name, target, implementation, parameters, directory, token_reference=None):
    settings = target.settings["llama"]
    config = settings["implementations"][implementation]
    deployment = {**target.settings.get("deployment", {}),
                  "cpus": settings.get("cpus", target.settings.get("deployment", {}).get("cpus", []))}
    executor = Executor(replace(target, settings={**target.settings, "deployment": deployment}), directory)
    source = config.get("source", target.settings.get("ggml", {}).get("source"))
    if not source:
        raise ValueError("model validation needs a configured llama.cpp source/include directory")
    library = str(Path(config["binary"]).parent)
    helper = executor.upload(NATIVE / "llama_validation.cpp", "llama_validation.cpp")
    tools, flags = flags_for(target)
    counters = None
    runtime_link = []
    build_environment = {**settings.get("env", {}), **config.get("env", {})}
    if config.get("package") and config.get("weft_include"):
        runtime = target.settings["triton_runtime"]
        flags = ["-DWB_WEFT_CHECK", "-I" + config["weft_include"],
                 "-I" + runtime["include"], *flags]
        runtime_link = [runtime["library"], "-Wl,-rpath," + str(Path(runtime["library"]).parent)]
        header = str(Path(config["weft_include"]) / "weft/ggml.h")
        counters = counter_capabilities(executor.command(["cat", header], stage="prepare").stdout)
        if not counters["per_type"] or not counters["ime1"]:
            raise ValueError(f"integration SDK lacks the required row/IME counter API: {header}")
        if counters["amx"]:
            flags += ["-DWB_WEFT_AMX_COUNTER"]
        if counters["ime2"]:
            flags += ["-DWB_WEFT_IME2_COUNTER"]
    executable = executor.workspace + "/validate-model"
    command = [*tools["cxx"], *flags, "-std=c++17", "-I" + source + "/include",
               "-I" + source + "/ggml/include", helper, "-L" + library, "-Wl,-rpath," + library,
               "-lllama", "-lggml", "-lggml-cpu", "-lggml-base", *runtime_link,
               *tools.get("ldflags", []), "-o", executable]
    done = executor.command(command, env=build_environment, stage="model_validation_build", check=False)
    (directory / "build.log").write_text(done.stdout + done.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
    if done.returncode:
        raise RuntimeError(f"model_validation_build: exit {done.returncode}: " + done.stderr[-6000:])
    tokens = executor.upload(token_reference, "reference.tokens.bin") if token_reference else "-"
    prefix = executor.workspace + "/output"
    environment = {**backend_policy(executor.env),
                   **settings.get("env", {}), **config.get("env", {})}
    if config.get("package"):
        environment["WEFT_GGML_PACKAGE"] = config["package"]
    check_dots = bool(settings.get("validation", {}).get("check_dots", False) and config.get("package"))
    if check_dots:
        if environment.get("LD_PRELOAD") or not config.get("weft_include"):
            raise ValueError("model dot checking needs weft_include and an unused LD_PRELOAD setting")
        interposer = executor.upload(NATIVE / "llama_dot_check.cpp", "llama_dot_check.cpp")
        check_library = executor.workspace + "/libcheck-dots.so"
        diagnostic = settings["validation"].get("dot_check_tolerance", {"atol": 1e-4, "rtol": 2e-3})
        if not isinstance(diagnostic, dict) or set(diagnostic) != {"atol", "rtol"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in diagnostic.values()
        ):
            raise ValueError("dot_check_tolerance needs finite nonnegative atol and rtol")
        if diagnostic["atol"] > 1e-4 or diagnostic["rtol"] > 2e-3:
            raise ValueError("diagnostic dot thresholds may only tighten the operator tolerance")
        dot_type = settings["validation"].get("dot_check_type")
        if dot_type is not None and (not isinstance(dot_type, str) or not dot_type):
            raise ValueError("dot_check_type must be a nonempty GGML type name")
        executor.command([*tools["cxx"], *flags, "-std=c++17", "-shared", "-fPIC",
                          "-DWB_DOT_ATOL=" + str(float(diagnostic["atol"])),
                          "-DWB_DOT_RTOL=" + str(float(diagnostic["rtol"])),
                          "-I" + source + "/ggml/include", "-I" + source + "/ggml/src",
                          interposer, "-L" + library, "-Wl,-rpath," + library,
                          "-lggml-cpu", "-lggml-base", *runtime_link, "-ldl", *tools.get("ldflags", []),
                          "-o", check_library], env=build_environment, stage="model_dot_check_build")
        environment.update(LD_PRELOAD=check_library, WB_DOT_CAPTURE=executor.workspace + "/dot")
        if dot_type is not None:
            environment["WB_DOT_CHECK_TYPE"] = dot_type
    command = ["env", "-u", "WEFT_GGML_PACKAGE", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH",
               *[arg for key in BACKEND_ENVIRONMENT for arg in ("-u", key)],
               *[str(k) + "=" + str(v) for k, v in environment.items()],
               executable, settings["models"][name], prefix, tokens, str(parameters["threads"]),
               str(parameters["prompt_tokens"]), str(parameters["decode_steps"]),
               str(parameters["gpu_layers"])]
    done = executor.command(command, affinity=True, stage="model_validation_run", check=False)
    (directory / "stdout.log").write_text(done.stdout)
    (directory / "stderr.log").write_text(done.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
    if done.returncode:
        if check_dots and done.returncode == 86:
            for suffix in ("json", "W.bin", "X.bin"):
                executor.download(executor.workspace + "/dot." + suffix, directory / ("dot." + suffix))
            captured = json.loads((directory / "dot.json").read_text())
            inputs = {"shape": {"rows": 1, "K": captured["K"]},
                      "buffers": {name: str((directory / ("dot." + name + ".bin")).resolve())
                                  for name in ("W", "X")}}
            (directory / "dot-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
            raise RuntimeError(f"model dot mismatch; reproduce {captured['type'].lower()}_dot "
                               f"with --inputs {directory / 'dot-inputs.json'}")
        raise RuntimeError(f"model_validation_run: exit {done.returncode}: " + done.stderr[-6000:])
    records = [json.loads(line[len("WB_MODEL_CHECK "):]) for line in done.stdout.splitlines() if line.startswith("WB_MODEL_CHECK ")]
    if len(records) != 1:
        raise ValueError("missing model validation output")
    if check_dots:
        counts = [int(line.split()[1]) for line in done.stderr.splitlines()
                  if line.startswith("WB_DOT_CHECK_CALLS ")]
        if len(counts) != 1 or counts[0] <= 0:
            raise ValueError("model dot checking did not observe any replaced native rows")
        records[0]["checked_native_dot_calls"] = counts[0]
        records[0]["dot_check_scope"] = {"type": dot_type or "all-replaced-types",
                                          "capture_tolerance": diagnostic}
    for suffix in ("logits.bin", "tokens.bin"):
        executor.download(prefix + "." + suffix, directory / suffix)
    records[0]["replacement_snapshots"] = [json.loads(line[len("WB_MODEL_HITS "):]) for line in done.stdout.splitlines() if line.startswith("WB_MODEL_HITS ")]
    records[0]["execution"] = {"cpus": executor.cpus, "numa": executor.numa,
                               "requested_environment": {**executor.env, **environment}}
    records[0]["native_build"] = {"toolchain": tools, "flags": flags}
    records[0]["counter_capabilities"] = counters
    return records[0]


def _logits(path):
    with Path(path).open("rb") as stream:
        rows, columns = struct.unpack("<II", stream.read(8))
        values = array("f")
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != rows * columns:
        raise ValueError("model logit output length mismatch")
    return rows, columns, values


def _reference_contract(name, target):
    settings = target.settings["llama"]
    selected = settings.get("validation", {})
    parameters = {"prompt_tokens": selected.get("prompt_tokens", 16),
                  "decode_steps": selected.get("decode_steps", 2),
                  "threads": settings.get("threads", 1)}
    if any(type(v) is not int or v < 1 for v in parameters.values()):
        raise ValueError("model reference prompt/decode/thread values must be positive integers")
    # Match llama-bench's placement selection, including non-GPU accelerators
    # exposed through its layer-offload interface. A forced CPU-only numerical
    # check would otherwise validate a different dispatch path from timing.
    gpu_layers = -1
    arguments = settings.get("args", [])
    for index, argument in enumerate(arguments):
        if argument in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
            if index + 1 == len(arguments):
                raise ValueError("model layer-offload option needs one integer")
            gpu_layers = int(arguments[index + 1])
        elif argument.startswith(("--n-gpu-layers=", "--gpu-layers=")):
            gpu_layers = int(argument.split("=", 1)[1])
    if gpu_layers < -1:
        raise ValueError("model layer-offload count must be -1 or nonnegative")
    parameters["gpu_layers"] = gpu_layers
    contract = {"model": settings["models"][name], "target": target.name, "parameters": parameters,
                "normal_binary": settings["implementations"]["normal"]["binary"],
                "protocol": "fixed-prompt-teacher-forced-logits-v1"}
    policy = backend_policy({**target.settings.get("env", {}), **settings.get("env", {}),
                             **settings["implementations"]["normal"].get("env", {})})
    if policy:
        contract["backend_environment"] = policy
    return contract, parameters


def read_matching_reference(name, target, directory):
    """Check a retained reference before starting any target execution."""
    record = Path(directory) / "reference.json"
    if not record.exists():
        return None
    saved = json.loads(record.read_text())
    current, _ = _reference_contract(name, target)
    if not same_reference_contract(saved["contract"], current):
        differences = {key: {"reference": saved["contract"].get(key), "requested": current.get(key)}
                       for key in saved["contract"].keys() | current.keys()
                       if saved["contract"].get(key) != current.get(key)}
        raise ValueError("model numerical reference input/execution differs: " +
                         json.dumps(differences, sort_keys=True) +
                         "; select the matching validation protocol or reference")
    return saved


def capture_model_reference(name, target, *, output):
    """Capture or reuse normal logits, checking their structure without self-comparison."""
    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    contract, parameters = _reference_contract(name, target)
    report = {"contract": contract, "reference": str(directory),
              "scope": "full-vocabulary last-prefill and teacher-forced decode logits; untimed reference capture"}
    try:
        record = directory / "reference.json"
        saved = read_matching_reference(name, target, directory)
        reused = saved is not None
        if reused:
            details = saved["details"]
        else:
            details = _run(name, target, "normal", parameters, directory)
        rows, columns, values = _logits(directory / "logits.bin")
        if rows != parameters["decode_steps"] + 1 or columns < 1 or (rows, columns) != (details["rows"], details["vocabulary"]):
            raise ValueError("model reference logit shape differs from its execution")
        if any(not math.isfinite(v) for v in values):
            raise ValueError("model reference logits contain nonfinite values")
        tokens = (directory / "tokens.bin").read_bytes()
        prompt, count = struct.unpack("<II", tokens[:8])
        if prompt != parameters["prompt_tokens"] or count != prompt + parameters["decode_steps"] or len(tokens) != 8 + 4 * count:
            raise ValueError("model reference token dimensions differ")
        ids = struct.unpack("<" + "i" * count, tokens[8:])
        if any(token < 0 or token >= columns for token in ids):
            raise ValueError("model reference token outside vocabulary")
        if not reused:
            record.write_text(json.dumps({"contract": contract, "details": details}, indent=2) + "\n")
        report.update(status="ok", correctness="reference-logits-captured", reference_reused=reused,
                      rows=rows, vocabulary=columns, finite=True, self_comparison=False,
                      logits=str(directory / "logits.bin"), tokens=str(directory / "tokens.bin"),
                      reference_execution=details)
    except Exception as error:
        report.update(status="validation_failed", correctness="not_checked", error=str(error))
    (directory / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_model_logits(reference, candidate, tolerance):
    """Compare saved native outputs without running or changing either model."""
    if any(not math.isfinite(tolerance[k]) or tolerance[k] < 0 for k in ("rtol", "atol")):
        raise ValueError("model validation tolerances must be finite and nonnegative")
    before_rows, before_columns, before = _logits(reference)
    rows, columns, after = _logits(candidate)
    if (rows, columns) != (before_rows, before_columns) or rows < 1 or columns < 1:
        raise ValueError("model logit shapes differ or are empty")
    metrics = []
    for row in range(rows):
        a, b = before[row * columns:(row + 1) * columns], after[row * columns:(row + 1) * columns]
        if any(not math.isfinite(v) for v in (*a, *b)):
            raise ValueError("model logits contain nonfinite values")
        errors = [abs(x - y) for x, y in zip(a, b)]
        outside = sum(error > tolerance["atol"] + tolerance["rtol"] * abs(x) for error, x in zip(errors, a))
        expected_token = max(range(columns), key=a.__getitem__)
        actual_token = max(range(columns), key=b.__getitem__)
        metrics.append({"stage": "prefill" if row == 0 else "decode", "step": row,
                        "max_abs_error": max(errors), "rms_error": math.sqrt(sum(e * e for e in errors) / columns),
                        "outside_tolerance": outside, "reference_top1": expected_token,
                        "candidate_top1": actual_token, "top1_matches": expected_token == actual_token})
    ok = all(m["outside_tolerance"] == 0 and m["top1_matches"] for m in metrics)
    return {"status": "ok" if ok else "numeric_mismatch",
            "correctness": "sampled-logits-within-tolerance" if ok else "failed",
            "vocabulary": columns, "metrics": metrics}


def validate_model(name, target, implementation="weft", *, output, reference_output=None):
    if implementation == "normal":
        return capture_model_reference(name, target, output=reference_output or output)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_output) if reference_output else output / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    settings = target.settings["llama"]
    selected = settings.get("validation", {})
    contract, parameters = _reference_contract(name, target)
    tolerance = {"rtol": selected.get("rtol", 1e-3), "atol": selected.get("atol", 1e-3)}
    if any(not math.isfinite(v) or v < 0 for v in tolerance.values()):
        raise ValueError("model validation tolerances must be finite and nonnegative")
    report = {"contract": contract, "tolerance": tolerance, "reference": str(reference.resolve()),
              "candidate_package": settings["implementations"][implementation].get("package"),
              "scope": "full-vocabulary last-prefill logits and teacher-forced decode logits; separate from phase timing"}
    try:
        captured = capture_model_reference(name, target, output=reference)
        if captured["status"] != "ok":
            raise ValueError(captured["error"])
        report["reference_reused"] = captured["reference_reused"]
        candidate = _run(name, target, implementation, parameters, output, reference / "tokens.bin")
        report["candidate_execution"] = candidate
        expected_hit = settings["implementations"][implementation].get("validation_expect_replacement", True)
        if type(expected_hit) is not bool:
            raise ValueError("validation_expect_replacement must be boolean")
        report["replacement_expected"] = expected_hit
        report["replacement_matches"] = (not report["candidate_package"] or
                                         candidate["replacement_observed"] == expected_hit)
        report.update(compare_model_logits(reference / "logits.bin", output / "logits.bin", tolerance))
        if report["status"] == "ok" and not report["replacement_matches"]:
            report.update(status="replacement_mismatch",
                          error="model logits match, but replacement observation differs from its explicit expectation")
    except Exception as error:
        report.update(status="validation_failed", correctness="not_checked", error=str(error))
        if (output / "dot-inputs.json").is_file():
            report["dot_inputs"] = str((output / "dot-inputs.json").resolve())
            report["dot_mismatch"] = json.loads((output / "dot.json").read_text())
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report


def attach_validation(rows, report, implementation="weft"):
    for row in rows:
        if row.implementation != implementation:
            continue
        row.details["model_validation"] = report
        row.correctness = report["correctness"]
        if report["status"] != "ok" and row.status == "ok":
            row.status = ("correctness_failed" if report["status"] == "numeric_mismatch" else
                          "replacement_mismatch" if report["status"] == "replacement_mismatch" else
                          "validation_failed")
            row.speedup = None
            row.error = report.get("error", "sampled model logits differ from the normal implementation")
    return rows

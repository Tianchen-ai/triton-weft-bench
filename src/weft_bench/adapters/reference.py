"""Thin bridges to the selected GGML build; reference math stays upstream."""

import json
from pathlib import Path
import shlex

from .dense import dense_defines


def prepare_ime2_reference(target, executor):
    """Link the selected upstream IME2 translation unit without replacing its math."""
    settings = target.settings.get("ime2_reference", {})
    database = settings.get("compile_commands")
    if not database:
        raise ValueError("IME2 reference requires its vendor build compile_commands")
    commands = json.loads(executor.command(["cat", database], stage="prepare").stdout)
    entries = [entry for entry in commands if entry["file"].endswith("/ime2_kernels.cpp")]
    if len(entries) != 1:
        raise ValueError("IME2 reference requires exactly one upstream translation unit")
    entry = entries[0]
    original = entry.get("arguments") or shlex.split(entry["command"])
    levels = [x for x in original if x in ("-O0", "-O1", "-O2", "-O3", "-Os", "-Og", "-Ofast")]
    if not levels or levels[-1] not in ("-O2", "-O3", "-Ofast"):
        raise ValueError("IME2 reference translation unit is not optimized")
    flags, skip = [], False
    for value in original[1:]:
        if skip:
            skip = False
        elif value in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif value not in ("-c", "-MD", "-MMD", entry["file"]):
            flags.append(value)
    obj = executor.workspace + "/ime2-reference.o"
    executor.command([original[0], *flags, "-ffunction-sections", "-fdata-sections",
                      "-c", entry["file"], "-o", obj], cwd=entry["directory"], stage="native_compile")
    executor.write("ime2-reference-build.json", json.dumps(entry, indent=2))
    return [obj, "-Wl,--gc-sections"], ["-I" + str(Path(entry["file"]).parent)]


def inspect_ggml_compilation(target, executor, *, require_optimized):
    """Check actual CPU translation units, not CMake's build-type label."""
    settings = target.settings["ggml"]
    database = settings.get("compile_commands", str(Path(settings["libraries"]).parent / "compile_commands.json"))
    commands = json.loads(executor.command(["cat", database], stage="prepare").stdout)
    root = Path(settings["source"]) / "ggml/src/ggml-cpu"
    units, records = [], []
    for entry in commands:
        source = Path(entry["file"])
        if not source.is_absolute():
            source = Path(entry["directory"]) / source
        if not source.is_relative_to(root) or source.suffix not in (".c", ".cpp", ".cc"):
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        levels = [flag for flag in arguments if flag in ("-O", "-O0", "-O1", "-O2", "-O3", "-Og", "-Os", "-Oz", "-Ofast")]
        optimization = levels[-1] if levels else "none"
        units.append({"source": str(source), "optimization": optimization})
        records.append(entry)
    if not units:
        raise ValueError("GGML reference needs CPU translation units in its compile_commands.json")
    executor.write("reference-compile-commands.json", json.dumps(records, indent=2))
    unoptimized = [unit for unit in units if unit["optimization"] not in ("-O2", "-O3", "-Ofast")]
    if require_optimized and unoptimized:
        unit = unoptimized[0]
        raise ValueError("GGML performance reference has an unoptimized CPU translation unit: "
                         f"{unit['source']} ({unit['optimization']}). Rebuild the library with -O2/-O3; "
                         "a Release label alone does not establish optimized compilation.")
    return {"database": database, "cpu_units": units, "optimized": not unoptimized,
            "evidence": str(executor.local / "reference-compile-commands.json")}


def prepare_mixed_projection_reference(target, executor):
    """Identify the upstream mixed-precision path; never substitute its math."""
    settings = target.settings["ggml"]
    root = Path(settings["source"]) / "ggml/src/ggml-cpu"
    source = str(root / "amx/mmq.cpp")
    database = settings.get("compile_commands", str(Path(settings["libraries"]).parent / "compile_commands.json"))
    commands = json.loads(executor.command(["cat", database], stage="prepare").stdout)
    builds = [entry for entry in commands if entry["file"].endswith("/ggml-cpu/amx/mmq.cpp")]
    if len(builds) != 1 or builds[0]["file"] != source:
        raise NotImplementedError("mixed F16/F32 reference requires the selected GGML build's AMX F16 source evidence")
    text = executor.command(["cat", source], stage="prepare").stdout
    traits_source = str(root / "amx/amx.cpp")
    traits = executor.command(["cat", traits_source], stage="prepare").stdout
    try:
        kernel_start = text.index("struct tinygemm_kernel_avx<float, ggml_fp16_t, float,")
        launch_start = text.index("#define LAUNCH_TINYGEMM_KERNEL_AVX", kernel_start)
        kernel = text[kernel_start:launch_start]
        launch = text[launch_start:text.index("// re-organize", launch_start)]
        dispatch_start = text.index("void ggml_backend_amx_mul_mat(")
        dispatch = text[dispatch_start:text.index("// pointer to work space", dispatch_start)]
        traits_start = traits.index("bool compute_forward(")
        route = traits[traits_start:traits.index("return false;", traits_start)]
    except ValueError as error:
        raise NotImplementedError("unrecognized GGML F16/F32 dispatch; inspect its activation precision before using this reference") from error
    recognized = (
        "const float * RESTRICT A" in kernel
        and "_mm512_loadu_ps(A + row * K + k)" in kernel
        and "_mm512_cvtph_ps" in kernel and "_mm512_fmadd_ps" in kernel
        and "(const float *)src1->data" in launch
        and "is_floating_type = TYPE == GGML_TYPE_F16" in dispatch
        and "LAUNCH_TINYGEMM_KERNEL_AVX(" in dispatch and "return;" in dispatch
        and "ggml_backend_amx_mul_mat(params, op);" in route
        and not any(token in kernel + launch + dispatch for token in ("from_float(", "_cvtps_ph", "FP32_TO_FP16"))
    )
    if not recognized:
        raise NotImplementedError("GGML source does not establish F16 weights with preserved F32 activations")
    return {
        "contract": "stored-f16-weights-original-f32-activation",
        "path": "ggml_backend_amx_mul_mat/f16/tinygemm_kernel_avx<float,f16,float>",
        "source": source,
        "kernel_line": text[:kernel_start].count("\n") + 1,
        "dispatch_line": text[:dispatch_start].count("\n") + 1,
        "traits_source": traits_source,
        "build": builds[0],
        "selection_check": "worker must select AMX weight buffer and resolve its F16/F32 compute entry; numerical check remains required",
    }


def prepare_dense_reference(case, target, executor):
    from .native import NATIVE

    executor.upload(NATIVE / "dense_reference.h", "dense_reference.h")
    objects = []
    if case.name == "rope_split":
        source = str(
            Path(target.settings["ggml"]["source"]) / "ggml/src/ggml-cpu/ops.cpp"
        )
        database = target.settings["ggml"].get(
            "compile_commands",
            str(
                Path(target.settings["ggml"]["libraries"]).parent
                / "compile_commands.json"
            ),
        )
        commands = json.loads(
            executor.command(["cat", database], stage="prepare").stdout
        )
        entries = [x for x in commands if x["file"].endswith("/ggml-cpu/ops.cpp")]
        if len(entries) != 1:
            raise ValueError("reference needs one GGML ops.cpp native compile command")
        entry = entries[0]
        original = entry.get("arguments") or shlex.split(entry["command"])
        flags = []
        skip = False
        for value in original[1:]:
            if skip:
                skip = False
                continue
            if value in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
            elif value not in ("-c", "-MD", "-MMD", entry["file"]):
                flags.append(value)
        code = "#include " + json.dumps(source) + "\n" + """
extern "C" void wb_ggml_rotate(size_t tokens, size_t heads, size_t pairs,
    const float *cache, const float *x, float *y) {
  for(size_t t=0;t<tokens;++t) for(size_t h=0;h<heads;++h)
    rotate_pairs<float>(pairs*2, pairs, cache+t*pairs*2,
                       x+(t*heads+h)*pairs*2, y+(t*heads+h)*pairs*2);
}
"""
        bridge = executor.write("rope_reference.cpp", code)
        obj = executor.workspace + "/rope_reference.o"
        executor.command(
            [
                original[0],
                *flags,
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
                bridge,
                "-o",
                obj,
            ],
            cwd=entry["directory"],
            stage="native_compile",
        )
        executor.write("reference-build.json", json.dumps(entry, indent=2))
        objects += [obj, "-Wl,--gc-sections"]
    return objects, dense_defines(case)

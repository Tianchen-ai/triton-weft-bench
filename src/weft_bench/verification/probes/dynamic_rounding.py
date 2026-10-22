"""Exercise actual fenv modes using one compiled kernel and literal integer oracles."""
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np


VALUES = (-129, -128.5, -127.5, -2.5, -1.5, -0.5, -0.25, 0,
          0.25, 0.5, 1.5, 2.5, 126.5, 127.5, 128, 129)
EXPECTED = {
    "FE_TONEAREST": (-128, -128, -128, -2, -2, 0, 0, 0, 0, 0, 2, 2, 126, 127, 127, 127),
    "FE_DOWNWARD": (-128, -128, -128, -3, -2, -1, -1, 0, 0, 0, 1, 2, 126, 127, 127, 127),
    "FE_UPWARD": (-128, -128, -127, -2, -1, 0, 0, 0, 1, 1, 2, 3, 127, 127, 127, 127),
    "FE_TOWARDZERO": (-128, -128, -127, -2, -1, 0, 0, 0, 0, 0, 1, 2, 126, 127, 127, 127),
}


def fenv_bridge(root, toolchain):
    """Build only the platform fenv shim; Triton builds the candidate separately."""
    compiler = toolchain["cc"]
    command = list(compiler) if isinstance(compiler, list) else [compiler]
    if toolchain.get("target"):
        command += ["--target=" + toolchain["target"]]
    if toolchain.get("sysroot"):
        command += ["--sysroot=" + toolchain["sysroot"]]
    source = Path(__file__).with_name("fenv_bridge.c")
    library = root / "fenv_bridge.so"
    command += [*toolchain.get("cflags", []), "-std=c11", "-O2", "-shared", "-fPIC",
                str(source), "-o", str(library), *toolchain.get("ldflags", []), "-lm"]
    (root / "fenv-build.json").write_text(json.dumps({"command": command, "source": str(source)}, indent=2))
    built = subprocess.run(command, capture_output=True, text=True)
    (root / "fenv-build.log").write_text(built.stdout + built.stderr)
    built.check_returncode()
    bridge = ctypes.CDLL(str(library))
    for name in ("wb_fe_mode_constant", "wb_fe_set"):
        function = getattr(bridge, name)
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_int
    for name in ("wb_fe_mode", "wb_fe_flags"):
        function = getattr(bridge, name)
        function.argtypes = []
        function.restype = ctypes.c_int
    bridge.wb_fe_save.argtypes = []
    bridge.wb_fe_save.restype = ctypes.c_void_p
    bridge.wb_fe_restore.argtypes = [ctypes.c_void_p]
    bridge.wb_fe_restore.restype = ctypes.c_int
    return bridge


def run(root, options, request):
    import triton
    from triton.backends.compiler import CPUTarget
    from triton.compiler import ASTSource
    from cpu_target import compiler_target
    from dynamic_rounding_kernels import dynamic_rounding
    from helper_native import capture

    bridge = fenv_bridge(root, request["toolchain"])
    constants = {name: bridge.wb_fe_mode_constant(index) for index, name in enumerate(EXPECTED)}
    if any(value == -1 for value in constants.values()) or len(set(constants.values())) != 4:
        raise ValueError("target fenv.h does not provide four distinct rounding modes")
    original_mode, original_flags = bridge.wb_fe_mode(), bridge.wb_fe_flags()
    if original_mode not in constants.values():
        raise ValueError("target reported an unknown initial rounding mode")
    saved = bridge.wb_fe_save()
    if not saved:
        raise RuntimeError("could not save target floating-point environment")
    environment = {"mode_constants_from": "target fenv.h", "constants": constants,
                   "original_mode": original_mode, "original_exception_flags": original_flags,
                   "modes": [], "restored": False}
    snapshots = {}
    snapshot_path = root / "dynamic-rounding.npz"
    try:
        x32 = np.asarray(VALUES, dtype=np.float32)
        x64 = np.asarray(VALUES, dtype=np.float64)
        original_x32, original_x64 = x32.copy(), x64.copy()
        storage32 = np.full(len(VALUES) + 4, -101, np.int8)
        storage64 = np.full(len(VALUES) + 4, -101, np.int8)
        y32, y64 = storage32[2:-2], storage64[2:-2]
        snapshots.update(input_f32=original_x32, input_f64=original_x64)
        for name, expected in EXPECTED.items():
            snapshots[name + "_expected"] = np.asarray(expected, dtype=np.int8)
        np.savez(snapshot_path, **snapshots)
        target = CPUTarget(**compiler_target(request["compile_target"]))
        compiled = triton.compile(ASTSource(dynamic_rounding, {}, {}), target=target, options=options)
        compiled.export(str(root / "dynamic-rounding-artifact"))
        evidence = capture(compiled, root, "dynamic-rounding")
        canonical = compiled.asm["weftir"]
        assert canonical.count('rounding = "dynamic"') == 2, "expected two dynamic narrow operations"
        for name, mode in constants.items():
            storage32.fill(-101)
            storage64.fill(-101)
            if bridge.wb_fe_set(mode) != 0 or bridge.wb_fe_mode() != mode:
                raise RuntimeError("could not set " + name)
            observed = {"name": name, "requested": mode, "before": bridge.wb_fe_mode()}
            environment["modes"].append(observed)
            # CompiledKernel dispatch cannot specialize or recompile for a new mode.
            compiled[(1,)](x32, x64, y32, y64)
            observed["after"] = bridge.wb_fe_mode()
            snapshots.update({name + "_f32_storage": storage32.copy(),
                              name + "_f64_storage": storage64.copy(),
                              "actual_input_f32": x32.copy(), "actual_input_f64": x64.copy()})
            np.savez(snapshot_path, **snapshots)
            assert observed["after"] == mode, "kernel changed the active rounding mode"
            expected = snapshots[name + "_expected"]
            np.testing.assert_array_equal(y32, expected, err_msg=name + " f32 to i8")
            np.testing.assert_array_equal(y64, expected, err_msg=name + " f64 to i8")
            np.testing.assert_array_equal(x32, original_x32)
            np.testing.assert_array_equal(x64, original_x64)
            for storage in (storage32, storage64):
                np.testing.assert_array_equal(storage[[0, 1, -2, -1]], np.full(4, -101, np.int8))
            observed["checked"] = True
    finally:
        restore_status = bridge.wb_fe_restore(saved)
        environment["restored_mode"] = bridge.wb_fe_mode()
        environment["restored_exception_flags"] = bridge.wb_fe_flags()
        environment["restored"] = (restore_status == 0 and environment["restored_mode"] == original_mode
                                   and environment["restored_exception_flags"] == original_flags)
        (root / "dynamic-rounding-fenv.json").write_text(json.dumps(environment, indent=2))
        if not environment["restored"]:
            raise RuntimeError("failed to restore the original floating-point environment")
    return {"checked": True, "rounding": "dynamic", "single_compiled_kernel_reused": True,
            "kernel_invocations": len(environment["modes"]), "input_types": ["f32", "f64"],
            "output_type": "i8", "saturation": True, "oracle": "literal exact integer arrays",
            "input_unchanged": True, "output_guards_unchanged": True, "fenv": environment,
            "snapshot": str(snapshot_path), "evidence": evidence}

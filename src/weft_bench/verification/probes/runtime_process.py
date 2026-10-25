"""Small native-runtime contracts; no performance scan or compiler changes."""
import array
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import gc
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import weakref

import numpy as np
import triton
import triton.language as tl
from kernels import affine, coordinates

torch = None
if any(name != "packed_layouts" for name in sys.argv[1:]):
    import torch

faulthandler.enable()
root = Path(os.environ.get("WEFT_BENCH_VERIFY_OUTPUT", Path(__file__).parent))
root.mkdir(parents=True, exist_ok=True)
request = json.loads((Path(__file__).parent / "request.json").read_text())
options = {"num_cpu_threads": 1}
if "backend_options" in request:
    options["backend_options"] = request["backend_options"]
if not request.get("default_toolchain", False):
    options["toolchain"] = request["toolchain"]
rows = json.loads((root / "results.json").read_text())["results"] if (root / "results.json").exists() else []


def close(actual, expected):
    a = actual.detach().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    b = expected.detach().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=2e-5, atol=2e-5):
        raise ValueError("numerical mismatch")


def call(x, y):
    return affine[(1,)](x, y, **options)


def dense_view():
    from cpu_target import check_jit_target
    from dense_view_probe import run_scenario
    from triton.runtime.driver import driver

    check_jit_target(request.get("architecture"), request.get("compile_target"),
                     vars(driver.for_model("weft").get_current_target()))
    return run_scenario(request["dense_view_scenario"], root, options, close, reject)


def source_auto():
    from cpu_target import check_jit_target
    from source_auto_probe import run_scenario
    from triton.runtime.driver import driver

    check_jit_target(request.get("architecture"), request.get("compile_target"),
                     vars(driver.for_model("weft").get_current_target()))
    return run_scenario(request["source_auto_scenario"], root, options, close, request["compile_target"])


def numerical_primitives():
    from cpu_target import check_jit_target
    from numerical_primitives import run_scenario
    from triton.runtime.driver import driver

    if request.get("execution_mode") == "aot":
        if request["numerical_scenario"] not in {"pointwise_costs", "array_packets", "array_contractions", "lookup", "indexed_packets", "extrema_and_half", "byte_sum", "byte_planes", "byte_contractions", "bfloat_contractions", "amx_snapshots", "word_byte_contractions", "amx_mixed_storage", "matrix_operand_layout", "matrix_standard_pointer", "contraction_schedules", "streaming_stores", "cpu_tuning"}:
            raise NotImplementedError("explicit target AOT is not implemented for this numerical scenario")
        return run_scenario(request["numerical_scenario"], root, options, close, request["compile_target"])
    check_jit_target(request.get("architecture"), request.get("compile_target"),
                     vars(driver.for_model("weft").get_current_target()))
    return run_scenario(request["numerical_scenario"], root, options, close)


def reduction_domains():
    """Replay closed physical reductions, including domains not emitted by arrays."""
    import ctypes
    import re
    import subprocess
    from triton._C.libtriton import ir, weft
    from triton.backends.compiler import CPUTarget
    from triton.backends.weft.toolchain import Toolchain

    optimizer = request["optimizer"]
    target = CPUTarget(backend="weft", arch="x86_64", march="x86-64-v3", abi="sysv", vlen_bits=0)
    toolchain = Toolchain.resolve({}, target)
    cases = [(0, 0, 1), (7, -5, 3), (0, 7, 4), (-3, 4, 2), (0, 12, 2),
             (-11, 71, 3), (0, 4097, 16), (-17, 4201, 17),
             (-(1 << 63) + 100, -(1 << 63) + 181, 3),
             ((1 << 63) - 100, (1 << 63) - 19, 3),
             ((1 << 63) - 1, -(1 << 63), 1), (0, 1, (1 << 62) + 1)]
    functions, calls = [], []
    for static in (False, True):
        for lower, upper, step in cases:
            name = "reduce_" + str(len(calls))
            bounds = (f"%lower = arith.constant {lower} : index\n"
                      f"%upper = arith.constant {upper} : index") if static else (
                      "%lower = arith.index_cast %lo : i64 to index\n"
                      "%upper = arith.index_cast %hi : i64 to index")
            functions.append(f'''func.func @{name}(%lo: i64, %hi: i64) -> i64
                attributes {{weft.x86.vector_bits = 256 : i64,
                            weft.x86.vector_registers = 16 : i64,
                            weft.x86.array_schedule = [{{kind = "reduction"}}]}} {{
              {bounds}
              %step = arith.constant {step} : index
              %zero = arith.constant dense<0> : vector<4xi64>
              %sum = scf.for %iv = %lower to %upper step %step
                  iter_args(%partial = %zero) -> vector<4xi64> {{
                %scalar = arith.index_cast %iv : index to i64
                %packet = vector.broadcast %scalar : i64 to vector<4xi64>
                %next = arith.addi %partial, %packet : vector<4xi64>
                scf.yield %next : vector<4xi64>
              }} {{weft.x86.array_reduction, weft.x86.array_schedule_id = 0 : i64}}
              %result = vector.reduction <add>, %sum : vector<4xi64> into i64
              return %result : i64
            }}''')
            calls.append((name, lower, upper, step, static))
    abi = json.dumps(json.dumps({"march": "x86-64-v3"}))
    original = root / "reduction-domains.mlir"
    original.write_text("module attributes {weft.x86.abi = " + abi + "} {\n" + "\n".join(functions) + "\n}")
    commands = []
    def run(arguments):
        commands.append(arguments)
        result = subprocess.run(arguments, capture_output=True, text=True)
        (root / "commands.json").write_text(json.dumps(commands, indent=2))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout
    scheduled = run([optimizer, str(original), "--weft-x86-schedule-array-reductions", "--canonicalize", "--cse"])
    (root / "reduction-domains.scheduled.mlir").write_text(scheduled)
    selected = [int(value) for value in re.findall(r"weft.x86.partial_copies = (\d+)", scheduled)]
    assert selected and max(selected) > 1, "probe did not exercise an interleaved reduction"
    final = root / "reduction-domains.cf.mlir"
    final.write_text(run([optimizer, str(root / "reduction-domains.scheduled.mlir"), "--convert-scf-to-cf"]))
    context = ir.context()
    ir.load_dialects(context)
    weft.load_dialects(context)
    physical = ir.parse_mlir_module(str(final), context)
    llir = root / "reduction-domains.ll"
    llir.write_text(weft.emit_x86(physical)["llvm_ir"])
    library = root / "reduction-domains.so"
    run([toolchain.cc, "-O3", "-fPIC", "-shared", str(llir), "-o", str(library)])
    native = ctypes.CDLL(str(library))
    records = []
    for name, lower, upper, step, static in calls:
        function = getattr(native, name)
        function.argtypes = [ctypes.c_int64, ctypes.c_int64]
        function.restype = ctypes.c_int64
        expected = ctypes.c_int64(4 * sum(range(lower, upper, step))).value
        actual = function(lower, upper)
        if actual != expected:
            raise ValueError(f"numerical mismatch: {name} {actual} != {expected}")
        records.append({"lower": lower, "upper": upper, "step": step, "static": static, "value": actual})
    return {"execution_mode": "physical-pass-replay-native", "integer_exact": True,
            "selected_partial_copies": selected, "cases": records}


def packed_layouts():
    from cpu_target import check_jit_target
    from packed_layouts import run_scenario
    from triton.runtime.driver import driver

    check_jit_target(request.get("architecture"), request.get("compile_target"),
                     vars(driver.for_model("weft").get_current_target()))
    try:
        return run_scenario(request["packed_scenario"], root, options)
    except AssertionError as error:
        raise ValueError("numerical mismatch: " + str(error)) from error


def dynamic_rounding():
    from dynamic_rounding import run

    return run(root, options, request)


def reject(operation, words):
    try:
        operation()
    except Exception as error:
        if not any(word in str(error).lower() for word in words):
            raise AssertionError("unexpected rejection: " + str(error)) from error
        return {"type": type(error).__name__, "diagnostic": str(error)}
    raise AssertionError("invalid invocation was accepted")


def numpy_and_buffer():
    x = np.arange(195, dtype=np.float32).reshape(3, 65) / 1000
    y = np.empty_like(x)
    x.setflags(write=False)
    compiled = call(x, y); close(y, x * 1.25 - 0.5)
    explicit_x = tl.View(x, encoding=tl.f32, shape=x.shape)
    explicit_y = tl.View(y, encoding=tl.f32, shape=y.shape)
    call(explicit_x, explicit_y); close(y, x * 1.25 - 0.5)
    packed_x = array.array("f", (i/1000 for i in range(65)))
    packed_y = array.array("f", [-7]*65)
    call(tl.View(packed_x, encoding=tl.f32, shape=(1, 65)),
         tl.View(packed_y, encoding=tl.f32, shape=(1, 65)))
    close(np.asarray(packed_y), np.asarray(packed_x)*1.25-0.5)
    y.fill(-9);y.setflags(write=False)
    error = reject(lambda: call(x, y), ("writ", "readonly", "read-only"))
    assert np.all(y == -9)
    return {"readonly_input": True, "numpy": True, "explicit_view": True,
            "pep3118_array": True, "readonly_output_rejected": error,
            "requested_toolchain": request["toolchain"], "native_build": compiled.metadata.native_build}


def aliases():
    x = torch.arange(195, dtype=torch.float32).reshape(3, 65) / 1000
    initial = x.clone()
    alias = reject(lambda: call(x, x), ("alias", "overlap"))
    close(x, initial)
    storage = torch.full((65,), -7.0)
    y = storage.as_strided((3, 65), (0, 1))
    overlap = reject(lambda: call(x, y), ("overlap", "stride"))
    assert bool(torch.all(storage == -7))
    return {"cross_group_alias_rejected": alias, "writable_self_overlap_rejected": overlap,
            "no_partial_writes": True}


def resize_views():
    x = torch.arange(65, dtype=torch.float32).reshape(1, 65) / 1000
    y = torch.empty_like(x)
    old_x = tl.View(x, encoding=tl.f32, shape=x.shape)
    old_y = tl.View(y, encoding=tl.f32, shape=y.shape)
    first = call(x, y);close(y, x*1.25-0.5)
    call(old_x, old_y);close(y, x*1.25-0.5)
    x.add_(1)
    call(old_x, old_y);close(y, x*1.25-0.5)
    saved = y.clone()
    x.resize_(1, 97);x.copy_(torch.arange(97, dtype=torch.float32).reshape(1, 97)/1000)
    new_y = torch.empty_like(x)
    second = call(x, new_y);close(new_y, x*1.25-0.5)
    error = reject(lambda: call(old_x, old_y), ("changed", "storage", "metadata", "owner", "stale"))
    close(y, saved)
    call(tl.View(x, encoding=tl.f32, shape=x.shape), tl.View(new_y, encoding=tl.f32, shape=new_y.shape))
    close(new_y, x*1.25-0.5)
    return {"ordinary_call_after_resize": True, "fresh_view_after_resize": True,
            "unchanged_metadata_view_reusable": True, "stale_view_rejected": error,
            "same_compiled_kernel_after_shape_change": first is second}


def zero_and_concurrency():
    x = torch.arange(65, dtype=torch.float32).reshape(1, 65) / 1000
    y = torch.full_like(x, -9)
    affine[(0,)](x, y, **options)
    assert bool(torch.all(y == -9))
    barrier = threading.Barrier(2)

    def worker(index):
        values = torch.arange(64*4097, dtype=torch.float32).reshape(64, 4097)/2048 + index
        output = torch.empty_like(values)
        for _ in range(3):
            barrier.wait(timeout=30)
            call(values, output)
            close(output, values*1.25-0.5)
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, index) for index in range(2)]
        threads = [future.result() for future in futures]
    assert len(set(threads)) == 2
    return {"zero_grid_preserves_output": True, "concurrent_threads": threads,
            "calls_per_thread": 3, "independent_input_output": True}


def pid_grid():
    scenarios=[]
    for grid in ((2, 3), (2, 3, 4)):
        shape = (*grid, *([1]*(3-len(grid))))
        y = torch.full(shape, -1.0)
        coordinates[grid](y, toolchain=request["toolchain"], num_cpu_threads=2)
        expected = np.fromfunction(lambda i,j,k: i+10*j+100*k+1000*shape[0]+10000*shape[1]+100000*shape[2], shape).astype(np.float32)
        close(y, expected);scenarios.append({"grid": grid, "shape": shape})
    return {"scenarios": scenarios, "all_pid_and_num_program_axes_checked": True}


def affinity():
    original = os.sched_getaffinity(0)
    cpus = request["affinity_cpus"]
    if len(set(cpus[:2])) != 2:
        raise ValueError("affinity verification requires two distinct configured legal CPUs")
    x = torch.arange(65, dtype=torch.float32).reshape(1, 65)/1000;y=torch.empty_like(x)
    kernels=[]
    observations=[]
    events=[]
    loaded_first=None
    context_rejection=None
    previous_listener=triton.knobs.compilation.listener
    def listener(**event):
        events.append({"cache_hit": bool(event["cache_hit"]),
                       "target": repr(event["metadata"].get("target")),
                       "metadata_group": event["metadata_group"]})
        if previous_listener is not None:
            previous_listener(**event)
    triton.knobs.compilation.listener=listener
    def launch(cpu):
        os.sched_setaffinity(0, {cpu})
        before=len(events)
        kernel=call(x,y);close(y,x*1.25-0.5);kernels.append(kernel)
        observations.append({"cpu": cpu, "affinity": sorted(os.sched_getaffinity(0)),
            "python_object_id": id(kernel), "kernel_hash": kernel.hash,
            "target": repr(getattr(kernel.metadata,"target",None)),
            "metadata_group": kernel.metadata_group, "compilation_events": events[before:]})
        return kernel
    try:
        first=launch(cpus[0])
        artifact=root/"affinity-artifact"
        first.export(artifact)
        loaded_first=triton.load(artifact)
        # Loading is lazy: establish the SDK context on the first CPU set.
        y.fill_(float("nan"))
        loaded_first[(1,)](x,y,1.25,bias=-0.5)
        close(y,x*1.25-0.5)
        launch(cpus[1])
        y.fill_(-7)
        context_rejection=reject(lambda: loaded_first[(1,)](x,y,1.25,bias=-0.5),
                                 ("affinity", "discovered cpu set", "execution scope"))
        assert bool(torch.all(y == -7)), "invalid execution context wrote output"
        returned=launch(cpus[0])
        assert returned is first, "returning to the first CPU set did not reuse its CompiledKernel"
        loaded_first[(1,)](x,y,1.25,bias=-0.5)
        close(y,x*1.25-0.5)
    finally:
        os.sched_setaffinity(0,original)
        triton.knobs.compilation.listener=previous_listener
        evidence={"calls": observations, "compiler_entry_events": len(events),
                  "backend_compilations": sum(not event["cache_hit"] for event in events),
                  "compiler_cache_hits": sum(event["cache_hit"] for event in events),
                  "illegal_context_rejection": context_rejection,
                  "kernel_hash_origin": "provided by Triton; not computed by Bench"}
        (root/"affinity-observations.json").write_text(json.dumps(evidence,indent=2)+"\n")
    assert observations[0]["kernel_hash"] == observations[1]["kernel_hash"], "legal CPU sets changed compiled code identity"
    assert observations[0]["target"] == observations[1]["target"], "configured legal CPU sets have different targets"
    second_events=observations[1]["compilation_events"]
    assert second_events and all(event["cache_hit"] for event in second_events), "second CPU set recompiled native code"
    assert not observations[2]["compilation_events"], "first CPU set revisit missed its in-memory cache"
    return {"legal_cpu_sets": [[cpu] for cpu in cpus[:2]], "same_compiled_code": True,
            "first_context_reused": True, "second_cpu_compile_cache_hit": True,
            "compiler_entry_events": len(events), "backend_compilations": evidence["backend_compilations"],
            "illegal_context_rejection": context_rejection,
            "worker_checks_retained": True, "openmp_pool_migration": False}


def lifetime():
    x=torch.arange(65,dtype=torch.float32).reshape(1,65)/1000;y=torch.empty_like(x)
    artifact=root/"lifetime-artifact"
    call(x,y).export(artifact)
    loaded=triton.load(artifact)
    loaded[(1,)](x,y,1.25,bias=-0.5)
    close(y,x*1.25-0.5)
    native=loaded.run
    module=loaded.module
    release=module._release
    owner=weakref.ref(module);compiled=weakref.ref(loaded)
    arguments={"X":x,"Y":y,"scale":1.25,"bias":-0.5,"BLOCK":64}
    tail=tuple(arguments[name] for name in loaded.metadata.launch_args)
    prefix=(1,1,1,None,loaded.function,loaded.packed_metadata,None,None,None)
    del loaded,module
    gc.collect()
    assert compiled() is None, "CompiledKernel retained unexpectedly"
    assert release.alive and owner() is not None, "SDK handle released before native callable"
    native(*prefix,*tail);close(y,x*1.25-0.5)
    del native
    gc.collect()
    assert not release.alive and owner() is None, "last callable reference did not release its SDK owner"
    return {"call_after_compiled_kernel_release": True, "sdk_owner_live_during_call": True,
            "last_reference_released": True}


def integer_type_protocol():
    from triton._C.libtriton import ir

    context = ir.context()
    builder = ir.builder(context)
    for bits in (8, 16, 32, 64):
        signed = getattr(tl, "int" + str(bits)).to_ir(builder)
        unsigned = getattr(tl, "uint" + str(bits)).to_ir(builder)
        assert signed == unsigned, "the ordinary TTIR builder must retain signless integer types"
        assert getattr(builder, "get_uint" + str(bits) + "_ty")() == unsigned
    return {"ordinary_ttir_signedness_representation_preserved": True, "integer_widths": [8, 16, 32, 64]}


def conversion_jit():
    from numerical_kernels import cast_roundtrip

    x = torch.tensor([1.0006, -1.0006, 65520.0, -65520.0, 2.0**-25, -(2.0**-25),
                      float("inf"), float("nan")], dtype=torch.float32)
    y = torch.empty_like(x)
    expected = np.array([1.0, -1.0, 65504.0, -65504.0, 0.0, -0.0, np.inf, np.nan], dtype=np.float32)
    first = None
    for _ in range(2):
        y.fill_(17)
        kernel = cast_roundtrip[(1,)](x, y, N=x.numel(), MODE=1, BLOCK=16, **options)
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-6, atol=0, equal_nan=True)
        assert np.array_equal(np.signbit(y.numpy()[4:6]), np.signbit(expected[4:6]))
        if first is None:
            first = kernel
        else:
            assert kernel is first, "warm conversion launch recompiled its kernel"
    return {"checked": True, "scalar_default_bound_by_jit": True,
            "cold_and_warm_kernel_reused": True, "rounding": "rtz",
            "special_values": ["signed-zero", "overflow", "infinity", "nan"],
            "native_build": first.metadata.native_build}


def exp_boundaries():
    from numerical_kernels import pointer_exp

    minimum = np.nextafter(np.float32(0), np.float32(1))
    tiny = np.finfo(np.float32).tiny
    maximum = np.finfo(np.float32).max
    values = [-200, -104, -103, -100, -88, -87, -80, -10, -5, -1, -0.1,
              -0.0, 0.0, 0.1, 1, 5, 10, 40, 80, 88.7, 88.72284, 90, 100]
    for boundary in (np.log(float(minimum)/2), np.log(float(minimum)),
                     np.log(float(tiny)), np.log(float(maximum))):
        center = np.float32(boundary)
        values.extend((np.nextafter(center, np.float32(-np.inf)), center,
                       np.nextafter(center, np.float32(np.inf))))
    values.extend((float("nan"), float("inf"), -float("inf")))
    original = np.array(values, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = np.exp(original.astype(np.float64)).astype(np.float32)
    count = original.size
    capacity = 1 << count.bit_length()
    x = torch.from_numpy(original.copy())
    y = torch.full((capacity,), -17.0, dtype=torch.float32)
    snapshot = root / "exp-boundaries.npz"
    np.savez(snapshot, inputs=original, expected=expected)
    compiled = pointer_exp[(1,)](x, y, N=count, BLOCK=capacity, **options)
    actual = y.numpy()[:count]
    np.savez(snapshot, inputs=original, expected=expected, actual=actual,
             output_storage=y.numpy())
    np.testing.assert_array_equal(x.numpy(), original)
    np.testing.assert_array_equal(y.numpy()[count:], np.full(capacity-count, -17.0, np.float32))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(actual == 0, expected == 0)
    assert not np.any(np.signbit(actual[~np.isnan(actual)])), "exp produced a negative result or negative zero"
    positive = np.isfinite(expected) & (expected > 0)
    assert np.all(np.isfinite(actual[positive]) & (actual[positive] > 0)), "finite positive exp was flushed or overflowed"
    error = np.abs(actual[positive].astype(np.float64) - expected[positive].astype(np.float64))
    # Keep the runtime relative tolerance; one subnormal step accounts for f32
    # quantization without allowing small positive results to flush to zero.
    allowed = np.maximum(2e-5 * expected[positive].astype(np.float64), float(minimum))
    if np.any(error > allowed):
        raise ValueError("numerical mismatch in exp; see exp-boundaries.npz")
    return {"checked": True, "elements": count, "masked": capacity-count,
            "subnormal_outputs": int(np.sum((expected > 0) & (expected < tiny))),
            "oracle": "numpy float64 exp rounded to float32", "rtol": 2e-5,
            "absolute_rounding_floor": float(minimum), "special_value_classes_exact": True,
            "positive_subnormals_preserved": True, "tail_guard_unchanged": True,
            "inputs_outputs": str(snapshot), "compile_target": vars(compiled.metadata.target),
            "native_build": compiled.metadata.native_build}


def integer_float_conversion():
    from numerical_kernels import integer_to_float

    def check(integer_bits, float_bits, unsigned):
        source_dtype = np.dtype(("uint" if unsigned else "int") + str(integer_bits))
        output_dtype = np.dtype("float" + str(float_bits))
        limits = np.iinfo(source_dtype)
        precision = np.finfo(output_dtype).nmant + 1
        # Include exact values, rounding ties and their neighbors, integer
        # endpoints, and unsigned values on both sides of the sign bit.
        points = [0, 1, 2, 3, 7, 17, int(limits.min), int(limits.max)]
        for center in (2**precision, 2**(integer_bits-1)):
            points.extend(center + delta for delta in (-3, -2, -1, 0, 1, 2, 3))
        if not unsigned:
            points += [-value for value in points]
        values = sorted({value for value in points if limits.min <= value <= limits.max})
        count = len(values)
        capacity = 1 << (count + 3).bit_length()
        original = np.full(capacity, 7, dtype=source_dtype)
        original[:count] = values
        initial_output = np.full(capacity, 17.25, dtype=output_dtype)
        expected = initial_output.copy()
        with np.errstate(over="ignore"):
            expected[:count] = original[:count].astype(output_dtype)
        incoming = torch.from_numpy(original.copy())
        output = torch.from_numpy(initial_output.copy())
        compiled = integer_to_float[(1,)](incoming, output, N=count, BLOCK=capacity, **options)
        np.testing.assert_array_equal(output.numpy(), expected)
        np.testing.assert_array_equal(incoming.numpy(), original)
        return {"source_dtype": source_dtype.name, "output_dtype": output_dtype.name,
                "inputs": values, "expected": [float(x) if np.isfinite(x) else str(x)
                                                for x in expected[:count]],
                "active": count, "masked": capacity-count, "exact_comparison": True,
                "native_build": compiled.metadata.native_build}

    pairs = ((8, 16), (16, 16), (16, 32), (32, 16), (32, 32), (32, 64), (64, 32), (64, 64))
    scenarios = [(("u" if unsigned else "i") + str(src) + "-f" + str(dst),
                  lambda src=src, dst=dst, unsigned=unsigned: check(src, dst, unsigned))
                 for src, dst in pairs for unsigned in (False, True)]
    return {"scenarios": runtime_scenarios("integer-float-conversion", scenarios),
            "oracle": "numpy.astype", "masked_storage_unchanged": True,
            "input_unchanged": True, "direct_conversion": True}


def helper_native_evidence(compiled, label, minimum_noinline=0):
    from cpu_target import check_jit_target
    from helper_native import capture

    check_jit_target(request.get("architecture"), request.get("compile_target"), vars(compiled.metadata.target))
    return capture(compiled, root, label, minimum_noinline)


def helper_jit():
    from control_kernels import helper_composition, helper_levels

    x = torch.linspace(-1, 1, 128, dtype=torch.float32)
    y = torch.empty_like(x)
    expected = (x + 3) * 0.25
    compiled = []
    for mode in (1, 3, 4):
        first = None
        for _ in range(2):
            y.fill_(float("nan"))
            kernel = helper_composition[(8 if mode == 3 else 128,)](
                x, y, N=x.numel(), MODE=mode, **options)
            close(y, expected)
            if first is None:
                first = kernel
            else:
                assert kernel is first, "warm helper launch recompiled its kernel"
        compiled.append(first)
    levels = helper_levels[(1,)](x, y, **options)
    close(y, expected)
    return {"checked": True, "cold_and_warm_kernel_reused": True,
            "paths": ["tuple-while", "constexpr-vector", "early-store", "nested-level-state-stage"],
            "native_build": compiled[0].metadata.native_build,
            "native_evidence": [helper_native_evidence(kernel, f"helper-jit-{index}")
                                for index, kernel in enumerate([*compiled, levels])]}


def pointer_memory():
    from memory_kernels import affine_pointer_control
    from cpu_target import check_jit_target
    from triton.runtime.driver import driver

    detected = check_jit_target(request.get("architecture"), request.get("compile_target"),
                               vars(driver.for_model("weft").get_current_target()))

    x = torch.linspace(-1, 1, 131, dtype=torch.float32)
    y = torch.full((138,), -17.0, dtype=torch.float32)
    first = None
    for _ in range(2):
        y.fill_(-17.0)
        kernel = affine_pointer_control[(2,)](x, y, x.numel(), BLOCK=128, **options)
        close(y[:131], (x + 3.0) * 0.25)
        assert torch.all(y[131:] == -17), "tail store changed the output guard"
        if first is None:
            first = kernel
        else:
            assert kernel is first, "warm pointer launch recompiled its kernel"
    if detected["arch"] == "riscv64":
        assert "rvv.pointer-unit.load" in first.asm["riscvir"]
        assert "rvv.pointer-indexed" not in first.asm["riscvir"]
        evidence = {"stage": "riscvir", "unit_load_selected": True}
    elif detected["arch"] == "x86_64":
        assert first.asm["x86ir"] and first.asm["llir"] and first.asm["so"]
        assert first.metadata.native_build["language"] == "llvm-ir"
        evidence = {"stages": ["x86ir", "llir", "so"], "native_language": "llvm-ir"}
    else:
        raise NotImplementedError("pointer physical evidence for " + detected["arch"])
    return first, x, y, {"checked": True, "cold_and_warm_kernel_reused": True,
                         "tail_guard_unchanged": True, "compile_target": detected,
                         "physical_evidence": evidence, "native_build": first.metadata.native_build}


def pointer_jit_aot():
    if request.get("default_toolchain", False):
        assert "toolchain" not in options and not request.get("toolchain"), "default toolchain probe has explicit compiler options"
        assert "TRITON_WEFT_CC" not in os.environ, "default toolchain probe inherited TRITON_WEFT_CC"
    first, x, y, evidence = pointer_memory()
    artifact = root / "pointer-artifact"
    first.export(artifact)
    loaded = triton.load(artifact)
    y.fill_(-17.0)
    loaded[(2,)](x, y, x.numel())
    close(y[:131], (x + 3.0) * 0.25)
    assert torch.all(y[131:] == -17), "AOT tail store changed the output guard"
    if request.get("default_toolchain", False):
        from bundled_toolchain import audit_runtime

        sdk = (Path(triton.__file__).parent / "_C" / "weft").resolve()
        build = first.metadata.native_build["toolchain"]
        exported = json.loads((artifact / "triton.json").read_text())["runtime_files"]
        loaded_omp = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                             if "/libomp" in line})
        audit = audit_runtime(sdk=sdk, compiler=build["cc"], installed=build["runtime_files"],
                              cached=first.metadata.runtime_files, exported=exported, artifact=artifact,
                              cache=os.environ["TRITON_CACHE_DIR"], loaded=loaded_omp,
                              report=root / "bundled-toolchain.json")
        evidence["bundled_toolchain"] = {**audit, "explicit_toolchain_argument": False,
                                         "compiler_override": False}
    return {**evidence, "grid": [2], "python_aot_correct": True,
            "aot_tail_guard_unchanged": True, "artifact": str(artifact)}


def triton_frontend_memory():
    from triton.compiler.weft.frontend import CanonicalSemantic
    from triton.language.semantic import TritonSemantic
    from memory_kernels import standard_dot_options

    assert CanonicalSemantic.dot is TritonSemantic.dot, "ordinary dot bypassed Triton semantics"
    first, _, _, evidence = pointer_memory()
    a = torch.arange(512, dtype=torch.float32).reshape(16, 32) / 512
    b = torch.arange(512, dtype=torch.float32).reshape(32, 16) / 1024
    acc = torch.full((16, 16), 0.125, dtype=torch.float32)
    output = torch.empty_like(acc)
    for mode in (0, 1, 4):
        standard_dot_options[(1,)](a, b, acc, output, MODE=mode, **options)
        close(output, a @ b + acc)
    return {**evidence, "ordinary_dot_uses_triton_semantic": True,
            "masked_unit_memory": evidence["physical_evidence"].get("unit_load_selected", False)}


def associative_scan():
    from riscv_kernels import scan_axes

    values = (np.arange(32, dtype=np.int32).reshape(4, 8) % 11) - 5
    x = torch.from_numpy(values.copy())
    y, z = torch.empty_like(x), torch.empty_like(x)
    records = []
    for axis in (0, 1, -1):
        for reverse in (False, True):
            for tuple_input in (False, True):
                y.fill_(-999); z.fill_(-999)
                scan_axes[(1,)](x, y, z, M=4, N=8, AXIS=axis, REVERSE=reverse,
                               TUPLE=tuple_input, **options)
                sequence = np.flip(values, axis) if reverse else values
                summed = np.cumsum(sequence, axis=axis, dtype=np.int32)
                maximum = np.maximum.accumulate(sequence, axis=axis)
                if reverse:
                    summed, maximum = np.flip(summed, axis), np.flip(maximum, axis)
                np.testing.assert_array_equal(y.numpy(), summed)
                np.testing.assert_array_equal(z.numpy(), maximum if tuple_input else np.full_like(values, -999))
                np.testing.assert_array_equal(x.numpy(), values)
                records.append({"axis": axis, "reverse": reverse, "tuple_input": tuple_input,
                                "elements": values.size})
    return {"scenarios": records, "integer_exact": True, "input_unchanged": True}


def runtime_scenarios(name, scenarios):
    """Keep each fixed dtype result, including compiler rejection and its IR."""
    from compile_triton import save_backend_failures
    records, first_error = [], None
    for label, operation in scenarios:
        record = {"scenario": label, "status": "running"}
        records.append(record)
        try:
            record.update(operation(), status="passed", supported=True)
        except Exception as error:
            record.update(status="failed", supported=False, error=str(error),
                          traceback=traceback.format_exc(),
                          backend_diagnostics=save_backend_failures(error, root, name + "-" + label))
            if first_error is None:
                first_error = error
        (root / (name + "-scenarios.json")).write_text(json.dumps(records, indent=2)+'\n')
    if first_error is not None:
        failed = [r["scenario"] for r in records if r["status"] == "failed"]
        raise RuntimeError(name + ": failed " + ", ".join(failed) +
                           "; see " + name + "-scenarios.json") from first_error
    return records


def level_lifetimes():
    import level_kernels
    from level_scenarios import scenarios, SENTINEL

    name = request["level_scenario"]

    def check(spec):
        x = torch.from_numpy(spec["input"].copy())
        y = torch.full(spec["expected_output"].shape, SENTINEL, dtype=torch.float32)
        snapshot = root / ("level-" + name + "-" + spec["name"] + ".npz")
        captured = {"input": spec["input"], "expected_input": spec["expected_input"],
                    "expected_output": spec["expected_output"]}
        np.savez(snapshot, **captured)
        compiled = getattr(level_kernels, spec["symbol"])[(1,)](x, y, **spec["scalars"], **options)
        np.savez(snapshot, **captured, actual_input=x.numpy(), actual_output=y.numpy())
        close(x, spec["expected_input"])
        close(y, spec["expected_output"])
        np.testing.assert_array_equal(y.numpy()[spec["output_elements"]:],
                                      spec["expected_output"][spec["output_elements"]:])
        assert all(arg["kind"] in ("pointer", "scalar") for arg in compiled.metadata.kernel_abi["arguments"])
        return {"scalars": spec["scalars"], "output_elements": spec["output_elements"],
                "tail_guard_unchanged": True, "pointer_scalar_abi": True,
                "snapshot": str(snapshot), "native_build": compiled.metadata.native_build}

    records = runtime_scenarios("level-" + name,
                               [(spec["name"], lambda spec=spec: check(spec)) for spec in scenarios(name)])
    return {"checked": True, "level_scenario": name, "scenarios": records,
            "oracle": "independent NumPy lifecycle recurrence", "rtol": 2e-5, "atol": 2e-5}


def atomic_rmw():
    from riscv_kernels import atomic_rmw as kernel

    count, capacity = 27, 32
    names = ("add", "max", "min", "and", "or", "xor", "xchg")
    operations = (np.add, np.maximum, np.minimum, np.bitwise_and,
                  np.bitwise_or, np.bitwise_xor, lambda left, right: right)

    def integer(dtype, op):
        if dtype == np.uint32:
            original = np.arange(capacity, dtype=np.uint32) + np.uint32(0x80000000)
            values = np.resize(np.array([0, 17, 0x7fffffff, 0x80000000, 0xffffffff], np.uint32), capacity)
            sentinel = 0xa5a5a5a5
        else:
            factor = 2**33 if dtype == np.int64 else 1
            original = (np.arange(capacity, dtype=dtype)-17)*factor
            values = (19-np.arange(capacity, dtype=dtype))*(factor//2 if factor>1 else 1)
            sentinel = -999
        initial_old = np.full(capacity, sentinel, dtype=dtype)
        state = torch.from_numpy(original.copy()); incoming = torch.from_numpy(values.copy())
        old = torch.from_numpy(initial_old.copy())
        compiled = kernel[(1,)](state, incoming, old, N=count, OP=op, BLOCK=capacity, **options)
        expected = original.copy(); expected[:count] = operations[op](original[:count], values[:count])
        expected_old = initial_old.copy(); expected_old[:count] = original[:count]
        np.testing.assert_array_equal(state.numpy(), expected)
        np.testing.assert_array_equal(old.numpy(), expected_old)
        np.testing.assert_array_equal(incoming.numpy(), values)
        return {"operation": names[op], "dtype": np.dtype(dtype).name, "active": count,
                "masked": capacity-count, "native_build": compiled.metadata.native_build}

    def floating(dtype):
        original = np.arange(capacity, dtype=dtype)/dtype(4)
        values = np.full(capacity, 0.5, dtype=dtype)
        initial_old = np.full(capacity, -999, dtype=dtype)
        state = torch.from_numpy(original.copy()); incoming = torch.from_numpy(values.copy())
        old = torch.from_numpy(initial_old.copy())
        compiled = kernel[(1,)](state, incoming, old, N=count, OP=0, BLOCK=capacity, **options)
        expected = original.copy(); expected[:count] += dtype(0.5)
        expected_old = initial_old.copy(); expected_old[:count] = original[:count]
        # These binary fractions are exactly representable even in fp16.
        np.testing.assert_array_equal(state.numpy(), expected)
        np.testing.assert_array_equal(old.numpy(), expected_old)
        np.testing.assert_array_equal(incoming.numpy(), values)
        return {"operation": "add", "dtype": np.dtype(dtype).name, "active": count,
                "masked": capacity-count, "exact_binary_fractions": True,
                "native_build": compiled.metadata.native_build}

    scenarios = [(np.dtype(dtype).name + "-" + name, lambda dtype=dtype, op=op: integer(dtype, op))
                 for dtype in (np.int32, np.int64, np.uint32) for op, name in enumerate(names)]
    scenarios += [(np.dtype(dtype).name + "-add", lambda dtype=dtype: floating(dtype))
                  for dtype in (np.float32, np.float16, np.float64)]
    return {"scenarios": runtime_scenarios("atomic-rmw", scenarios), "old_values_checked": True,
            "masked_storage_unchanged": True}


def scan_affine_reverse():
    from riscv_kernels import scan_affine

    shape = (4, 8)
    a = (np.arange(32).reshape(shape) % 3 - 1).astype(np.int32)
    b = (np.arange(32).reshape(shape) % 7 - 3).astype(np.int32)
    records = []
    for axis in (0, 1):
        x, values = torch.from_numpy(a.copy()), torch.from_numpy(b.copy())
        y, z = torch.empty_like(x), torch.empty_like(x)
        scan_affine[(1,)](x, values, y, z, M=4, N=8, AXIS=axis, REVERSE=True, **options)
        flipped_a, flipped_b = np.moveaxis(np.flip(a, axis), axis, 0), np.moveaxis(np.flip(b, axis), axis, 0)
        expected_a, expected_b = np.empty_like(flipped_a), np.empty_like(flipped_b)
        carry_a = np.ones(flipped_a.shape[1:], dtype=np.int32)
        carry_b = np.zeros_like(carry_a)
        for i in range(len(flipped_a)):
            carry_a, carry_b = carry_a*flipped_a[i], carry_b*flipped_a[i]+flipped_b[i]
            expected_a[i], expected_b[i] = carry_a, carry_b
        expected_a = np.flip(np.moveaxis(expected_a, 0, axis), axis)
        expected_b = np.flip(np.moveaxis(expected_b, 0, axis), axis)
        np.testing.assert_array_equal(y.numpy(), expected_a)
        np.testing.assert_array_equal(z.numpy(), expected_b)
        records.append({"axis": axis, "elements": int(a.size)})
    return {"scenarios": records, "combiner": "(a,b)*(c,d)=(a*c,b*c+d)",
            "reference": "flip(scan(flip(input)))", "integer_exact": True}


def atomic_cas():
    from riscv_kernels import atomic_cas_slots

    count, capacity = 27, 32

    def check(dtype):
        if dtype == np.uint32:
            initial = np.arange(capacity, dtype=dtype)+np.uint32(0x80000000)
            sentinel = 0xa5a5a5a5
        else:
            initial = (np.arange(capacity, dtype=dtype)-7)*(2**33 if dtype==np.int64 else 1)
            sentinel = -999
        comparisons = initial.copy(); comparisons[1::2] += 1
        values = (np.arange(capacity, dtype=dtype)+100)*(2**32 if dtype==np.int64 else 1)
        initial_old = np.full(capacity, sentinel, dtype=dtype)
        state = torch.from_numpy(initial.copy()); compare = torch.from_numpy(comparisons.copy())
        incoming = torch.from_numpy(values.copy()); old = torch.from_numpy(initial_old.copy())
        compiled = atomic_cas_slots[(capacity,)](state, compare, incoming, old, N=count, **options)
        expected = initial.copy(); expected[:count:2] = values[:count:2]
        expected_old = initial_old.copy(); expected_old[:count] = initial[:count]
        np.testing.assert_array_equal(state.numpy(), expected)
        np.testing.assert_array_equal(old.numpy(), expected_old)
        np.testing.assert_array_equal(compare.numpy(), comparisons)
        np.testing.assert_array_equal(incoming.numpy(), values)
        return {"dtype": np.dtype(dtype).name, "successful_and_failed_compare": True,
                "old_values_checked": True, "guarded_calls": count, "inactive_calls": capacity-count,
                "native_build": compiled.metadata.native_build}

    scenarios = [(np.dtype(dtype).name, lambda dtype=dtype: check(dtype))
                 for dtype in (np.int32, np.int16, np.int64, np.uint32)]
    return {"scenarios": runtime_scenarios("atomic-cas", scenarios)}


def atomic_contention():
    from riscv_kernels import atomic_contended, atomic_cas_contended

    count, capacity = 53, 64
    threads = int(request.get("atomic_threads", 2))
    launch_options = {**options, "num_cpu_threads": threads}
    state = torch.tensor([11, 29], dtype=torch.int32)
    old = torch.full((capacity,), -999, dtype=torch.int32)
    atomic_contended[(capacity//8,)](state, old, N=count, BLOCK=8, **launch_options)
    for slot, start in enumerate((11, 29)):
        observed = old[:count].numpy()[slot::2]
        np.testing.assert_array_equal(np.sort(observed), np.arange(start, start+observed.size))
        assert state[slot].item() == start+observed.size
    assert bool(torch.all(old[count:] == -999))
    state = torch.zeros(1, dtype=torch.int32); old = torch.full((16,), -999, dtype=torch.int32)
    atomic_cas_contended[(16,)](state, old, **launch_options)
    winner = int(state.item())
    assert 1 <= winner <= 16
    assert old[winner-1].item() == 0 and int((old == 0).sum()) == 1
    assert bool(torch.all((old == 0) | (old == winner)))
    return {"threads": threads, "rmw_active_lanes": count, "rmw_buckets": 2,
            "rmw_old_value_permutations_checked": True, "cas_contenders": 16,
            "cas_single_winner": winner, "ordering": "scheduler-independent invariants"}


def helper_loop_returns():
    from riscv_kernels import helper_returns

    x = torch.arange(35, dtype=torch.float32) / 8
    expected_offset = np.array([3*stop+1 if 0 <= stop < 4 else 22
                                for stop in np.arange(35)%6-1], dtype=np.float32)
    evidence = []
    for mode in (0, 1):
        storage = torch.full((41,), -999.0); y = storage[:35]
        compiled = helper_returns[(40,)](x, y, N=35, MODE=mode, **options)
        evidence.append(helper_native_evidence(compiled, f"helper-loop-{mode}"))
        close(y, (x.numpy()+expected_offset)*2)
        assert bool(torch.all(storage[35:] == -999))
    return {"for_and_while": True, "early_and_fallthrough_paths": True,
            "caller_continues_after_return": True,
            "tail_guard_unchanged": True, "native_evidence": evidence}


def helper_noinline():
    from riscv_kernels import helper_returns

    x = torch.arange(35, dtype=torch.float32) / 8
    storage = torch.full((41,), -999.0); y = storage[:35]
    compiled = helper_returns[(40,)](x, y, N=35, MODE=2, **options)
    evidence = helper_native_evidence(compiled, "helper-noinline", minimum_noinline=1)
    close(y, (x.numpy()+np.arange(35)%2+1)*2)
    assert bool(torch.all(storage[35:] == -999))
    return {"noinline_requested": True, "both_helper_returns_checked": True,
            "caller_continues_after_return": True, "tail_guard_unchanged": True, "native_evidence": evidence}


def helper_context_expected(source, stop):
    expected = np.full(source.shape, -999.0, dtype=np.float32)
    if 0 <= stop < 3:
        if stop > 0:
            expected[:] = source + stop
        expected[:64] = source[:64] + stop + 1
    else:
        expected[:] = source + 10
    return expected


def helper_context_return():
    from riscv_kernels import helper_context_return as kernel

    scenarios, evidence = [], []
    for n in (35, 131):
        x = torch.arange(n, dtype=torch.float32) / 8
        original = x.clone()
        for stop in (-1, 0, 1, 2, 3):
            storage = torch.full((n+6,), -999.0); y = storage[:n]
            flag = torch.zeros(1, dtype=torch.int32)
            expected = helper_context_expected(original.numpy(), stop)
            snapshot = root / f"helper-context-n{n}-stop{stop}.npz"
            np.savez(snapshot, source=original.numpy(), expected=expected, initial=storage.numpy())
            compiled = kernel[(1,)](x, y, flag, stop, **options)
            np.savez(snapshot, source=original.numpy(), actual_source=x.numpy(), expected=expected,
                     actual=storage.numpy(), caller_flag=flag.numpy())
            close(y, expected)
            assert torch.equal(x, original), "helper changed input storage"
            assert flag.item() == 77 and bool(torch.all(storage[n:] == -999))
            scenarios.append({"N": n, "stop": stop, "first_block_elements": min(n, 64),
                              "remaining_blocks": "sentinel" if stop == 0 else
                                  "previous complete step" if stop in (1, 2) else "fallthrough",
                              "snapshot": str(snapshot)})
        evidence.append(helper_native_evidence(compiled, f"helper-context-n{n}"))
    return {"level_inside_loop": True, "early_and_fallthrough_paths": True,
            "caller_continues_after_return": True, "partial_tile_guard_unchanged": True,
            "remaining_level_instances_skipped": True, "scenarios": scenarios, "native_evidence": evidence}


def helper_descriptors():
    from control_kernels import helper_descriptor_composition

    m, n = 3, 35
    source = torch.arange(m*(2*n+5), dtype=torch.float32).reshape(m, 2*n+5)/2048
    original = source.clone()
    destination = torch.full((m, 3*n+6), -999.0)
    temporary = torch.full((m, 4*n+7), -777.0)
    x = source[:, 1:1+2*n:2]
    y = destination[:, 2:2+3*n:3]
    scratch = temporary[:, 3:3+4*n:4]
    bias = torch.tensor([-0.5], dtype=torch.float32)
    reference = x.numpy().astype(np.float64)
    expected_scratch = temporary.numpy().copy()
    expected_scratch[:, 3:3+4*n:4] = reference*1.25-0.5
    expected = destination.numpy().copy()
    expected[:, 2:2+3*n:3] = reference*3.75-0.5
    output_guard = torch.ones_like(destination, dtype=torch.bool)
    output_guard[:, 2:2+3*n:3] = False
    scratch_guard = torch.ones_like(temporary, dtype=torch.bool)
    scratch_guard[:, 3:3+4*n:4] = False
    snapshot = root / "helper-descriptors.npz"
    np.savez(snapshot, source=original.numpy(), expected=expected, expected_scratch=expected_scratch)
    compiled = helper_descriptor_composition[(1,)](x, y, scratch, bias, **options)
    np.savez(snapshot, source=original.numpy(), actual_source=source.numpy(), expected=expected,
             actual=destination.numpy(), expected_scratch=expected_scratch, actual_scratch=temporary.numpy())
    evidence = helper_native_evidence(compiled, "helper-descriptors", minimum_noinline=2)
    close(destination, expected)
    close(temporary, expected_scratch)
    assert torch.equal(source, original) and bias.item() == -0.5, "helper changed read-only storage"
    assert bool(torch.all(destination[output_guard] == -999)), "helper overwrote output guard"
    assert bool(torch.all(temporary[scratch_guard] == -777)), "helper overwrote scratch guard"
    return {"shape": [m, n], "fixed_value_shape": [1, 16], "nested_helpers": True,
            "outer_level_captures": True, "descriptor_strides": [list(v.stride()) for v in (x, y, scratch)],
            "descriptor_offsets": [v.storage_offset() for v in (x, y, scratch)],
            "scratch_then_output_then_caller_effect_checked": True, "guards_unchanged": True,
            "oracle": "independent float64 affine formulas", "snapshot": str(snapshot), "native_evidence": evidence}


def helper_level_capture():
    from control_kernels import helper_parent_capture

    n = 73
    source = torch.arange(2*n+5, dtype=torch.float32)/1024
    original = source.clone()
    output = torch.full((3*n+7,), -999.0)
    x, y = source[1:1+2*n:2], output[2:2+3*n:3]
    bias = torch.tensor([-0.25], dtype=torch.float32)
    flag = torch.zeros(1, dtype=torch.int32)
    expected = output.numpy().copy()
    expected[2:2+3*n:3] = x.numpy().astype(np.float64)*3.75+0.25
    guard = torch.ones_like(output, dtype=torch.bool)
    guard[2:2+3*n:3] = False
    snapshot = root / "helper-level-capture.npz"
    np.savez(snapshot, source=original.numpy(), expected=expected, initial=output.numpy())
    compiled = helper_parent_capture[(1,)](x, y, bias, flag, **options)
    np.savez(snapshot, source=original.numpy(), actual_source=source.numpy(), expected=expected,
             actual=output.numpy(), caller_flag=flag.numpy())
    evidence = helper_native_evidence(compiled, "helper-level-capture", minimum_noinline=2)
    close(output, expected)
    assert torch.equal(source, original) and bias.item() == -0.25, "helper changed read-only storage"
    assert bool(torch.all(output[guard] == -999)), "helper overwrote View padding"
    assert flag.item() == 3, "caller did not continue after each parent helper call"
    return {"N": n, "parent_extent": 32, "child_extent": 8, "parent_origins": [0, 32, 64],
            "parent_tail": 9, "last_child_tail": 1, "point_argument": True,
            "selected_view_arguments": True, "state_and_stage_births": True, "nested_helper": True,
            "caller_continuations": flag.item(), "x_stride": list(x.stride()), "y_stride": list(y.stride()),
            "x_offset": x.storage_offset(), "y_offset": y.storage_offset(), "guards_unchanged": True,
            "oracle": "independent float64 affine formula", "snapshot": str(snapshot), "native_evidence": evidence}


def default_toolchain():
    x = torch.arange(65, dtype=torch.float32).reshape(1, 65)/1000
    y = torch.empty_like(x)
    compiled = call(x, y)
    close(y, x*1.25-0.5)
    build = compiled.metadata.native_build
    toolchain = build["toolchain"]
    identity = toolchain["identity"]["native_runtime"]
    assert request["toolchain"] == {} and identity
    assert "--gcc-install-dir=" + identity["installation"] in toolchain["cflags"]
    for name in ("libgcc.a", "crtbeginS.o", "crtendS.o"):
        metadata = identity["inputs"][name]
        actual = (Path(identity["installation"]) / name).stat()
        assert metadata["size"] == actual.st_size and metadata["mtime_ns"] == actual.st_mtime_ns
    return {"requested_toolchain": {}, "gcc_runtime_identity_retained": True,
            "actual_runtime_file_identity_matches": True, "native_build": build}


operations={"reduction_domains":reduction_domains,"dynamic_rounding":dynamic_rounding,"packed_layouts":packed_layouts,"numerical_primitives":numerical_primitives,"source_auto":source_auto,"dense_view":dense_view,
            "numpy_buffer":numpy_and_buffer,"aliases":aliases,"resize_views":resize_views,
            "zero_concurrency":zero_and_concurrency,"pid_grid":pid_grid,"affinity":affinity,"lifetime":lifetime,
            "default_toolchain":default_toolchain,"integer_type_protocol":integer_type_protocol,
            "conversion_jit":conversion_jit,"integer_float_conversion":integer_float_conversion,
            "exp_boundaries":exp_boundaries,"helper_jit":helper_jit,"level_lifetimes":level_lifetimes,
            "triton_frontend_memory":triton_frontend_memory,"pointer_jit_aot":pointer_jit_aot,
            "associative_scan":associative_scan,"scan_affine_reverse":scan_affine_reverse,"atomic_rmw":atomic_rmw,
            "atomic_cas":atomic_cas,"atomic_contention":atomic_contention,
            "helper_loop_returns":helper_loop_returns,"helper_context_return":helper_context_return,
            "helper_noinline":helper_noinline,"helper_descriptors":helper_descriptors,
            "helper_level_capture":helper_level_capture}


def save():
    (root/'results.json').write_text(json.dumps({"target":request['target'],"triton":triton.__file__,"torch":torch.__version__ if torch is not None else None,"results":rows},indent=2))


for name in sys.argv[1:]:
    row={"case":name,"status":"running"};rows[:]=[r for r in rows if r['case']!=name]+[row]
    save();print("BEGIN "+name,flush=True)
    try:
        row.update(status="ok",**operations[name]())
    except Exception as error:
        row.update(status="failed",error=str(error),traceback=traceback.format_exc())
        from compile_triton import save_backend_failures
        row["backend_diagnostics"] = save_backend_failures(error, root)
    save();print("WB_RESULT "+json.dumps(row),flush=True)

sys.exit(any(row["status"] != "ok" for row in rows))

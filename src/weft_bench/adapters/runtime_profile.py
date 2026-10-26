"""Opt-in timings around the installed CompiledKernel's real native launcher."""
import statistics
import time
import ctypes


def paired_jit_calls(invoke, compiled, reference, *, iterations, groups, check):
    abi = reference.metadata.kernel_abi
    if (abi["arguments"] != compiled.metadata.kernel_abi["arguments"] or
            abi["shape_parameters"] != compiled.metadata.kernel_abi["shape_parameters"] or
            reference.metadata.launch_args != compiled.metadata.launch_args or
            reference.metadata.num_cpu_threads != compiled.metadata.num_cpu_threads):
        raise ValueError("same-storage JIT comparison requires identical typed ABI, launch arguments and threads")
    # Both alternatives use the normal JITFunction dispatch and exactly the
    # same storage. Only this instance's loaded native implementation changes.
    launcher = compiled.run
    launches = {
        "candidate": (launcher, compiled.function, compiled.packed_metadata),
        "reference": (reference.run, reference.function, reference.packed_metadata),
    }
    selected = launches["candidate"]

    def paired_launch(gx, gy, gz, stream, function, metadata, *args):
        run, entry, packed = selected
        return run(gx, gy, gz, stream, entry, packed, *args)

    compiled._run = paired_launch
    hot = {name: [] for name in launches}
    try:
        for name in launches:
            selected = launches[name]
            invoke()
            check()
        for group in range(groups):
            order = list(launches) if group % 2 == 0 else list(reversed(launches))
            for name in order:
                selected = launches[name]
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    invoke()
                hot[name].append((time.perf_counter_ns() - start) / iterations / 1e6)
    finally:
        compiled._run = launcher
    return {
        "scope": "same JITFunction and workload storage; alternating loaded implementations through the current runtime",
        "hook": "the same per-instance launcher hook on both implementations",
        "samples_ms": hot,
        "median_ms": {name: statistics.median(values) for name, values in hot.items()},
        "runtime_comparison": False,
    }


def profile_runtime(invoke, compiled, *, iterations, groups, plain_samples, bound, grid, check,
                    reference=None, launches_per_workload=1):
    # CompiledKernel.run resolves this callable on every launch, including the
    # native C++ launcher. Instrument only this kernel instance; keep its ABI
    # validation, worker checks and actual computation in the measured call.
    launcher = compiled.run
    durations, native_calls = 0, 0
    per_call = []

    def measured(*args, **kwargs):
        nonlocal durations, native_calls
        start = time.perf_counter_ns()
        try:
            return launcher(*args, **kwargs)
        finally:
            durations += time.perf_counter_ns() - start
            native_calls += 1

    compiled._run = measured
    try:
        for _ in range(groups):
            for _ in range(iterations):
                start = time.perf_counter_ns()
                invoke()
                per_call.append(time.perf_counter_ns() - start)
    finally:
        compiled._run = launcher

    calls, total = len(per_call), sum(per_call)
    if native_calls != calls * launches_per_workload:
        raise RuntimeError("runtime component profiling did not observe every declared workload launch")
    components = {
        "native_launcher_and_kernel": {
            "mean_ms": durations / calls / 1e6,
            "calls_per_invocation": native_calls / calls,
            "fraction": durations / total,
        },
        "python_dispatch_and_specialization": {
            "mean_ms": (total - durations) / calls / 1e6,
            "calls_per_invocation": 1,
            "fraction": (total - durations) / total,
        },
    }
    profiled_mean = statistics.mean(per_call) / 1e6
    plain_mean = statistics.mean(plain_samples)
    result = {
        "status": "ok",
        "scope": "instrumented complete kernel[grid] calls through the installed native launcher",
        "calls": calls,
        "launches_per_workload": launches_per_workload,
        "instrumented_mean_ms": profiled_mean,
        "instrumented_median_ms": statistics.median(per_call) / 1e6,
        "plain_mean_ms": plain_mean,
        "instrumentation_delta_ms": profiled_mean - plain_mean,
        "grouped_mean_ms": {name: values["mean_ms"] for name, values in components.items()},
        "components": components,
        "hooks": {"native_launcher_and_kernel": "CompiledKernel._run"},
        "notes": [
            "Native launcher time includes ABI preparation, worker capability checks, grid dispatch and kernel execution.",
            "The remaining time covers the ordinary Triton Python dispatch and specialization path.",
            "Component means sum to instrumented time; instrumentation is not subtracted from the plain measurement.",
        ],
    }
    if reference is not None:
        result["paired_jit_calls"] = paired_jit_calls(
            invoke, compiled, reference, iterations=iterations, groups=groups, check=check)
        result["paired_jit_calls"]["launches_per_workload"] = launches_per_workload
    if all(size == 1 for size in grid) and launches_per_workload == 1:
        # Reuse the compiler's ABI binder and the loaded kernel at the very same
        # fixture addresses. This isolates native computation from launch work;
        # it is a diagnostic boundary, not the independent native baseline run.
        from triton.runtime.weft.abi import ABIPlan
        from triton.runtime.weft.native import Error, NativeSignature
        signature = NativeSignature(compiled.metadata.kernel_abi,
                                    bindings=compiled.metadata.source_bindings)
        values = {}
        for argument in signature.arguments:
            if argument.name.startswith("__triton_pid_"):
                values[argument.name] = 0
            elif argument.name.startswith("__triton_grid_"):
                values[argument.name] = 1
            else:
                values[argument.name] = bound[argument.name]
        plan = ABIPlan(signature.arguments, signature.shape_parameters, 64)
        binding = plan.bind(signature.normalize((), values))
        module = compiled.module
        error = Error()
        status = check_worker(module, error)
        if status:
            raise RuntimeError(bytes(error.message).decode())
        entry = ctypes.CFUNCTYPE(None, *plan.ctypes)(module.entry.value)
        entries = {"candidate": entry}
        if reference is not None:
            reference.run  # Load and verify the native module through the product.
            abi = reference.metadata.kernel_abi
            if (abi["arguments"] != compiled.metadata.kernel_abi["arguments"] or
                    abi["shape_parameters"] != compiled.metadata.kernel_abi["shape_parameters"]):
                raise ValueError("same-storage native comparison requires the same typed ABI")
            status = check_worker(reference.module, error)
            if status:
                raise RuntimeError(bytes(error.message).decode())
            entries["reference"] = ctypes.CFUNCTYPE(None, *plan.ctypes)(reference.module.entry.value)
        for function in entries.values():
            function(*binding.arguments)
            check()
        timings = {name: [] for name in entries}
        for group in range(groups):
            order = list(entries) if group % 2 == 0 else list(reversed(entries))
            for name in order:
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    entries[name](*binding.arguments)
                timings[name].append((time.perf_counter_ns() - start) / iterations / 1e6)
        native_samples = timings["candidate"]
        result["prepared_single_instance"] = {
            "scope": "loaded native entry with the original JIT fixture storage; ctypes call overhead included",
            "samples_ms": native_samples,
            "median_ms": statistics.median(native_samples),
            "entry_address": module.entry.value,
            "buffers": [{"name": view.argument.name, "address": view.buffer.address,
                         "bytes": view.span} for view in binding.views],
        }
        if reference is not None:
            result["prepared_single_instance"]["paired_reference"] = {
                "order": "alternating", "samples_ms": timings["reference"],
                "median_ms": statistics.median(timings["reference"]),
                "entry_address": reference.module.entry.value,
            }
    return result


def check_worker(module, error):
    callback = module.sdk.weft_kernel_check_worker
    callback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    callback.restype = ctypes.c_int
    return callback(module.handle, ctypes.addressof(error) + type(error).message.offset, 1024)

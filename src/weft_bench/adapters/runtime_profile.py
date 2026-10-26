"""Opt-in timings around the installed CompiledKernel's real native launcher."""
import statistics
import time
import ctypes


def profile_runtime(invoke, compiled, *, iterations, groups, plain_samples, bound, grid):
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
    if native_calls != calls:
        raise RuntimeError("runtime component profiling requires one native launch per invocation")
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
    if all(size == 1 for size in grid):
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
        check = module.sdk.weft_kernel_check_worker
        check.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        check.restype = ctypes.c_int
        error = Error()
        status = check(module.handle, ctypes.addressof(error) + Error.message.offset, 1024)
        if status:
            raise RuntimeError(bytes(error.message).decode())
        entry = ctypes.CFUNCTYPE(None, *plan.ctypes)(module.entry.value)
        native_samples = []
        for _ in range(groups):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                entry(*binding.arguments)
            native_samples.append((time.perf_counter_ns() - start) / iterations / 1e6)
        result["prepared_single_instance"] = {
            "scope": "loaded native entry with the original JIT fixture storage; ctypes call overhead included",
            "samples_ms": native_samples,
            "median_ms": statistics.median(native_samples),
            "entry_address": module.entry.value,
            "buffers": [{"name": view.argument.name, "address": view.buffer.address,
                         "bytes": view.span} for view in binding.views],
        }
    return result

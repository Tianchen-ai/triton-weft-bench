"""Opt-in timings around real runtime calls; no alternate launch implementation."""
from collections import defaultdict
import hashlib
import inspect
from pathlib import Path
import statistics
import time


def profile_runtime(invoke, compiled, *, iterations, groups, plain_samples):
    from triton.runtime.weft import abi, buffer, native, signature

    sources = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (abi, buffer, native, signature)}
    if not inspect.isclass(native.NativeLauncher):
        return {
            "status": "unavailable",
            "reason": "This runtime constructs a native C++ launcher; Python wrappers cannot observe its internal costs.",
            "plain_mean_ms": statistics.mean(plain_samples),
            "runtime_source_sha256": sources,
        }

    stack, patches, active = [], [], {}
    durations, counts = defaultdict(int), defaultdict(int)
    per_call = []

    def wrap(owner, name, label):
        original = getattr(owner, name, None)
        if original is None:
            return

        def measured(*args, **kwargs):
            frame = [0]
            stack.append(frame)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - start
                stack.pop()
                durations[label] += elapsed - frame[0]
                counts[label] += 1
                if stack:
                    stack[-1][0] += elapsed

        patches.append((owner, name, original))
        setattr(owner, name, measured)
        active[label] = (type(owner).__name__ if not inspect.isclass(owner) else owner.__name__) + "." + name

    # Exclusive nesting avoids counting Buffer construction twice under normalize.
    # The launcher residual contains the actual ctypes pointer vector and glue.
    try:
        wrap(native.NativeLauncher, "__call__", "launcher_marshalling_and_glue")
        wrap(signature.CallSignature, "normalize", "argument_normalization")
        wrap(buffer.Buffer, "__init__", "buffer_borrow")
        wrap(buffer.Buffer, "validate", "storage_validation")
        wrap(abi.ViewABI, "bind", "view_abi_validation")
        wrap(abi.ABIPlan, "bind", "abi_binding")
        wrap(compiled.module, "grid", "native_grid")
        for _ in range(groups):
            for _ in range(iterations):
                start = time.perf_counter_ns()
                invoke()
                per_call.append(time.perf_counter_ns() - start)
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
    calls = len(per_call)
    total = sum(per_call)
    remainder = total - sum(durations.values())
    components = {label: {"mean_ms": elapsed / calls / 1e6,
                           "calls_per_invocation": counts[label] / calls,
                           "fraction": elapsed / total}
                  for label, elapsed in durations.items()}
    components["python_dispatch_and_other"] = {
        "mean_ms": remainder / calls / 1e6, "fraction": remainder / total,
        "calls_per_invocation": 1,
    }
    argument_labels = ("argument_normalization", "buffer_borrow", "storage_validation", "view_abi_validation")
    grouped = {"argument_and_storage": sum(components.get(key, {}).get("mean_ms", 0) for key in argument_labels),
               **{key: components.get(key, {}).get("mean_ms", 0) for key in
                  ("abi_binding", "launcher_marshalling_and_glue", "native_grid", "python_dispatch_and_other")}}
    profiled_mean = total / calls / 1e6
    return {
        "status": "ok",
        "scope": "instrumented complete kernel[grid] calls; checks and native work stay in the call",
        "calls": calls, "instrumented_mean_ms": profiled_mean,
        "instrumented_median_ms": statistics.median(per_call) / 1e6,
        "plain_mean_ms": statistics.mean(plain_samples),
        "instrumentation_delta_ms": profiled_mean - statistics.mean(plain_samples),
        "grouped_mean_ms": grouped, "components": components, "hooks": active,
        "runtime_source_sha256": sources,
        "notes": [
            "Native grid includes ctypes crossing, SDK worker checks and the actual native launch/OpenMP work.",
            "ABI binding includes scalar/alias legality checks as well as constructing ABI values.",
            "Launcher glue includes pointer-vector marshalling, grid checks, dictionaries and output bookkeeping.",
            "Python dispatch residual includes JIT/cache/specialization and instrumentation overhead; no cost is subtracted.",
            "Exclusive component means sum to the instrumented mean, not to the uninstrumented median.",
        ],
    }

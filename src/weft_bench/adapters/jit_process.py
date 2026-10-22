"""Target process: ordinary Triton JIT calls borrowing the native workload."""
import ctypes
import importlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

from .triton_source import load_kernel
from .cpu_target import check_jit_target


SCALARS = {"f32": ctypes.c_float, "f64": ctypes.c_double, "index": ctypes.c_size_t,
           "i1": ctypes.c_bool, "u1": ctypes.c_bool,
           **{prefix + str(bits): getattr(ctypes, "c_" + ctype + str(bits))
              for prefix, ctype in (("i", "int"), ("u", "uint")) for bits in (8, 16, 32, 64)}}


def child_cpu_ms():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (usage.ru_utime + usage.ru_stime) * 1000


def source_kernel(request):
    source = request["source"]
    if request.get("source_module"):
        source = importlib.import_module(request["source_module"]).__file__
    return load_kernel(source, request["symbol"])


def detect(request):
    from triton.runtime.driver import driver
    if platform.machine() != request["architecture"]:
        raise RuntimeError("native JIT requires execution on " + request["architecture"])
    return check_jit_target(request["architecture"], request.get("compile_target"),
                            vars(driver.for_model("weft").get_current_target()))


def describe(request):
    import triton
    from triton.compiler.weft.interface import describe as interface
    kernel = source_kernel(request)
    definition = getattr(kernel, "jit_fn", kernel)
    configurations = getattr(kernel, "configs", None)
    bindings = ([{**config.kwargs, **request["constexprs"]} for config in configurations]
                if configurations else [request["constexprs"]])
    interfaces = [interface(definition, constants, signature=request["signature"]) for constants in bindings]
    if any(candidate["arguments"] != interfaces[0]["arguments"] for candidate in interfaces[1:]):
        raise ValueError("autotune configurations require different caller-visible storage contracts")
    return {"interface": interfaces[0], "target": detect(request), "source": definition.fn.__code__.co_filename,
            "triton_package": triton.__file__, "bench_package": str(Path(__file__).resolve()),
            "cpus": sorted(os.sched_getaffinity(0))}


def run(request):
    preparation_started = time.perf_counter_ns()
    import triton
    import triton.language as tl
    from triton import knobs

    target = detect(request)
    kernel = source_kernel(request)
    fixture = ctypes.CDLL(request["fixture"])
    pointer = ctypes.c_void_p
    launch_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(pointer))
    session_type = ctypes.CFUNCTYPE(ctypes.c_int, pointer)
    fixture.wb_fixture_session.argtypes = [session_type, pointer, ctypes.c_size_t]
    fixture.wb_fixture_invoke.argtypes = [pointer, launch_type, pointer, ctypes.c_size_t]
    fixture.wb_fixture_action.argtypes = [pointer, ctypes.c_int, pointer, ctypes.c_size_t]
    error = ctypes.create_string_buffer(4096)
    phase = "prepare"
    failures = []
    result = {}
    retained = []
    specs = request["invocation"]
    grid = tuple(request["grid"])
    options = {**request["backend_options"], "toolchain": request["toolchain"],
               "num_cpu_threads": request["num_cpu_threads"]}
    buffers = {}
    compilations = []
    knobs.compilation.listener = lambda **event: compilations.append({"cache_hit": event["cache_hit"]})

    def arguments(addresses):
        bound = dict(request["constexprs"])
        for index, spec in enumerate(specs):
            address = addresses[index]
            if spec["kind"] == "scalar":
                bound[spec["name"]] = SCALARS[spec["dtype"]].from_address(address).value
                continue
            key = index, address
            if key not in buffers:
                storage = (ctypes.c_ubyte * spec["bytes"]).from_address(address)
                # Borrow the original addresses. Overlapping parameters remain
                # overlapping; neither input nor state is independently cloned.
                buffers[key] = tl.View(storage, encoding=spec["encoding"],
                                       shape=tuple(spec["shape"]), strides=spec.get("strides"))
            bound[spec["name"]] = buffers[key]
        return bound

    @session_type
    def session(handle):
        nonlocal phase
        try:
            result["source_and_fixture_prepare_ms"] = (time.perf_counter_ns() - preparation_started) / 1e6
            def action(code):
                if fixture.wb_fixture_action(handle, code, error, len(error)):
                    raise ValueError(error.value.decode())

            bound_calls = []

            @launch_type
            def capture(addresses):
                try:
                    bound_calls.append(arguments(addresses))
                    return 0
                except Exception as exception:
                    failures.append(exception)
                    return 1

            # Enumerate the existing workload's row/tile bindings once. This
            # performs no kernel computation and does not synthesize a grid.
            if fixture.wb_fixture_invoke(handle, capture, error, len(error)):
                raise failures.pop() if failures else ValueError(error.value.decode())
            if not bound_calls:
                raise ValueError("the workload produced no Triton invocations")

            def call_arguments(bound):
                signature = getattr(kernel, "jit_fn", kernel).signature
                if hasattr(kernel, "configs"):
                    binding = signature.bind_partial(**bound)
                else:
                    binding = signature.bind(**bound)
                    binding.apply_defaults()
                return binding.args, binding.kwargs

            calls = [call_arguments(bound) for bound in bound_calls]

            def invoke():
                compiled = []
                for args, keywords in calls:
                    compiled.append(kernel[grid](*args, **keywords, **options))
                return compiled

            def timed():
                action(0)
                start = time.perf_counter_ns()
                compiled = invoke()
                return (time.perf_counter_ns() - start) / 1e6, compiled

            phase = "cold_launch"
            child_start = child_cpu_ms()
            cold_ms, first = timed()
            cold_children = child_cpu_ms() - child_start
            retained.extend(first)
            phase = "check"
            action(1)
            phase = "cache_hit_launch"
            compiled_count = len(compilations)
            child_start = child_cpu_ms()
            hit_ms, cached = timed()
            hit_children = child_cpu_ms() - child_start
            if any(a is not b for a, b in zip(first, cached)):
                raise RuntimeError("identical calls did not reuse their CompiledKernel objects")
            if len(compilations) != compiled_count or hit_children != 0:
                raise RuntimeError("cache-hit invocation entered the compiler")
            if len({id(compiled) for compiled in first}) != 1:
                raise NotImplementedError("native comparison needs one specialization for this workload")
            phase = "check"
            action(1)
            phase = "export"
            if request.get("export", True):
                first[0].export(request["artifact"])
            phase = "measure"
            measurement = request["measurement"]
            samples = []
            count = max(1, min(10000, math.ceil(measurement["batch_ms"] / hit_ms)))
            if not measurement["check_only"]:
                for _ in range(measurement["warmup"]):
                    timed()
                for _ in range(measurement["samples"]):
                    if request["reset_each_call"]:
                        samples.append(sum(timed()[0] for _ in range(count)) / count)
                    else:
                        start = time.perf_counter_ns()
                        for _ in range(count):
                            invoke()
                        samples.append((time.perf_counter_ns() - start) / count / 1e6)
            phase = "check"
            action(1)
            if len(compilations) != compiled_count:
                raise RuntimeError("hot invocations re-entered the compiler")
            framework = {}
            if request.get("check_python_aot"):
                phase = "python_aot"
                action(0)
                start = time.perf_counter_ns()
                loaded = triton.load(request["artifact"])
                for bound in bound_calls:
                    loaded[grid](**{name: value for name, value in bound.items() if name not in request["constexprs"]})
                framework["python_aot_load_launch_ms"] = (time.perf_counter_ns() - start) / 1e6
                phase = "check"
                action(1)
                framework["python_aot_correct"] = True
            if request.get("check_torch_compile"):
                import torch
                tensor_calls = []
                for bound in bound_calls:
                    tensors = dict(bound)
                    for spec in specs:
                        if spec["kind"] == "scalar":
                            continue
                        view = bound[spec["name"]]
                        if not view.encoding.startswith("dense."):
                            raise NotImplementedError("Torch capture fixture needs dense Tensor inputs")
                        dtype = {"f32": torch.float32, "f64": torch.float64, "f16": torch.float16,
                                 "bf16": torch.bfloat16, "i1": torch.bool,
                                 "i8": torch.int8, "i16": torch.int16, "i32": torch.int32,
                                 "i64": torch.int64, "u8": torch.uint8}[spec["dtype"]]
                        tensors[spec["name"]] = torch.frombuffer(view.owner, dtype=dtype).as_strided(view.shape, view.strides)
                    tensor_calls.append(tensors)
                phase = "torch_compile"
                def tensor_call():
                    for args, keywords in torch_calls:
                        kernel[grid](*args, **keywords, **options)
                torch_calls = [call_arguments(bound) for bound in tensor_calls]
                compiled_call = torch.compile(tensor_call, fullgraph=True)
                for _ in range(2):
                    action(0)
                    compiled_call()
                    phase = "check"
                    action(1)
                framework["torch_compile_fullgraph_correct"] = True
            if request.get("profile_runtime"):
                from .runtime_profile import profile_runtime
                if request["reset_each_call"] or len(bound_calls) != 1:
                    raise NotImplementedError("runtime component profiling requires a stateless single launch")
                framework["runtime_profile"] = profile_runtime(
                    invoke, first[0], iterations=count, groups=measurement["samples"], plain_samples=samples)
            if request["save_outputs"]:
                action(2)
            result.update(status="ok", samples_ms=samples, iterations=count,
                          cold_compile_and_launch_ms=cold_ms, first_cache_hit_launch_ms=hit_ms,
                          cold_child_cpu_ms=cold_children, cache_hit_child_cpu_ms=hit_children,
                          same_compiled_kernel=True, launches_per_workload=len(bound_calls),
                          correctness=request["correctness"], native_build=first[0].metadata.native_build,
                          compilations=compilations, **framework)
            if hasattr(kernel, "best_config"):
                result["autotune"] = {"selected_kwargs": kernel.best_config.kwargs,
                                      "selected_backend_options": kernel.best_config.backend_options,
                                      "provider": "triton.runtime.autotuner.Autotuner"}
            return 0
        except Exception as exception:
            failures.append(exception)
            return 1

    status = fixture.wb_fixture_session(session, error, len(error))
    if status:
        exception = failures[0] if failures else RuntimeError(error.value.decode())
        from .compile_triton import save_backend_failures
        save_backend_failures(exception, Path(request["artifact"]).parent / "failure")
        import traceback
        traceback.print_exception(type(exception), exception, exception.__traceback__)
        return {"status": "check_failed" if phase == "check" else "run_failed",
                "stage": phase, "error": str(exception), "samples_ms": []}
    result.update(machine=platform.machine(), triton=triton.__version__, triton_package=triton.__file__,
                  bench_package=str(Path(__file__).resolve()), cpus=sorted(os.sched_getaffinity(0)),
                  compile_target=target, target_selection="product-native-detection",
                  artifact=request["artifact"],
                  boundary="Python kernel[grid] calls covering the complete workload; reset excluded")
    return result


def main():
    request = json.loads(Path(sys.argv[-1]).read_text())
    record = describe(request) if "--describe" in sys.argv else run(request)
    print("WB_RESULT " + json.dumps(record), flush=True)
    return int(record.get("status", "ok") != "ok")


if __name__ == "__main__":
    sys.exit(main())

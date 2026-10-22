"""Native-target Python call boundary, borrowing Bench's C++ fixture and oracle."""
import ctypes
import inspect
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

from triton_source import load_kernel
from cpu_target import check_jit_target


def child_cpu_ms():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (usage.ru_utime + usage.ru_stime) * 1000


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    phase = "prepare"
    handle = None
    try:
        if platform.machine() != request["architecture"]:
            raise RuntimeError("native JIT requires execution on " + request["architecture"])
        import torch
        import triton
        from triton.runtime.driver import driver

        detected_target = check_jit_target(request["architecture"], request.get("compile_target"),
                                          vars(driver.for_model("weft").get_current_target()))

        fixture = ctypes.CDLL(request["fixture"])
        pointer = ctypes.c_void_p
        error = ctypes.create_string_buffer(2048)
        fixture.wb_fixture_create.argtypes = [pointer, ctypes.c_size_t]
        fixture.wb_fixture_create.restype = pointer
        fixture.wb_fixture_release.argtypes = [pointer]
        fixture.wb_fixture_buffer.argtypes = [pointer, ctypes.c_char_p]
        fixture.wb_fixture_buffer.restype = pointer
        fixture.wb_fixture_check.argtypes = [pointer, pointer, ctypes.c_size_t, pointer, ctypes.c_size_t]
        handle = fixture.wb_fixture_create(error, len(error))
        if not handle:
            raise RuntimeError(error.value.decode())
        buffers = {}
        for port in request["ports"]:
            address = fixture.wb_fixture_buffer(handle, port["name"].encode())
            if not address:
                raise ValueError("fixture has no buffer " + port["name"])
            storage = (ctypes.c_float * math.prod(port["shape"])).from_address(address)
            # Caller tensors are separate storage; the shared oracle retains its original inputs.
            buffers[port["name"]] = torch.frombuffer(storage, dtype=torch.float32).reshape(port["shape"]).clone()
        kernel = load_kernel(request["source"], request["symbol"])
        constexprs = request["constexprs"]
        arguments = {}
        for name in kernel.arg_names:
            role = request["arguments"].get(name, name)
            if name in constexprs:
                arguments[name] = constexprs[name]
            elif role in buffers:
                arguments[name] = buffers[role]
            elif role in request["parameters"]:
                arguments[name] = request["parameters"][role]
            elif role in request["shape"]:
                arguments[name] = request["shape"][role]
            elif kernel.signature.parameters[name].default is inspect.Parameter.empty:
                raise ValueError("no workload binding for " + name)
        launch_options = {**request["backend_options"], "toolchain": request["toolchain"],
                          "num_cpu_threads": request["num_cpu_threads"]}

        def invoke():
            return kernel[tuple(request["grid"])](**arguments, **launch_options)

        def check():
            output = buffers[request["output_port"]]
            if fixture.wb_fixture_check(handle, output.data_ptr(), output.numel() * output.element_size(), error, len(error)):
                raise ValueError("numeric check: " + error.value.decode())

        phase = "cold_launch"
        child_start = child_cpu_ms()
        start = time.perf_counter_ns()
        first = invoke()
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        cold_children = child_cpu_ms() - child_start
        phase = "check"
        check()
        phase = "cache_hit_launch"
        child_start = child_cpu_ms()
        start = time.perf_counter_ns()
        cached = invoke()
        hit_ms = (time.perf_counter_ns() - start) / 1e6
        hit_children = child_cpu_ms() - child_start
        if cached is not first:
            raise RuntimeError("identical launch did not reuse its CompiledKernel object")
        phase = "check"
        check()
        phase = "export"
        first.export(request["artifact"])
        framework = {}
        if request.get("check_torch_compile"):
            phase = "torch_compile"

            def tensor_call():
                kernel[tuple(request["grid"])](**arguments, **launch_options)
                return buffers[request["output_port"]]

            start = time.perf_counter_ns()
            compiled_call = torch.compile(tensor_call, fullgraph=True)
            compiled_call()
            framework["torch_compile_first_call_ms"] = (time.perf_counter_ns() - start) / 1e6
            phase = "check"
            check()
            phase = "torch_compile_cached"
            compiled_call()
            phase = "check"
            check()
            framework["torch_compile_fullgraph_correct"] = True
            framework["torch_package"] = torch.__file__
        phase = "measure"
        measurement = request["measurement"]
        for _ in range(measurement["warmup"]):
            invoke()
        count = max(1, min(10000, math.ceil(measurement["batch_ms"] / hit_ms)))
        samples = []
        for _ in range(measurement["samples"]):
            start = time.perf_counter_ns()
            for _ in range(count):
                invoke()
            samples.append((time.perf_counter_ns() - start) / (count * 1e6))
        profile = {}
        if request.get("profile_runtime"):
            phase = "runtime_profile"
            from runtime_profile import profile_runtime
            profile = profile_runtime(invoke, first, iterations=count, groups=measurement["samples"],
                                      plain_samples=samples)
            phase = "check"
            check()
        aot = {}
        if request["check_python_aot"]:
            phase = "python_aot"
            child_start = child_cpu_ms()
            start = time.perf_counter_ns()
            loaded = triton.load(request["artifact"])
            loaded[tuple(request["grid"])](**{k: v for k, v in arguments.items() if k not in constexprs})
            aot = {"python_aot_load_launch_ms": (time.perf_counter_ns() - start) / 1e6,
                   "python_aot_child_cpu_ms": child_cpu_ms() - child_start}
            phase = "check"
            check()
            aot["python_aot_correct"] = True
            phase = "python_aot_measure"
            aot_arguments = {k: v for k, v in arguments.items() if k not in constexprs}
            def invoke_aot():
                loaded[tuple(request["grid"])](**aot_arguments)
            for _ in range(measurement["warmup"]):
                invoke_aot()
            start = time.perf_counter_ns()
            invoke_aot()
            estimate_ms = max((time.perf_counter_ns() - start) / 1e6, 1e-9)
            aot_count = max(1, min(10000, math.ceil(measurement["batch_ms"] / estimate_ms)))
            aot_samples = []
            for _ in range(measurement["samples"]):
                start = time.perf_counter_ns()
                for _ in range(aot_count):
                    invoke_aot()
                aot_samples.append((time.perf_counter_ns() - start) / aot_count / 1e6)
            aot["python_aot_hot_samples_ms"] = aot_samples
            aot["python_aot_hot_iterations"] = aot_count
            phase = "check"
            check()
        if request["save_outputs"]:
            output = buffers[request["output_port"]]
            Path("Y.bin").write_bytes(ctypes.string_at(output.data_ptr(), output.numel() * output.element_size()))
        runtime_libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                    if any(name in line for name in ("libtriton.so", "libweft_runtime.so", "libomp.so"))})
        print("WB_RESULT " + json.dumps({
            "status": "ok", "samples_ms": samples, "iterations": count,
            "cold_compile_and_launch_ms": cold_ms, "first_cache_hit_launch_ms": hit_ms,
            "cold_child_cpu_ms": cold_children, "cache_hit_child_cpu_ms": hit_children,
            "same_compiled_kernel": True, "correctness": request["correctness"],
            "machine": platform.machine(), "torch": torch.__version__, "triton": triton.__version__,
            "tensor_device": "cpu", "triton_package": triton.__file__,
            "native_build": first.metadata.native_build, "loaded_runtime_libraries": runtime_libraries,
            "compile_target": detected_target, "target_selection": "product-native-detection",
            "runtime_profile": profile,
            "artifact": request["artifact"], "boundary": "Python kernel[grid] including binding, guards and native launch",
            **aot, **framework,
        }), flush=True)
    except Exception as exception:
        import traceback
        traceback.print_exc()
        print("WB_RESULT " + json.dumps({"status": "check_failed" if phase == "check" else "run_failed",
              "stage": phase, "error": str(exception), "samples_ms": []}), flush=True)
        return 1
    finally:
        if handle:
            fixture.wb_fixture_release(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Selected native framework contracts; results are integration evidence, not timings."""
import importlib
import json
import math
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
import triton

root = Path(os.environ.get("WEFT_BENCH_FRAMEWORK_OUTPUT", Path(__file__).parent))
root.mkdir(parents=True, exist_ok=True)
request = json.loads((Path(__file__).parent / "request.json").read_text())
rows = json.loads((root / "results.json").read_text())["results"] if (root / "results.json").exists() else []


def native_options():
    toolchain = json.loads((Path(__file__).parent / "toolchain.json").read_text())
    return {"toolchain": toolchain, "num_cpu_threads": 2}


def close(actual, expected):
    if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise ValueError("numerical mismatch: max_abs=" + str((actual - expected).abs().max().item()))


def record(name, operation):
    try:
        details = operation()
        row = {"case": name, "status": "ok", **details}
    except Exception as error:
        row = {"case": name, "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        from compile_triton import save_backend_failures
        row["backend_diagnostics"] = save_backend_failures(error, root, name + "-failure")
    rows[:] = [previous for previous in rows if previous["case"] != name] + [row]
    (root / "results.json").write_text(json.dumps({"target": request["target"], "torch": torch.__version__,
        "triton": triton.__version__, "results": rows}, indent=2))
    print("WB_RESULT " + json.dumps(row), flush=True)


def strided():
    affine = importlib.import_module("kernels.affine_strided").affine_strided
    options = native_options()
    def call(x, y):
        affine[(1,)](x, y, 1.25, bias=-0.5, B=64, **options)
        return y

    captured = torch.compile(call, fullgraph=True, dynamic=True)
    details = []
    for m, n in ((3, 65), (5, 97)):
        storage = torch.arange(m * (n * 2 + 3), dtype=torch.float32).reshape(m, n * 2 + 3) / 2048
        x = storage[:, 1:1 + 2*n:2]
        y_storage = torch.full((m, n*3+2), -7.0)
        y = y_storage[:, 1:1 + 3*n:3]
        expected = x * 1.25 - 0.5
        call(x, y)
        close(y, expected)
        y.fill_(-7)
        captured(x, y)
        close(y, expected)
        guard = torch.ones_like(y_storage, dtype=torch.bool)
        guard[:, 1:1+3*n:3] = False
        if not torch.all(y_storage[guard] == -7):
            raise ValueError("strided output overwrote padding")
        details.append({"shape": list(x.shape), "x_stride": list(x.stride()), "y_stride": list(y.stride()),
                        "x_offset": x.storage_offset(), "y_offset": y.storage_offset()})
    return {"eager": True, "fullgraph_dynamic": True, "scenarios": details}


def tune_alias():
    from compile_triton import save_backend_failures
    state_update = importlib.import_module("kernels.state_update").state_update
    options = native_options()
    configs = [triton.Config({"B": 32}, backend_options={"lmul_eighths": 8}),
               triton.Config({"B": 64}, backend_options={"lmul_eighths": 16}),
               triton.Config({"B": 256}, backend_options={"lmul_eighths": 64,
                             "pointwise_issue_copies": 4, "private_stack_bytes": 1})]
    tuned = triton.autotune(configs=configs, key=[])(state_update)
    records = []
    for m, n in ((3, 65), (5, 97)):
        base = torch.arange(m * (n + 1), dtype=torch.float32).reshape(m, n + 1) / 1000
        initial = base.clone()
        state, delta = base[:, :-1], base[:, 1:]
        cache_before = len(tuned.cache)
        tuned[(1,)](state, delta, 0.75, **options)
        expected = initial.clone()
        expected[:, :-1] = initial[:, :-1] * 0.75 + initial[:, 1:]
        close(base, expected)
        cache_after = len(tuned.cache)
        if cache_after != cache_before + 1:
            raise ValueError("changed shape reused an unrelated tuning-cache entry")
        rejected = [{"config": c.__dict__, "error": str(error),
                     "candidate_rejected": error.candidate_rejected,
                     "backend_diagnostics": save_backend_failures(error, root, f"autotune-{m}x{n}-rejection-{index}")}
                    for index, (c, error) in enumerate(tuned.candidate_errors.items())]
        legal = [c.__dict__ for c, timing in tuned.configs_timings.items() if math.isfinite(timing[0])]
        if len(legal) != 2 or len(rejected) != 1 or "resource" not in rejected[0]["error"].lower():
            raise ValueError("expected two legal candidates and one real resource rejection: " + repr((legal, rejected)))
        base.copy_(initial)
        tuned[(1,)](state, delta, 0.75, **options)
        close(base, expected)
        if len(tuned.cache) != cache_after:
            raise ValueError("identical shape missed its tuning-cache entry")
        records.append({"shape": [m,n], "shared_storage": True, "relative_offset": 1,
                        "caller_updated_once": True, "cached_call_updated_once": True,
                        "cache_entries": cache_after, "winner": tuned.best_config.__dict__,
                        "legal": legal, "rejected": rejected})
    return {"scenarios": records}


def backward():
    module = importlib.import_module("kernels.torch_affine").Affine(1.25, -0.5)
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    records = []
    for label, call in (("eager", module), ("fullgraph", compiled)):
        for shape in ((3, 65), (5, 97)):
            x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).div(2048).requires_grad_()
            out = call(x)
            close(out, x.detach() * 1.25 - 0.5)
            out.sum().backward()
            close(x.grad, torch.full_like(x, 1.25))
            records.append({"mode": label, "shape": list(shape), "forward": True,
                            "backward": True, "gradient_source": "sum creates zero-stride read-only gradient"})
    return {"scenarios": records}


@contextmanager
def observe_inductor_tuning(enabled):
    """Record the existing tuner's decisions without changing candidates or selection."""
    if not enabled:
        yield
        return
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from compile_triton import save_backend_failures

    original_compile = CachingAutotuner._precompile_config
    original_bench = CachingAutotuner.bench
    owners = {}
    evidence = {"candidates": [], "timings": [], "remaining_launchers": []}

    def config(cfg):
        return {key: getattr(cfg, key, None) for key in
                ("kwargs", "num_warps", "num_stages", "backend_options", "num_cpu_threads")}

    def precompile(owner, cfg, *args, **kwargs):
        owners[id(owner)] = owner
        row = {"kernel": owner.fn.__name__, "config": config(cfg)}
        evidence["candidates"].append(row)
        try:
            result = original_compile(owner, cfg, *args, **kwargs)
            row["status"] = "compiled"
            return result
        except Exception as error:
            row.update(status="rejected" if getattr(error, "candidate_rejected", False) else "failed",
                       error=str(error), backend_diagnostics=save_backend_failures(
                           error, root, "inductor-candidate-" + str(len(evidence["candidates"]))))
            raise

    def bench(owner, launcher, *args, **kwargs):
        elapsed = original_bench(owner, launcher, *args, **kwargs)
        evidence["timings"].append({"kernel": owner.fn.__name__, "config": config(launcher.config),
                                    "ms": elapsed if math.isfinite(elapsed) else None})
        return elapsed

    CachingAutotuner._precompile_config = precompile
    CachingAutotuner.bench = bench
    try:
        yield
    finally:
        CachingAutotuner._precompile_config = original_compile
        CachingAutotuner.bench = original_bench
        evidence["remaining_launchers"] = [
            {"kernel": owner.fn.__name__, "configs": [config(launcher.config) for launcher in owner.launchers]}
            for owner in owners.values()]
        (root / "inductor-tuning.json").write_text(json.dumps(evidence, indent=2))


def automatic_inductor(name):
    from torch._inductor.utils import run_and_get_code

    torch.set_num_threads(2)
    compiler_options = {"cpu_backend": "triton"}
    if name == "pointwise":
        operation = lambda x: x * 1.25 - 0.5
        arguments = (torch.arange(131, dtype=torch.float32) / 512,)
    elif name in ("matmul", "matmul_bias_relu", "mlp"):
        compiler_options.update(max_autotune_gemm=True, max_autotune_gemm_backends="TRITON")
        if name == "mlp":
            operation = lambda x, w1, b1, w2, b2: torch.relu(x @ w1 + b1) @ w2 + b2
            arguments = ((torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32) - 256) / 512,
                         (torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) % 31 - 15) / 64,
                         torch.linspace(-0.5, 0.5, 32),
                         (torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) % 17 - 8) / 64,
                         torch.linspace(-0.25, 0.25, 16))
        else:
            arguments = (torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32) / 512,
                         torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) / 1024)
            if name == "matmul":
                operation = lambda left, right: torch.mm(left, right)
            else:
                operation = lambda left, right, bias: torch.relu(torch.mm(left, right) + bias)
                arguments += (torch.linspace(-2.0, 2.0, 16),)
    else:
        width = 4096 if name == "norm4096" else 117
        x = torch.arange(4 * width, dtype=torch.float32).reshape(4, width) / 512
        if name == "broadcast":
            operation = lambda x, bias: x + bias
            arguments = (x, torch.linspace(-0.5, 0.5, 117))
        elif name == "reduction":
            operation = lambda x: x.sum(dim=1)
            arguments = (x,)
        elif name in ("norm", "norm4096"):
            operation = lambda x, weight, bias: torch.nn.functional.layer_norm(x, (width,), weight, bias, eps=1e-5)
            arguments = (x, torch.linspace(0.75, 1.25, width), torch.linspace(-0.25, 0.25, width))
        else:
            raise ValueError("unknown automatic Inductor workload " + name)
    expected = operation(*arguments)
    captured = torch.compile(operation, backend="inductor", fullgraph=True, options=compiler_options)
    with observe_inductor_tuning(name == "norm4096"):
        output, codes = run_and_get_code(captured, *arguments)
        close(output, expected)
        close(captured(*arguments), expected)
    paths = []
    has_weft = False
    for index, code in enumerate(codes):
        path = root / ("inductor-" + name + "-" + str(index) + ".py")
        path.write_text(code)
        paths.append(str(path))
        has_weft |= "async_compile.triton" in code and ("model='weft'" in code or 'model="weft"' in code)
    if not has_weft:
        raise ValueError("numerical output alone does not prove automatic Weft lowering; generated Inductor code has no Weft Triton kernel")
    if name in ("matmul", "matmul_bias_relu", "mlp") and (
                              sum(code.count("tl.dot(") for code in codes) < (2 if name == "mlp" else 1) or
                              any("extern_kernels.mm(" in code or "extern_kernels.addmm(" in code for code in codes)):
        raise ValueError("matmul was not fully generated by the Triton template")
    return {"backend": "inductor", "cpu_backend": "triton", "author_code": "ordinary torch operations",
            "fullgraph": True, "first_and_cached_call": True, "generated_weft_kernel": True,
            "generated_code": paths, "compiler_options": compiler_options, "input_shapes": [list(x.shape) for x in arguments],
            "max_abs_error": float((output - expected).abs().max())}


choices = {"strided": strided, "autotune_alias": tune_alias, "backward": backward,
           **{"inductor_" + name: (lambda selected=name: automatic_inductor(selected))
              for name in ("pointwise", "broadcast", "reduction", "norm", "norm4096", "matmul", "matmul_bias_relu", "mlp")}}
def missing_capability(name):
    if name == "autotune_alias":
        return None
    from torch._dynamo import device_interface
    if not hasattr(device_interface, "CpuInterface"):
        return "installed Torch has no torch._dynamo.device_interface.CpuInterface"
    if name.startswith("inductor_"):
        from torch._inductor import config
        if "cpu_backend" not in config._config:
            return "installed Torch has no Inductor cpu_backend option"
    return None


for name in sys.argv[1:] or ("strided", "autotune_alias", "backward"):
    missing = missing_capability(name)
    if missing:
        row = {"case": name, "status": "unsupported", "stage": "framework_capability",
               "error": missing, "torch": torch.__version__, "checked": False}
        rows.append(row)
        print("WB_RESULT " + json.dumps(row), flush=True)
        (root / "results.json").write_text(json.dumps({"results": rows}, indent=2))
    else:
        record(name, choices[name])
sys.exit(any(row["status"] != "ok" for row in rows))

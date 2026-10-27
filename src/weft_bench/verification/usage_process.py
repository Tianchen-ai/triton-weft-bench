"""Run the real usage example and inspect its returned frame, then load its export."""
import json
from pathlib import Path
import runpy
import sys
import traceback


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    # Author imports are explicit even when inherited PYTHONPATH is cleared.
    sys.path.insert(0, str(Path(request["source"]).resolve().parent))
    if request.get("contract", "use") != "use":
        return catalog_example(request)
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(request["source"]).resolve()
    result = {"checked": False, "source": str(source)}
    captured = {}
    previous = sys.getprofile()
    stage = "example"

    def returned(frame, event, argument):
        if event == "return" and frame.f_code.co_name == "main" and Path(frame.f_code.co_filename).resolve() == source:
            captured.update({name: frame.f_locals[name] for name in ("x", "y", "compiled") if name in frame.f_locals})

    try:
        import numpy as np
        import triton

        artifact = output / "artifact"
        size = request.get("size", 4097)
        sys.argv = [str(source), "--size", str(size), "--threads", str(request.get("threads", 2)),
                    "--toolchain", request["toolchain_file"], "--output", str(artifact)]
        sys.setprofile(returned)
        try:
            runpy.run_path(str(source), run_name="__main__")
        finally:
            sys.setprofile(previous)
        if set(captured) != {"x", "y", "compiled"}:
            raise ValueError("usage example no longer exposes its input/output/compiled result; update its observation contract")
        x, y = captured["x"], captured["y"]
        stage = "numerical"
        original = np.arange(size, dtype=np.float32)
        np.testing.assert_array_equal(x, original)
        expected = original * np.float32(1.25) - np.float32(0.5)
        np.testing.assert_allclose(y, expected, atol=2e-5, rtol=2e-5)
        np.savez(output / "example-output.npz", X=x, Y=y)
        stage = "artifact_load"
        loaded = triton.load(artifact)
        y.fill(np.nan)
        stage = "artifact_launch"
        loaded[(triton.cdiv(size, 64),)](x, y, 1.25, bias=-0.5)
        stage = "numerical"
        np.testing.assert_array_equal(x, original)
        np.testing.assert_allclose(y, expected, atol=2e-5, rtol=2e-5)
        np.savez(output / "aot-output.npz", Y=y)
        result.update(status="passed", checked=True, elements=int(y.size),
                      real_example_executed=True, exported_artifact_reloaded=True,
                      triton_version=triton.__version__, triton_package=triton.__file__)
    except Exception as error:
        from compile_triton import save_backend_failures
        result.update(status="numerical_failed" if stage == "numerical" else stage + "_failed",
                      error=str(error), stage=stage, traceback=traceback.format_exc(),
                      backend_diagnostics=save_backend_failures(error, output))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("WB_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


def catalog_example(request):
    """Observe the real example's main frame, then check complete outputs."""
    import numpy as np

    source = Path(request["source"]).resolve()
    output = Path(request["output"]); output.mkdir(parents=True, exist_ok=True)
    contract = request["contract"]
    result = {"source": str(source), "contract": contract, "checked": False}
    captured = {}
    previous = sys.getprofile()
    stage = "example"

    def returned(frame, event, argument):
        if event == "return" and frame.f_code.co_name == "main" and Path(frame.f_code.co_filename).resolve() == source:
            for name in ("x", "y", "a", "b", "indices", "matrix", "compiled", "loaded", "tuned", "block",
                         "logical", "storage"):
                if name in frame.f_locals:
                    captured[name] = frame.f_locals[name]

    def exact(actual, expected):
        np.testing.assert_array_equal(actual, expected)

    def close(actual, expected):
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("nonfinite example output")
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    try:
        import triton
        artifact = output / "artifact"
        if contract == "triton_basics":
            arguments = ["--operation", request["operation"], "--threads", str(request["threads"])]
        elif contract == "autotune_export":
            arguments = [str(artifact), "--size", str(request["size"]), "--threads", str(request["threads"])]
        elif contract in ("pytorch_cpu", "encoding", "levels"):
            arguments = []
        else:
            raise ValueError("unknown usage observation contract " + contract)
        sys.argv = [str(source), *arguments]
        sys.setprofile(returned)
        try:
            runpy.run_path(str(source), run_name="__main__")
        finally:
            sys.setprofile(previous)
        stage = "numerical"
        if contract == "triton_basics":
            operation = request["operation"]
            y = captured["y"]
            if operation == "affine":
                x = np.linspace(-1, 1, 131, dtype=np.float32)
                exact(captured["x"], x); expected = x.astype(np.float64)*1.25-0.5
            elif operation == "softmax":
                x = np.linspace(-2, 2, 4*117, dtype=np.float32).reshape(4, 117)
                exact(captured["x"], x)
                exponent = np.exp(x.astype(np.float64)-x.max(axis=1, keepdims=True))
                expected = exponent/exponent.sum(axis=1, keepdims=True)
            elif operation == "matmul":
                a = np.linspace(-1, 1, 16*32, dtype=np.float32).reshape(16, 32)
                b = np.linspace(-0.5, 0.5, 32*16, dtype=np.float32).reshape(32, 16)
                exact(captured["a"], a); exact(captured["b"], b)
                expected = a.astype(np.float64) @ b.astype(np.float64)
            elif operation == "scan":
                x = np.arange(4*31, dtype=np.int32).reshape(4, 31)%7-3
                exact(captured["x"], x); expected = np.cumsum(x, axis=1, dtype=np.int32)
            elif operation == "histogram":
                indices = np.arange(131, dtype=np.int32)%16
                exact(captured["indices"], indices); expected = np.bincount(indices, minlength=16).astype(np.int32)
            else:
                raise ValueError("unconfigured basic operation " + operation)
            (exact if operation in ("scan", "histogram") else close)(y, expected)
            np.savez(output / "example-output.npz", Y=y, expected=expected)
            result.update(operation=operation, elements=int(y.size))
        elif contract == "autotune_export":
            x, y = captured["x"], captured["y"]
            expected_x = np.linspace(-1, 1, request["size"], dtype=np.float32)
            exact(x, expected_x); expected = expected_x.astype(np.float64)*1.25-0.5
            close(y, expected)
            block = int(captured["block"])
            chosen = int(captured["tuned"].best_config.kwargs["B"])
            if block != chosen or not (artifact / "manifest.json").is_file():
                raise ValueError("example did not retain the selected exported artifact")
            y.fill(np.nan)
            stage = "artifact_launch"
            captured["loaded"][(triton.cdiv(x.size, block),)](x, y, x.size, 1.25, -0.5)
            stage = "numerical"; close(y, expected); exact(x, expected_x)
            np.savez(output / "example-output.npz", X=x, Y=y)
            result.update(elements=int(y.size), selected_block=block, exported_artifact_reloaded=True,
                          tuning_candidates=len(captured["tuned"].configs))
        elif contract == "encoding":
            logical = (np.arange(64, dtype=np.uint8) % 16).reshape(2, 32)
            exact(captured["logical"], logical)
            exact(captured["storage"], logical[:, :16] | (logical[:, 16:] << 4))
            exact(captured["y"], logical.reshape(-1))
            result.update(elements=logical.size, integer_exact=True,
                          explicit_packing_checked=True)
        elif contract == "levels":
            x = np.linspace(-1, 1, 4 * 65, dtype=np.float32).reshape(4, 65)
            exact(captured["x"], x)
            close(captured["y"], (x.astype(np.float64) ** 2).sum(axis=1))
            result.update(elements=int(x.size), partial_last_block=True)
        else:
            import torch
            x, y, matrix = captured["x"], captured["y"], captured["matrix"]
            expected_x = torch.linspace(-1, 1, 131, dtype=torch.float32).numpy()
            expected_matrix = torch.linspace(-2, 2, 4*64, dtype=torch.float32).reshape(4, 64).numpy()
            exact(x.detach().numpy(), expected_x); exact(matrix.detach().numpy(), expected_matrix)
            close(y.detach().numpy(), expected_x.astype(np.float64)*1.25-0.5)
            stage = "framework_launch"
            actual = captured["compiled"](matrix).detach().numpy()
            centered = expected_matrix.astype(np.float64)-expected_matrix.astype(np.float64).mean(axis=1, keepdims=True)
            expected = centered/np.sqrt((centered*centered).mean(axis=1, keepdims=True)+1e-5)
            stage = "numerical"; close(actual, expected)
            generated = []
            for path in (output / "inductor-cache").rglob("*.py"):
                code = path.read_text(errors="replace")
                if "async_compile.triton" in code and ("model='weft'" in code or 'model="weft"' in code):
                    generated.append(str(path.relative_to(output)))
            if not generated:
                raise ValueError("example output is correct but generated Weft Inductor code was not observed")
            np.savez(output / "example-output.npz", X=x.numpy(), Y=y.numpy(), normalized=actual)
            result.update(elements=int(y.numel()+actual.size), generated_weft_code=generated,
                          cached_compiled_full_output_checked=True, torch_version=torch.__version__)
        result.update(status="passed", checked=True, real_example_executed=True,
                      triton_version=triton.__version__, triton_package=triton.__file__)
    except Exception as error:
        from compile_triton import save_backend_failures
        result.update(status="numerical_failed" if stage == "numerical" else stage + "_failed",
                      stage=stage, error=str(error), traceback=traceback.format_exc(),
                      backend_diagnostics=save_backend_failures(error, output))
    (output / "result.json").write_text(json.dumps(result, indent=2)+'\n')
    print("WB_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Execute a prepared native Program artifact through the public Triton loader."""
import json
from pathlib import Path
import platform
import sys
import traceback


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    stage = "prepare"
    result = {"case": request["case"], "checked": False, "measurement": "correctness-only"}
    try:
        import numpy as np
        import triton
        from program_oracles import prepare

        result.update(triton_version=triton.__version__, triton_package=triton.__file__, machine=platform.machine())
        scenario = prepare(request["case"], request.get("bindings"), request.get("seed", 410))
        arguments = scenario.arguments
        reference = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arguments.items()}
        np.savez(output / "inputs.npz", **{k: v for k, v in arguments.items() if isinstance(v, np.ndarray)})
        result.update(properties=list(scenario.properties),
                      inputs={k: {"shape": list(v.shape), "dtype": str(v.dtype)} if isinstance(v, np.ndarray) else v
                              for k, v in arguments.items()})
        stage = "artifact_load"
        manifest = json.loads((Path(request["artifact"]) / "manifest.json").read_text())
        result.update(required_extensions=manifest["kernel"].get("required_extensions", []),
                      used_extensions=manifest["kernel"].get("used_extensions", []))
        compiled = triton.load(request["artifact"])
        launch_arguments = dict(arguments)
        encoding_module = None
        if scenario.encodings:
            import triton.language as tl
            import importlib.util

            for name, (encoding_name, shape) in scenario.encodings.items():
                base, _, parameters = encoding_name.partition("[")
                encoding = getattr(tl, base, None)
                if encoding is None:
                    if encoding_module is None:
                        spec = importlib.util.spec_from_file_location("bench_author_encodings", request["encoding_source"])
                        encoding_module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = encoding_module
                        spec.loader.exec_module(encoding_module)
                    encoding = encoding_module
                    for member in base.split("."):
                        encoding = getattr(encoding, member)
                if parameters:
                    encoding = encoding[int(parameters.removesuffix("]"))]
                launch_arguments[name] = tl.View(arguments[name], encoding=encoding, shape=shape)
        checks = []
        result["checks"] = checks
        for index in range(scenario.calls):
            stage = "oracle"
            expected = scenario.oracle(reference)
            if set(expected) != set(scenario.outputs):
                raise ValueError("oracle outputs differ from the declared mutable arguments")
            stage = "artifact_launch"
            compiled[tuple(request.get("grid", [1]))](**launch_arguments)
            np.savez(output / ("outputs-" + str(index + 1) + ".npz"), **{k: arguments[k] for k in scenario.outputs})
            stage = "numerical"
            for key, actual in arguments.items():
                if not isinstance(actual, np.ndarray):
                    continue
                wanted = expected.get(key, reference[key])
                exact = key not in expected or actual.dtype.kind in "biu" or key in scenario.exact_outputs
                record = {"call": index + 1, "argument": key, "elements": int(actual.size), "exact": exact}
                checks.append(record)
                if exact:
                    np.testing.assert_array_equal(actual, wanted, err_msg=key)
                else:
                    if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
                        raise ValueError("nonfinite Program output " + key)
                    record["max_abs_error"] = float(np.abs(actual.astype(np.float64) - wanted).max(initial=0))
                    # Same defaults as wb::close in the existing native worker.
                    np.testing.assert_allclose(actual, wanted, atol=1e-4, rtol=2e-3, err_msg=key)
                record["passed"] = True
            for key, wanted in expected.items():
                reference[key] = np.asarray(wanted, dtype=arguments[key].dtype)
        result.update(status="passed", checked=True, calls=scenario.calls,
                      output_elements=sum(arguments[k].size for k in scenario.outputs), atol=1e-4, rtol=2e-3)
    except Exception as error:
        result.update(status="numerical_failed" if stage == "numerical" else stage + "_failed",
                      error=str(error), stage=stage, traceback=traceback.format_exc())
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("WB_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

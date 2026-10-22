"""Installed CPU wheel checks through the normal Bench verification worker."""
import array
from importlib.metadata import version
import json
import os
from pathlib import Path
import traceback


def main():
    root = Path(os.environ["WEFT_BENCH_VERIFY_OUTPUT"])
    request = json.loads(Path("request.json").read_text())
    result = {"case": "distribution", "status": "running", "checked": False}
    try:
        import triton
        import triton.language as tl
        from triton.backends.weft.driver import WeftDriver
        from kernels import affine, coordinates

        package = Path(triton.__file__).resolve()
        if "site-packages" not in package.parts or request.get("forbidden_source", "<unset>") in str(package):
            raise ValueError("distribution verification requires a wheel installation without a source overlay")
        target = WeftDriver().get_current_target()
        x = array.array("f", (i / 8 for i in range(131)))
        y = array.array("f", [-99] * 131)
        xv = tl.View(x, encoding=tl.f32, shape=(1, len(x)))
        yv = tl.View(y, encoding=tl.f32, shape=(1, len(y)))
        kernel = affine[(1,)](xv, yv, num_cpu_threads=1)
        assert all(actual == value * 1.25 - .5 for actual, value in zip(y, x))
        assert affine[(1,)](xv, yv, num_cpu_threads=1) is kernel
        export = root / "artifact"
        kernel.export(export)
        loaded = triton.load(export)
        y[:] = array.array("f", [-99] * len(y))
        loaded[(1,)](xv, yv)
        assert all(actual == value * 1.25 - .5 for actual, value in zip(y, x))
        grid = (2, 3, 2)
        storage = array.array("f", [-1] * 12)
        view = tl.View(storage, encoding=tl.f32, shape=grid)
        coordinates[grid](view, num_cpu_threads=2)
        expected = [i + 10*j + 100*k + 1000*2 + 10000*3 + 100000*2
                    for i in range(2) for j in range(3) for k in range(2)]
        assert list(storage) == expected
        toolchain = kernel.metadata.native_build["toolchain"]
        result.update(status="ok", checked=True, triton_distribution=version("triton"),
                      bench_distribution=version("triton-weft-bench"), triton_package=str(package),
                      target=vars(target), cpus=sorted(os.sched_getaffinity(0)),
                      jit="exact", warm_reuse=True, aot_reload="exact", grid_openmp="exact",
                      native_build=kernel.metadata.native_build, compiler=toolchain["cc"])
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("WB_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

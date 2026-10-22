"""Native library calls; the existing workload, worker and timer own execution."""

import json
from pathlib import Path

from ..core import NativeCall, PreparedNative


def library_configuration(target, options, default):
    key = options.get("library", default)
    configuration = target.settings.get("references", {}).get(key)
    if not configuration or not configuration.get("include") or not configuration.get("library"):
        raise ValueError(f"target.references.{key} needs include and library paths")
    includes = configuration["include"]
    includes = [includes] if isinstance(includes, str) else includes
    library = configuration["library"]
    if not Path(library).is_absolute():
        raise ValueError("reference library must be an absolute target-side library path")
    return configuration, ["-I" + str(p) for p in includes], [library, "-Wl,-rpath," + str(Path(library).parent)]


class OpenBLASAdapter:
    def prepare_native(self, case, target, options, directory, executor, workload, name):
        if case.name != "gemm_f32":
            raise NotImplementedError("OpenBLAS reference implements the gemm_f32 workload")
        configuration, includes, links = library_configuration(target, options, "openblas")
        threads = target.settings.get("threads", 1)
        call = NativeCall(name,
            f"OpenBLASReference reference({workload.buffers['X']},{workload.buffers['W']},"
            f"{workload.buffers['Y']},WB_M,WB_N,WB_K,{threads});",
            "reference.run();", collect="reference.describe(" + json.dumps(name) + ");",
            includes=("openblas_reference.h",))
        return PreparedNative(call, links, {"reference_build": configuration,
                              "reference_boundary": "plain-inputs SGEMM; allocation and initialization excluded"}, includes)


class OneDNNAdapter:
    def prepare_native(self, case, target, options, directory, executor, workload, name):
        kinds = {"gemm_f32": 0, "gemm_bf16_f32": 1, "gemm_s8_s32": 2}
        if case.name not in kinds:
            raise NotImplementedError("oneDNN reference implements F32, BF16/F32 and S8/S32 matmul workloads")
        if options.get("weights", "plain") != "plain":
            raise NotImplementedError("this matmul boundary requires plain strides; prepacked weights need a separate workload scope")
        configuration, includes, links = library_configuration(target, options, "onednn")
        if configuration.get("threading", "omp") not in ("omp", "seq"):
            raise NotImplementedError("oneDNN reference currently accepts omp or seq threading")
        threads = target.settings.get("threads", 1)
        call = NativeCall(name,
            f"OneDNNReference reference({workload.buffers['X']},{workload.buffers['W']},"
            f"{workload.buffers['Y']},WB_M,WB_N,WB_K,{kinds[case.name]},{threads});",
            "reference.run();", collect="reference.describe(" + json.dumps(name) + ");",
            includes=("onednn_reference.h",))
        return PreparedNative(call, links, {"reference_build": configuration,
            "reference_boundary": "plain-inputs execute+wait including internal packing; primitive and scratchpad prepared outside timing"},
            includes, {"OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE",
                       "ONEDNN_VERBOSE": "0", "DNNL_VERBOSE": "0"})

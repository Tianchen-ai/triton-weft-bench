"""Weft candidates and the fixed GGML library, independent of input selection."""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from .core import NativeCall
from .adapters.dense import bind_metadata


@dataclass(frozen=True)
class WeftKernel:
    source: str = ""
    symbol: str = ""
    artifact: str = ""
    compiler: str = ""
    options: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value):
        if ":" in value:
            source, symbol = value.rsplit(":", 1)
            return cls(source=str(Path(source).expanduser().resolve()), symbol=symbol)
        return cls(artifact=str(Path(value).expanduser().resolve()))

    def configuration(self):
        result = {}
        if self.source:
            result.update(source=self.source, symbol=self.symbol)
        if self.artifact:
            result.update(
                path=self.artifact, factory="weft_bench.adapters.native:ArtifactAdapter"
            )
        if self.compiler:
            result["compiler"] = self.compiler
        if self.options:
            result["compile_options"] = self.options
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass(frozen=True)
class GGML:
    source: str
    libraries: str
    compile_commands: str = ""

    def configuration(self):
        return {k: v for k, v in self.__dict__.items() if v}


def candidate_configuration(case, base, candidate):
    """Let the selected provider bind a source/artifact override to this workload."""
    if base.get("factory"):
        from .loading import load_object
        configure = getattr(load_object(base["factory"]), "configure_candidate", None)
        if configure:
            return configure(case, base, candidate)
    return {**base, **candidate.configuration()}


def weft_call(name, case, workload, metadata, aliases=None, packed_inputs=None):
    dimensions = {**case.shape}
    if case.family == "ime":
        dimensions["M"] = case.data.get("tile_rows", 4)
    whole_workload = case.family in ("vec_dot", "dequantize") and any(
        (aliases or {}).get(a["name"], a["name"]) == "W" and a["kind"] == "view" and len(a["shape"]) == 2
        for a in metadata["arguments"])
    if whole_workload and not workload.batch_buffers:
        raise ValueError("the workload has no complete row-batch binding")
    validate_binding(case, metadata, aliases or {}, dimensions, packed_inputs, whole_workload=whole_workload)
    call = bind_metadata(
        metadata, workload.batch_buffers if whole_workload else workload.buffers, dimensions, workload.scalars, aliases,
        workload.strides
    )
    if case.name == "rope_full":
        call = "data.prepare_rope(); " + call
    converted_projection = case.family == "projection" and case.data.get("activation_dtype") != "f32"
    setup = "std::vector<uint8_t> workspace(data.M*data.activation_bytes);" if converted_projection else ""
    from .workloads import ports
    output_only = any(port.name == "Y" and port.access == "output" for port in ports(case))
    initializes_output = output_only and any((aliases or {}).get(arg["name"], arg["name"]) == "Y"
                                             and arg.get("access") == "readwrite" for arg in metadata["arguments"])
    if case.family == "projection" or initializes_output:
        setup += "wb::poison_output(output);"
    collect = ""
    if converted_projection:
        collect = (
            "\n#ifdef WB_PROJECTION_Q8K_CONTRACT\n"
            "if(workspace!=data.quantized)throw std::runtime_error(\"candidate activation bytes differ from declared Q8_K contract\");\n"
            "#endif\n"
        )
    return NativeCall(name, setup, call, collect=collect, includes=(name + "-kernel.h",),
                      whole_workload=whole_workload,
                      snapshots={"Xq": "workspace"} if converted_projection else {})


def validate_binding(case, metadata, aliases, dimensions, packed_inputs=None, *, whole_workload=False):
    from .workloads import ports

    declared = ports(case)
    strided_ports = {port.name for port in declared if port.strides}
    port_access = {p.name: {"input": "read", "output": "write", "inout": "readwrite"}[p.access] for p in declared}
    expected = {
        p.name: (p.shape, "dense." + p.dtype if p.dtype in
                 ("i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64") else p.dtype.upper())
        for p in declared
    }
    k = case.shape.get("K")
    if case.family in ("vec_dot", "quantize", "dequantize") and not whole_workload:
        expected = {
            name: ((1,) if name == "Y" and case.family == "vec_dot" else (k,), encoding)
            for name, (_, encoding) in expected.items()
        }
    if case.family == "projection":
        from .operators import FORMATS, PARTNERS

        fmt = case.data["format"]
        activation = (
            "dense.f16" if fmt == "f16" else PARTNERS[FORMATS.index(fmt)].upper()
        )
        if case.data.get("activation_dtype") != "f32":
            expected["Xh" if fmt == "f16" else "Xq"] = ((dimensions["M"], k), activation)
        if fmt == "f16":
            expected["W"] = (expected["W"][0], "dense.f16")
    if case.family == "ime":
        rows = dimensions["M"]
        port_access.update(W=port_access["W_f32"], Xq=port_access["X_f32"])
        expected = {
            "W": (
                (dimensions["N"], k),
                "IMEQ4Affine_I[16]" if case.data["affine"] else "IMEQ4Centered_I[16]",
            ),
            "Xq": ((rows, k), f"IMEQ8_I[{rows}]"),
            "Y": ((rows, dimensions["N"]), "dense.f32"),
        }
    if case.family == "ime2":
        rows = dimensions["M"]
        port_access.update(W=port_access["W_codes"], Xq=port_access["X_codes"])
        expected = {"W": ((dimensions["N"], k), "IME2Weight8Rows[32]"),
                    "Xq": ((rows, k), f"IME2Activation8Rows[{case.data['tile_rows']}]"),
                    "Y": ((rows, dimensions["N"]), "dense.f32")}
    dims = {**metadata.get("bindings", {}), **dimensions}
    for arg in metadata["arguments"]:
        role = aliases.get(arg["name"], arg["name"])
        if case.family == "projection" and case.data.get("activation_dtype") == "f32" and role in ("Xq", "Xh"):
            raise ValueError("mixed F16/F32 projection preserves X and has no converted activation scratch")
        if arg["kind"] not in ("view", "pointer") or role not in expected:
            continue
        wanted, encoding = expected[role]
        if role in strided_ports and (arg["kind"] != "view" or not arg.get("strided")):
            raise ValueError(f"workload {role} requires an explicit strided View ABI")
        actual_encoding = (packed_inputs or {}).get(arg["name"], arg["encoding"])
        if actual_encoding != encoding:
            raise ValueError(
                f"kernel buffer {arg['name']} is {arg['encoding']}; workload {role} requires {encoding}"
            )
        if arg["kind"] == "view":
            shape = tuple(int(x) if str(x).isdecimal() else dims[x] for x in arg["shape"])
            if shape != wanted:
                raise ValueError(
                    f"kernel buffer {arg['name']} shape is {shape}; workload {role} requires {wanted}"
                )
        access = (
            "readwrite"
            if role in ("Xq", "Xh") and case.family == "projection"
            else port_access[role]
        )
        # An operator may initialize Y and then use it for partial sums.
        # Their initial output is poisoned above: relying on caller-provided
        # zeros fails the numerical check. Initialization remains in the call.
        initialized_output = (role == "Y" and
                              access == "write" and arg.get("access") == "readwrite")
        if arg.get("access") != access and not initialized_output:
            raise ValueError(
                f"kernel access for {role} does not match the workload input/output contract"
            )


def ggml_call(name, case):
    call = _ggml_call(name, case)
    if case.family == "dense":
        entry = {"gemm_f32": "ggml_graph_compute", "rms_norm": "ggml_compute_forward_rms_norm_mul_fused",
                 "layer_norm": "ggml_compute_forward_norm", "softmax": "ggml_compute_forward_soft_max",
                 "rope_split": "wb_ggml_rotate", "rope_full": "ggml_compute_forward_rope",
                 "affine": "ggml_compute_forward_scale"}[case.name]
        pointer = "&" + entry
    elif case.family == "vec_dot":
        entry, pointer = "ggml_cpu_type_traits.vec_dot", "data.traits->vec_dot"
    elif case.family == "quantize":
        entry = {"q8_k": "quantize_row_q8_K", "q8_0": "quantize_row_q8_0", "q8_1": "quantize_row_q8_1"}[case.dtype.split("->")[1]]
        pointer = "&" + entry
    elif case.family == "dequantize":
        entry, pointer = "ggml_type_traits.to_float", "ggml_get_type_traits(data.weight_type)->to_float"
    elif case.family == "projection":
        entry, pointer = "ggml_backend_graph_plan_compute", "&ggml_backend_graph_plan_compute"
    else:
        return call
    extra = ",reference.dispatch_details()" if case.family == "projection" else ""
    observe = "wb_ggml_reference_info(" + json.dumps(name) + "," + json.dumps(entry) + ",reinterpret_cast<const void*>(" + pointer + ")" + extra + ");"
    if case.family == "projection" and case.data.get("activation_dtype") == "f32":
        observe += "reference.check_activation_precision();"
    return replace(call, collect=call.collect + observe, includes=(*call.includes, "ggml_reference_info.h"))


def _ggml_call(name, case):
    family = case.family
    if family == "ime2":
        return NativeCall(name, "",
            'for(size_t m=0;m<data.M;m+=WB_TILE_ROWS) for(size_t n=0;n<data.N;n+=32) { if(spacemit_kernels::ime2::gemm_kernel_i8i8(32,data.activation.data()+m*(data.K/32)*38,data.weights.data()+n*(data.K/32)*34,nullptr,output.data()+m*data.N+n,WB_TILE_ROWS,32,data.K/32,data.N)!=WB_TILE_ROWS)throw std::runtime_error("IME2 row tile unhandled"); }',
            includes=("ime_kernels.h",))
    if family == "amx_tile":
        weight, activation, block = {"q4_0": ("Q4_0", "Q8_0", 32), "q4_k": ("Q4_K", "Q8_K", 256),
                                     "q6_k": ("Q6_K", "Q8_K", 256)}[case.data["format"]]
        entry = (f"tinygemm_kernel_vnni<{activation},{weight},f32,1,64,{block}>" if case.data.get("decode") else
                 f"tinygemm_kernel_amx<{activation},{weight},f32,{block}>")
        details = {"implementation": name, "reference_dispatch": {
            "library": "GGML", "entry": entry, "source": "ggml/src/ggml-cpu/amx/mmq.cpp",
            "linkage": "configured upstream template compiled into native worker",
            "weight_buffer": "normal-GGML-AMX", "activation": "prequantized-" + activation,
            "timing": "one complete native tile; excludes graph dispatch and activation quantization"}}
        return NativeCall(name, "AMXTileReference reference(data,output);", "reference.run();",
                          collect="std::puts(" + json.dumps("WB_DETAILS " + json.dumps(details)) + ");",
                          includes=("amx_tile_reference.h",))
    if family == "dense":
        return NativeCall(
            name,
            "DenseReference reference(data.input,data.weight,data.bias,data.cosine,data.sine,output,data.epsilon,data.scale,data.scalar_bias);",
            "reference.run(data.input,output);",
            includes=("dense_reference.h",),
        )
    if family == "vec_dot":
        return NativeCall(
            name,
            'if(!data.traits||!data.traits->vec_dot||data.traits->nrows!=1)throw std::runtime_error("GGML single-row dot unavailable");',
            "data.traits->vec_dot(data.K,output.data()+row,0,data.weights.data()+row*data.weight_bytes,0,data.activation.data(),0,1);",
        )
    if family == "quantize":
        fmt = case.dtype.split("->")[1]
        symbol = {
            "q8_k": "quantize_row_q8_K",
            "q8_0": "quantize_row_q8_0",
            "q8_1": "quantize_row_q8_1",
        }[fmt]
        return NativeCall(
            name,
            "",
            symbol
            + "(data.source.data()+row*data.K,output.data()+row*data.weight_bytes,data.K);",
        )
    if family == "dequantize":
        return NativeCall(
            name,
            "",
            "ggml_get_type_traits(data.weight_type)->to_float(data.weights.data()+row*data.weight_bytes,output.data()+row*data.K,data.K);",
        )
    if family == "projection":
        return NativeCall(
            name,
            "ProjectionReference reference(data.weight_type,data.M,data.N,data.K,data.weights.data(),data.activation.data(),output);",
            "reference.run();",
            "reference.collect();",
            includes=("projection_reference.h",),
        )
    if family == "ime":
        return NativeCall(
            name,
            "",
            'if(spacemit_kernels::ime1::gemm_kernel_i8i4(32,data.activation.data()+m*(data.K/32)*36,data.weights.data(),WB_AFFINE?data.weights.data():nullptr,output.data()+m*data.N,WB_TILE_ROWS,data.N,data.K/32,data.N)!=WB_TILE_ROWS)throw std::runtime_error("IME row tile unhandled");',
            includes=("spacemit/ime_kernels.h",),
        )
    raise ValueError(f"GGML does not implement {case.name}")

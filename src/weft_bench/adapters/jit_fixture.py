"""Bind Canonical parameter contracts to the existing native workload."""
from dataclasses import replace
import math

from ..libraries import weft_call
from ..workloads import ports
from .workspaces import invocation_workspaces
from .persistent import persistent_inputs


TYPES = {"f32": ("float", 4), "f64": ("double", 8), "f16": ("_Float16", 2),
         "bf16": ("uint16_t", 2), "i1": ("bool", 1), "u1": ("bool", 1), "index": ("size_t", 8),
         **{prefix + str(bits): (ctype + str(bits) + "_t", bits // 8)
            for prefix, ctype in (("i", "int"), ("u", "uint")) for bits in (8, 16, 32, 64)}}


def prepare(case, workload, interface, options):
    metadata = {**interface, "symbol": "jit_entry", "bindings": options.get("backend_options", {}).get("meta", {})}
    arguments = []
    for argument in interface["arguments"]:
        arg = dict(argument)
        scalar = arg["kind"] == "scalar"
        if arg["dtype"]:
            ctype, width = TYPES[arg["dtype"]]
            arg.update(storage_bytes=width, record_elements=1, alignment=width, interleave_rows=0)
        else:
            ctype = "uint8_t"
        writable = arg["access"] in ("write", "readwrite")
        arg["c_type"] = ctype if scalar else ("" if writable else "const ") + ctype + " *"
        arguments.append(arg)
    metadata["arguments"] = arguments
    aliases = options.get("arguments", {})
    scratch_names = options.get("workspaces", [])
    if any(name in aliases or name in aliases.values() for name in scratch_names):
        raise ValueError("workspace arguments cannot alias workload ports")
    workload, scratch, scratch_details = invocation_workspaces(case, workload, metadata, scratch_names)
    workload, packing, packed, packed_details = persistent_inputs(
        case, workload, metadata, options.get("persistent_inputs", []), aliases, options.get("persistent_providers"))
    call = weft_call("jit", case, workload, metadata, aliases, packed)
    includes = ("projection_reference.h",) if any(p["provider"] == "ggml-model-default/AMX" for p in packed_details) else ()
    if includes:
        workload = replace(workload, defines=(*workload.defines, "-DWB_GGML_MODEL_WEIGHTS=1"))
    call = replace(call, prepare=call.prepare + scratch + packing, includes=includes)
    dimensions = {**metadata["bindings"], **case.shape}
    if case.family == "ime":
        dimensions["M"] = case.data.get("tile_rows", 4)
    declared = {p.name: p for p in ports(case)}
    invocation = []
    for argument in arguments:
        spec = dict(argument)
        role = aliases.get(spec["name"], spec["name"])
        if spec["kind"] != "scalar":
            shape = ([int(x) if str(x).isdecimal() else dimensions[x] for x in spec["shape"]]
                     if spec["kind"] == "view" else list(workload.pointer_shapes.get(role, declared[role].shape)))
            strides = workload.strides.get(role, tuple(math.prod(shape[i + 1:]) for i in range(len(shape))))
            if spec["encoding"].startswith("dense."):
                size = (1 + sum((n - 1) * s for n, s in zip(shape, strides))) * spec["storage_bytes"] if math.prod(shape) else 0
            else:
                if shape[-1] % spec["record_elements"]:
                    raise ValueError("JIT View must cover complete packed records")
                size = math.prod(shape) // spec["record_elements"] * spec["storage_bytes"]
            spec.update(shape=shape, bytes=size)
            if spec.get("strided"):
                spec["strides"] = list(strides)
        invocation.append(spec)
    return workload, call, metadata, invocation, {"workspaces": scratch_details, "persistent_inputs": packed_details}

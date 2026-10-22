"""Explicit invocation scratch, allocated from the native artifact's typed ABI."""
from dataclasses import replace
import math
import re


def invocation_workspaces(case, workload, metadata, names):
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("workspaces must be a list of native argument names")
    if len(set(names)) != len(names):
        raise ValueError("workspace arguments must be unique")
    arguments = {arg["name"]: arg for arg in metadata["arguments"]}
    dimensions = {**metadata.get("bindings", {}), **case.shape}
    types = {"f32": ("float", 4), "f64": ("double", 8),
             "f16": ("uint16_t", 2), "bf16": ("uint16_t", 2),
             **{prefix + str(bits): (ctype + str(bits) + "_t", bits // 8)
                for prefix, ctype in (("i", "int"), ("u", "uint")) for bits in (8, 16, 32, 64)}}
    buffers = dict(workload.buffers)
    batch_buffers = dict(workload.batch_buffers)
    preparation, description = [], []
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in buffers or name not in arguments:
            raise ValueError(f"workspace must name an extra native argument: {name!r}")
        arg = arguments[name]
        if arg["kind"] != "view" or arg.get("strided") or arg.get("access") not in ("write", "readwrite"):
            raise ValueError("invocation workspace requires a writable contiguous View")
        encoding = arg.get("encoding", "")
        shape = [int(extent) if str(extent).isdecimal() else dimensions[extent] for extent in arg["shape"]]
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError("workspace shape requires nonnegative concrete extents")
        variable = "invocation_workspace_" + name
        if encoding.startswith("dense.") and encoding[6:] in types:
            ctype, width = types[encoding[6:]]
            count, size = math.prod(shape), math.prod(shape) * width
        else:
            if not shape:
                raise NotImplementedError("encoded workspace requires a record domain")
            if arg["record_elements"] <= 0 or arg["storage_bytes"] <= 0 or shape[-1] % arg["record_elements"]:
                raise ValueError("encoded workspace requires complete declared records")
            ctype = "uint8_t"
            width = 1
            count = size = math.prod(shape) // arg["record_elements"] * arg["storage_bytes"]
            if arg["interleave_rows"]:
                rows = int(arg["interleave_rows"])
                packing = arg.get("artifact_pack", {})
                symbol = packing.get("size_symbol", "")
                if len(shape) != 2 or rows <= 0 or (not arg.get("storage_layout") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol)):
                    raise NotImplementedError("interleaved workspace requires a two-dimensional native size export")
                if shape[0] % rows:
                    raise ValueError("interleaved workspace requires complete row cohorts")
                if arg.get("artifact_pack") or not arg.get("storage_layout"):
                    preparation.append(f'if (native.symbol<decltype(&{symbol})>("{symbol}")({shape[0]}, {shape[1]}) != {size}) '
                                       'throw std::runtime_error("workspace size differs from native layout");')
        alignment = math.lcm(64, int(arg["alignment"]), width)
        preparation.append(f"std::vector<{ctype}> {variable}_storage({count}+{alignment // width});"
                           f"wb::poison_output({variable}_storage);"
                           f"auto {variable}=reinterpret_cast<{ctype}*>((reinterpret_cast<uintptr_t>"
                           f"({variable}_storage.data())+{alignment}-1)/{alignment}*{alignment});")
        buffers[name] = variable
        if batch_buffers:
            batch_buffers[name] = variable
        description.append({"argument": name, "shape": shape, "encoding": encoding,
                            "bytes": size, "allocation": "before-timing",
                            "required_alignment": arg["alignment"], "allocation_alignment": alignment,
                            "population": "candidate-invocation"})
    return replace(workload, buffers=buffers, batch_buffers=batch_buffers), "".join(preparation), description

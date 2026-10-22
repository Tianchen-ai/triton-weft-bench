"""Bind native ABI from typed artifact metadata; never parse generated C text."""

import math
import re

DENSE_KINDS = {
    "rms_norm": 0,
    "layer_norm": 1,
    "softmax": 2,
    "rope_split": 3,
    "gemm_f32": 4,
    "rope_full": 5,
    "affine": 6,
}


def dense_defines(case):
    return [f"-DWB_KIND={DENSE_KINDS[case.name]}"] + [
        f"-DWB_{k}={case.shape.get(k,1)}" for k in ("M", "N", "K", "T", "H", "P")
    ]


def bind_metadata(metadata, pointers, dimensions, scalars=None, aliases=None, buffer_strides=None):
    """Pure metadata to direct native call binding, reusable by custom workloads."""
    scalars, aliases = scalars or {}, aliases or {}
    args, strides = [], []
    dims = {**metadata.get("bindings", {}), **dimensions}
    for parameter in metadata["arguments"]:
        name = aliases.get(parameter["name"], parameter["name"])
        if parameter["kind"] in ("view", "pointer"):
            if name not in pointers:
                raise ValueError(
                    f"missing workload buffer {name!r}; configure implementation arguments mapping"
                )
            pointer = pointers[name]
            # Dtype/shape/access are checked before this binder. Encoding lookup
            # tables may share byte storage; the exported ABI determines their
            # signed/unsigned view type just as it does an ordinary pointer.
            cast_type = re.sub(r"\b(?:restrict|__restrict|__restrict__)\b", "", parameter["c_type"])
            pointer = "reinterpret_cast<" + cast_type + ">(" + pointer + ")"
            args.append(pointer)
            if parameter["kind"] == "view" and parameter["strided"]:
                shape = [
                    int(x) if str(x).isdecimal() else dims[x]
                    for x in parameter["shape"]
                ]
                selected = (buffer_strides or {}).get(name, tuple(math.prod(shape[i + 1:]) for i in range(len(shape))))
                if len(selected) != len(shape):
                    raise ValueError("workload stride rank differs from native View " + name)
                strides.extend(str(value) for value in selected)
        elif parameter["kind"] == "scalar":
            args.append(str(scalars[name]))
        else:
            raise ValueError(f"unsupported ABI kind {parameter['kind']}")
    args.extend(str(dims[d]) for d in metadata["shape_parameters"])
    args.extend(strides)
    return metadata["symbol"] + "(" + ", ".join(args) + ");"

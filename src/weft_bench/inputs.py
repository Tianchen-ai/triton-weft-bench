"""Reusable input selections, independent of implementations and execution hosts."""

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


@dataclass(frozen=True)
class Inputs:
    preset: str = "default"
    shape: dict[str, int] = field(default_factory=dict)
    seed: int = 410
    buffers: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        values = json.loads(path.read_text())
        buffers = values.get("buffers", {})
        for key, value in buffers.items():
            if isinstance(value, str):
                buffers[key] = str((path.parent / value).resolve())
        return cls(**values)

    def resolve(self, operation):
        from .operators import get

        case = get(operation) if isinstance(operation, str) else operation
        shape = dict(case.shape)
        if self.preset == "decode":
            if case.family in ("dense", "matmul"):
                shape.update({k: 1 for k in ("M", "T") if k in shape})
            elif case.family in ("quantize", "dequantize", "vec_dot"):
                shape["rows"] = 1 if case.scope == "vec-dot-leaf" else shape["rows"]
            elif case.family == "projection":
                shape["M"] = 1
        elif self.preset == "prefill":
            if "M" in shape:
                shape["M"] = 128
            if "T" in shape:
                shape["T"] = 128
        elif self.preset not in ("default", "quick", "llama7b"):
            raise ValueError(
                f"unknown inputs preset {self.preset!r}; select default, decode, prefill or a JSON file"
            )
        unknown = self.shape.keys() - shape.keys()
        if unknown or any(type(v) is not int or v <= 0 for v in self.shape.values()):
            raise ValueError(f"invalid input shape {self.shape}")
        if type(self.seed) is not int or not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("input seed must fit uint32")
        shape.update(self.shape)
        from .workloads import parameter_defaults

        unknown_parameters = self.parameters.keys() - parameter_defaults(case).keys()
        if unknown_parameters:
            raise ValueError(f"unsupported parameters for {case.name}: {sorted(unknown_parameters)}")
        scope = case.scope
        if case.family == "vec_dot":
            scope = (
                "vec-dot-leaf"
                if shape["rows"] == 1
                else "independent-dots-shared-activation"
            )
        return replace(case, shape=shape, scope=scope)

    def materialize(self, directory, ports=()):
        """Snapshot custom buffers so candidate/reference/replay see identical bytes."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        declared = {p.name: p for p in ports if p.access in ("input", "inout")}

        def array_bytes(role, array):
            spec = declared.get(role)
            wanted = {"f32": "<f4", "f16": "<f2", "bf16": "<u2", "f64": "<f8", "i1": "|b1",
                      "i8": "|i1", "u8": "|u1", "i16": "<i2", "u16": "<u2",
                      "i32": "<i4", "u32": "<u4", "i64": "<i8", "u64": "<u8"}.get(spec.dtype) if spec else None
            if wanted:
                if array.dtype.str != wanted or tuple(array.shape) != spec.shape:
                    raise ValueError(f"{role} array must be {spec.dtype}{spec.shape}")
            elif array.dtype.str not in ("|u1", "|i1"):
                raise ValueError(f"{role} encoded input must contain packed bytes")
            return array.tobytes(order="C")

        for role, value in self.buffers.items():
            if not role.isidentifier():
                raise ValueError(f"invalid input role {role!r}")
            if declared and role not in declared:
                raise ValueError(f"unknown input role {role!r}")
            if isinstance(value, (str, Path)):
                source = Path(value).expanduser()
                if source.suffix == ".npy":
                    import numpy as np

                    array = np.load(source, allow_pickle=False)
                    content = array_bytes(role, array)
                else:
                    content = source.read_bytes()
            elif isinstance(value, (bytes, bytearray, memoryview)):
                content = bytes(value)
            elif hasattr(value, "dtype"):
                content = array_bytes(role, value)
            else:
                spec = declared.get(role)
                if spec and spec.dtype == "bf16":
                    converted = []
                    for number in value:
                        bits = struct.unpack("<I", struct.pack("<f", number))[0]
                        if bits & 0x7f800000 == 0x7f800000:
                            converted.append((bits >> 16) | (0x40 if bits & 0x7fffff else 0))
                        else:
                            converted.append(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff)
                    content = struct.pack("<" + "H" * len(converted), *converted)
                else:
                    fmt = {"f16": "e", "f32": "f", "f64": "d", "i1": "?", "i8": "b", "u8": "B",
                       "i16": "h", "u16": "H", "i32": "i", "u32": "I", "i64": "q", "u64": "Q"}.get(spec.dtype if spec else "f32", "f")
                    content = struct.pack("<" + fmt * len(value), *value)
            path = directory / (role + ".bin")
            path.write_bytes(content)
            files[role] = {
                "path": str(path.resolve()),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        return files

    def describe(self, files=None):
        return {
            "preset": self.preset,
            "shape": self.shape,
            "seed": self.seed,
            "parameters": self.parameters,
            "buffers": files or {},
        }


def resolve_inputs(value=None):
    """Resolve the same preset, JSON file or mapping for every bench entry point."""
    if value is None:
        return Inputs()
    if isinstance(value, Inputs):
        return value
    if isinstance(value, dict):
        return Inputs(**value)
    if isinstance(value, Path):
        return Inputs.load(value)
    if isinstance(value, str):
        return Inputs.load(value) if Path(value).is_file() else Inputs(preset=value)
    raise TypeError("inputs must be Inputs, a preset, a JSON path or a mapping")


def native_inputs(inputs, files):
    lines = [f"wb::InputSet inputs({inputs.seed});"]
    for role, info in files.items():
        lines.append(
            "inputs.files["
            + json.dumps(role)
            + "] = "
            + json.dumps(info["remote"])
            + ";"
        )
    for key, value in inputs.parameters.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or (key == "epsilon" and value <= 0)
            or (key == "start" and (value < 0 or int(value) != value))
        ):
            raise ValueError(f"unsupported input parameter {key!r}")
        lines.append(
            "inputs.parameters[" + json.dumps(key) + "] = " + repr(float(value)) + ";"
        )
    return "\n".join(lines)

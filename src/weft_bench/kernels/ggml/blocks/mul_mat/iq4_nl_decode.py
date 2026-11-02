"""Compatibility import; implementations live in the installed Triton product."""
from importlib import import_module as _import_module
globals().update({name: value for name, value in vars(_import_module('triton.kernels.weft.ggml.blocks.mul_mat.iq4_nl_decode')).items() if not name.startswith("__")})

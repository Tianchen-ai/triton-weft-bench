"""Compatibility import; implementations live in the installed Triton product."""
from importlib import import_module as _import_module
globals().update({name: value for name, value in vars(_import_module('triton.kernels.weft.ggml.blocks.vec_dot.iq2_xxs_q8_k')).items() if not name.startswith("__")})

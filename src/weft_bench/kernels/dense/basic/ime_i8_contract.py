"""Compatibility import; implementations live in the installed Triton product."""
from importlib import import_module as _import_module
globals().update({name: value for name, value in vars(_import_module('triton.kernels.weft.dense.basic.ime_i8_contract')).items() if not name.startswith("__")})

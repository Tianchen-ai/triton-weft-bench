"""Load a user kernel file identically for compilation and native JIT calls."""
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys


def load_kernel(path, symbol):
    path = Path(path).expanduser().resolve(strict=True)
    parts = [] if path.name == "__init__.py" else [path.stem]
    root = path.parent
    while (root / "__init__.py").is_file():
        parts.insert(0, root.name)
        root = root.parent
    # A private namespace preserves real package initializers and relative
    # imports without replacing public modules with an unrelated same-named file.
    # The reversible path encoding is stable across processes; it is not a hash.
    namespace = "_weft_bench_source_" + os.fsencode(root).hex()
    if namespace not in sys.modules:
        spec = importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        sys.modules[namespace] = importlib.util.module_from_spec(spec)
    module = importlib.import_module(".".join([namespace, *parts]))
    kernel = getattr(module, symbol)
    if getattr(kernel, "model", None) != "weft":
        raise ValueError("Triton candidate must use @triton.jit(model='weft')")
    return kernel

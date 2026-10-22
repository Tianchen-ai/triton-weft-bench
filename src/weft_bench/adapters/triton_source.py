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


def deploy_source(executor, source):
    """Resolve installed corpus modules or transfer a complete custom package."""
    source = Path(source).expanduser().resolve(strict=True)
    package = Path(__file__).resolve().parents[1]
    if source.is_relative_to(package / "kernels"):
        return {"source": str(source),
                "source_module": "weft_bench." + ".".join(source.relative_to(package).with_suffix("").parts)}
    root = source.parent
    while (root / "__init__.py").is_file():
        root = root.parent
    package_root = source.parent if root == source.parent else root / source.relative_to(root).parts[0]
    executor.upload_many((path, "author/" + str(path.relative_to(root))) for path in package_root.rglob("*.py"))
    return {"source": str(Path(executor.workspace) / "author" / source.relative_to(root)), "source_module": None}


def collect_artifact(executor, relative, directory):
    """Collect an artifact made on the target, without rebuilding its code."""
    import tarfile
    directory = Path(directory)
    archive = executor.workspace + "/" + relative + ".tar.gz"
    executor.command(["tar", "-czf", archive, "-C", executor.workspace, relative], stage="collect")
    destination = directory / (relative + ".tar.gz")
    executor.download(archive, destination)
    with tarfile.open(destination) as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unexpected native artifact archive member")
        bundle.extractall(directory)
    destination.unlink()
    return directory / relative

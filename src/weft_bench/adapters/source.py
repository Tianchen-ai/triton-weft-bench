"""Load either native frontend's author source without importing the other."""
import importlib
import importlib.machinery
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import sys


def product_package(producer):
    package = {"triton": "triton.kernels.weft", "tilelang": "tilelang.weft.kernels"}.get(producer)
    if package is None:
        raise ValueError(f"unknown source frontend {producer!r}")
    root, *parts = package.split(".")
    spec = importlib.util.find_spec(root)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Install {root} to use its source JIT")
    # A compiler installation can compile a custom author package without
    # shipping our optional optimized corpus. Import a corpus module only
    # when the source actually belongs to that installed package.
    return package, Path(next(iter(spec.submodule_search_locations))).resolve().joinpath(*parts)


def validate_kernel(kernel, producer):
    if producer == "triton" and getattr(kernel, "model", None) != "weft":
        raise ValueError("Triton candidate must use @triton.jit(model='weft')")
    if producer == "tilelang":
        from tilelang.jit import JITImpl
        from tvm.target import Target
        definition = getattr(kernel, "jit_impl", kernel)
        target = definition.target
        kind = target.kind.name if isinstance(target, Target) else str(target)
        if not isinstance(definition, JITImpl) or kind != "weft":
            raise ValueError("TileLang candidate must use @tilelang.jit(target='weft')")
    return kernel


def load_kernel(path, symbol, *, producer="triton"):
    path = Path(path).expanduser().resolve(strict=True)
    package, product = product_package(producer)
    if path.is_relative_to(product):
        parts = path.relative_to(product).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = importlib.import_module(package + ("." + ".".join(parts) if parts else ""))
        return validate_kernel(getattr(module, symbol), producer)
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
    return validate_kernel(getattr(module, symbol), producer)


def deploy_source(executor, source, *, producer="triton"):
    """Resolve installed corpus modules or transfer a complete custom package."""
    source = Path(source).expanduser().resolve(strict=True)
    package, product = product_package(producer)
    if source.is_relative_to(product):
        return {"source": str(source), "source_module": package + "." +
                ".".join(source.relative_to(product).with_suffix("").parts)}
    package = Path(__file__).resolve().parents[1]
    try:
        installed = distribution("triton-weft-bench")
        wheel_package = (Path(installed.locate_file("weft_bench")).resolve() == package and
                         any(str(path).endswith(".dist-info/RECORD") for path in installed.files or ()))
    except PackageNotFoundError:
        wheel_package = False
    if wheel_package and source.is_relative_to(package / "kernels"):
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

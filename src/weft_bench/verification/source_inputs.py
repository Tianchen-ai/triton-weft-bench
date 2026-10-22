"""Original author sources deployed alongside a configured runtime probe."""
from pathlib import Path, PurePosixPath


def source_files(configuration):
    sources = configuration.get("source_files", {})
    if not isinstance(sources, dict):
        raise ValueError("verification.runtime.source_files must map relative filenames to local files")
    result = []
    for name, original in sources.items():
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("runtime source destination must be a relative POSIX filename")
        destination = PurePosixPath(name)
        if destination.is_absolute() or ".." in destination.parts or str(destination) == ".":
            raise ValueError("runtime source destination must stay inside the worker directory")
        if not isinstance(original, str) or not original:
            raise ValueError("runtime source origin must be a local file path")
        path = Path(original).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"runtime source file does not exist: {path}")
        result.append((path, str(destination)))
    if len({name for _, name in result}) != len(result):
        raise ValueError("runtime source destinations must be distinct")
    return result

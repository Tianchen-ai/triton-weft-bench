"""Trace bundled OpenMP through the product's declared JIT and AOT copies."""
import json
from pathlib import Path


def audit_runtime(*, sdk, compiler, installed, cached, exported, artifact, cache, loaded, report):
    observation = {
        "status": "checking", "sdk": str(sdk), "compiler": str(compiler),
        "installed_runtime_files": installed, "jit_runtime_files": cached,
        "exported_runtime_files": exported, "artifact": str(artifact),
        "cache": str(cache), "loaded_openmp": loaded, "file_checks": [],
        "loaded_matches": {},
    }
    report = Path(report)
    report.write_text(json.dumps(observation, indent=2) + "\n")

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    try:
        sdk, artifact, cache = Path(sdk).resolve(), Path(artifact).resolve(), Path(cache).resolve()
        require(Path(compiler).resolve().is_relative_to(sdk / "bin"),
                "default toolchain did not select bundled Clang")
        providers = {name: path for name, path in installed.items() if Path(name).name.startswith("libomp")}
        require(bool(providers), "installed package did not provide OpenMP runtime_files")
        allowed = {}
        for name, provider in providers.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "runtime dependency name is not relative: " + name)
            source = Path(provider).resolve()
            require(source.is_relative_to(sdk / "lib"), "OpenMP provider is outside the installed package: " + str(source))
            require(name in cached, "JIT metadata omitted bundled OpenMP: " + name)
            require(exported.get(name) == name, "AOT metadata omitted bundled OpenMP: " + name)
            copies = {"installed": source, "jit_cache": Path(cached[name]).resolve(),
                      "export": (artifact / relative).resolve()}
            require(copies["jit_cache"].is_relative_to(cache), "declared JIT OpenMP is outside this cache: " + str(copies["jit_cache"]))
            require(copies["export"].is_relative_to(artifact), "exported OpenMP escaped its artifact")
            expected = source.read_bytes()
            for origin, path in copies.items():
                content = path.read_bytes()
                same = content == expected
                observation["file_checks"].append({"name": name, "origin": origin, "path": str(path),
                                                   "provider": str(source), "bytes": len(content),
                                                   "equals_installed_provider": same})
                require(same, "OpenMP content differs from installed provider: " + str(path))
                allowed[str(path)] = {"name": name, "origin": origin, "provider": str(source)}
        require(bool(loaded), "execution did not load an observable OpenMP library")
        for path in loaded:
            resolved = str(Path(path).resolve())
            require(resolved in allowed, "execution loaded undeclared/external OpenMP: " + path)
            observation["loaded_matches"][path] = allowed[resolved]
        observation["status"] = "passed"
    except Exception as error:
        observation.update(status="failed", error=str(error))
        raise
    finally:
        report.write_text(json.dumps(observation, indent=2) + "\n")
    return observation

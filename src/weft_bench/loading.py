from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from .types import Target


def load_object(spec):
    module, separator, name = spec.rpartition(":")
    if not separator:
        raise ValueError("expected module:object or file.py:object")
    path = Path(module)
    if path.suffix == ".py":
        loader = importlib.util.spec_from_file_location("_weft_bench_user_" + path.stem, path.resolve())
        if loader is None or loader.loader is None:
            raise ValueError(f"cannot load {module}")
        loaded = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = loaded
        loader.loader.exec_module(loaded)
    else:
        loaded = importlib.import_module(module)
    return getattr(loaded, name)


def load_config(path=None):
    path = path or os.environ.get("WEFT_BENCH_CONFIG")
    if not path:
        raise ValueError("provide --config or WEFT_BENCH_CONFIG (see configs/example.json)")
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text())
    if not isinstance(data.get("targets"), dict):
        raise ValueError("configuration needs a targets mapping")
    for target in data["targets"].values():
        if target.get("bindings_file"):
            bindings_path = Path(target["bindings_file"]).expanduser()
            if not bindings_path.is_absolute():
                bindings_path = config_path.parent / bindings_path
            external = json.loads(bindings_path.read_text())
            target["bindings"] = {**external["bindings"], **target.get("bindings", {})}
    return data


def select_target(config, name):
    if name not in config["targets"]:
        raise ValueError(f"unknown target {name!r}; choose from {', '.join(config['targets'])}")
    settings = {**config.get("defaults", {}), **config["targets"][name]}
    implementations = {**config.get("implementations", {}), **settings.pop("implementations", {})}
    return Target(name, settings, implementations)

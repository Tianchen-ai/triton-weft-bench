#!/usr/bin/env python3
"""Locate a local Weft Bench installation; forward its CLI unchanged."""

import json
import os
from pathlib import Path
import shutil
import sys


def local_runtime(selected=None):
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    settings_path = root / "weft-bench" / "codex.local.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    if not isinstance(settings, dict):
        raise ValueError(f"{settings_path} must contain a JSON object")
    environments = settings.get("environments", {})
    if selected:
        if selected not in environments:
            raise ValueError(f"unknown bench environment {selected!r}; choose from {', '.join(environments)}")
        settings = environments[selected]
    command = settings.get("command", ["weft-bench"])
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ValueError("command must be a nonempty JSON array of argument strings")
    command = list(command)
    command[0] = os.path.expanduser(command[0])
    configured = settings.get("config")
    if configured is not None and (not isinstance(configured, str) or not configured):
        raise ValueError("config must be a nonempty path string")
    if configured:
        config = Path(configured).expanduser()
        if not config.is_absolute():
            config = settings_path.parent / config
    else:
        config = root / "weft-bench" / "bench.local.json"
    environment = os.environ.copy()
    if configured or config.is_file():
        if selected:
            environment["WEFT_BENCH_CONFIG"] = str(config)
        else:
            environment.setdefault("WEFT_BENCH_CONFIG", str(config))
    return command, environment, settings_path, sorted(environments)


def describe(command, environment, settings_path, environments, selected=None):
    executable = shutil.which(command[0])
    config = environment.get("WEFT_BENCH_CONFIG")
    result = {
        "command": command,
        "executable": executable,
        "settings": str(settings_path),
        "config": config,
        "targets": [],
        "environment": selected or "default",
        "environments": environments,
    }
    if config:
        payload = json.loads(Path(config).expanduser().read_text())
        result["targets"] = sorted(payload.get("targets", {}))
    result["ready"] = bool(executable and config and result["targets"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


def main():
    try:
        args = sys.argv[1:]
        selected = None
        if args[:1] == ["--environment"]:
            if len(args) < 2:
                raise ValueError("--environment requires a configured name")
            selected, args = args[1], args[2:]
        command, environment, settings_path, environments = local_runtime(selected)
        if args == ["--plugin-info"]:
            return describe(command, environment, settings_path, environments, selected)
        if not shutil.which(command[0]):
            raise ValueError(
                "Weft Bench executable not found. Install weft-bench or set command "
                f"in {settings_path}."
            )
        os.execvpe(command[0], command + args, environment)
    except (OSError, ValueError) as error:
        print(f"weft-bench plugin: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

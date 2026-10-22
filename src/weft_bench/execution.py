from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid


class ExecutionError(RuntimeError):
    def __init__(self, stage, command, completed):
        self.stage = stage
        self.completed = completed
        super().__init__(
            f"{stage}: exit {completed.returncode}: {shlex.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


class Executor:
    """Transport only. Native programs own measurement and numerical checks."""

    def __init__(self, target, local_directory):
        self.target = target
        self.settings = target.settings
        self.local = Path(local_directory)
        self.local.mkdir(parents=True, exist_ok=True)
        deployment = self.settings.get("deployment", {})
        self.host = deployment.get("host")
        self.ssh_options = deployment.get("ssh_options", [])
        self.launch_prefix = deployment.get("launch_prefix", [])
        for name, arguments in (("ssh_options", self.ssh_options), ("launch_prefix", self.launch_prefix)):
            if not isinstance(arguments, list) or any(not isinstance(arg, str) or not arg for arg in arguments):
                raise ValueError(f"deployment.{name} must be an argv list of nonempty strings")
        self.cpus = deployment.get("cpus", [])
        self.numa = deployment.get("numa", {})
        if not isinstance(self.numa, dict) or self.numa.keys() - {"memory_nodes", "policy", "command"}:
            raise ValueError("deployment.numa accepts memory_nodes, policy and command")
        if self.numa.get("policy", "bind") not in ("bind", "interleave", "preferred"):
            raise ValueError("NUMA policy must be bind, interleave or preferred")
        nodes = self.numa.get("memory_nodes", [])
        if not isinstance(nodes, list) or any(type(n) is not int or n < 0 for n in nodes):
            raise ValueError("NUMA memory_nodes must be nonnegative integers")
        if self.numa.get("policy") == "preferred" and len(nodes) != 1:
            raise ValueError("NUMA preferred requires exactly one memory node")
        self.timeout = self.settings.get("timeout", 1800)
        self.env = {str(k): str(v) for k, v in self.settings.get("env", {}).items()}
        root = deployment.get("workspace", str(self.local.resolve()))
        self.workspace = (
            str(Path(root) / ("run-" + uuid.uuid4().hex[:12]))
            if self.host
            else str(self.local.resolve())
        )
        self.commands = []

    def execution_argv(self, argv):
        """Apply execution policy to the actual native process, before exec."""
        command = [str(x) for x in argv]
        if self.cpus:
            command = ["taskset", "-c", ",".join(map(str, self.cpus)), *command]
        if self.numa.get("memory_nodes"):
            option = {"bind": "--membind", "interleave": "--interleave", "preferred": "--preferred"}[self.numa.get("policy", "bind")]
            command = [self.numa.get("command", "numactl"), option + "=" +
                       ",".join(map(str, self.numa["memory_nodes"])), *command]
        if self.launch_prefix:
            # Platform registration must precede affinity and exec: a runtime
            # may inspect vector width during its dynamic-loader initialization.
            command = [*self.launch_prefix, *command]
        return command

    def command(
        self, argv, *, cwd=None, env=None, affinity=False, stage="run", check=True,
        unset_env=(),
    ):
        command = [str(x) for x in argv]
        variables = {**self.env, **(env or {})}
        if unset_env:
            if isinstance(unset_env, str) or any(not isinstance(key, str) or
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in unset_env):
                raise ValueError("unset_env must contain environment variable names")
            command = ["env", *[arg for key in unset_env for arg in ("-u", key)], *command]
        if affinity:
            command = self.execution_argv(command)
        transport_timeout = self.timeout
        if self.host:
            command = ["env", *[f"{k}={v}" for k, v in variables.items()], *command]
            shell = shlex.join(command)
            if cwd:
                shell = "cd " + shlex.quote(str(cwd)) + " && " + shell
            # The remote process group owns the deadline. Killing only the SSH
            # client leaves native workers running and contaminates the next run.
            shell = shlex.join(["timeout", "--signal=TERM", "--kill-after=5s",
                                str(self.timeout) + "s", "sh", "-c", shell])
            command = ["ssh", *self.ssh_options, self.host, shell]
            transport_timeout += 15
            process_env = None
            process_cwd = None
        else:
            process_env = {**os.environ, **variables}
            process_cwd = cwd
        self.commands.append(command)
        result = subprocess.run(
            command,
            cwd=process_cwd,
            env=process_env,
            text=True,
            capture_output=True,
            timeout=transport_timeout,
        )
        if check and result.returncode:
            raise ExecutionError(stage, command, result)
        return result

    def upload(self, source, relative):
        source = Path(source)
        # Retain the exact worker/header inputs with the run, including SSH runs.
        import shutil

        snapshot = self.local / relative
        if source.resolve() != snapshot.resolve():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
        source = snapshot
        destination = str(Path(self.workspace) / relative)
        if not self.host:
            if source.resolve() != Path(destination).resolve():
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            return destination
        self.command(["mkdir", "-p", str(Path(destination).parent)], stage="prepare")
        with source.open("rb") as stream:
            result = subprocess.run(
                ["ssh", *self.ssh_options, self.host, "cat > " + shlex.quote(destination)],
                stdin=stream,
                capture_output=True,
                timeout=self.timeout,
            )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return destination

    def write(self, relative, text):
        path = self.local / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return self.upload(path, relative)

    def upload_many(self, files):
        """Snapshot and deploy a bundle with one transfer, retaining individual inputs."""
        import shutil

        snapshots = []
        destinations = {}
        for source, relative in files:
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
                raise ValueError("upload path must remain within the workspace")
            source = Path(source)
            snapshot = self.local / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != snapshot.resolve():
                shutil.copy2(source, snapshot)
            snapshots.append((snapshot, str(relative)))
            destinations[str(relative)] = str(Path(self.workspace) / relative)
        if not self.host:
            return destinations
        self.command(["mkdir", "-p", self.workspace], stage="prepare")
        command = ["ssh", *self.ssh_options, self.host, shlex.join([
            "tar", "xf", "-", "--no-same-owner", "--no-same-permissions", "-C", self.workspace])]
        self.commands.append(command)
        with tempfile.TemporaryFile() as stream:
            with tarfile.open(fileobj=stream, mode="w", dereference=True) as archive:
                for snapshot, relative in snapshots:
                    archive.add(snapshot, arcname=relative)
            stream.seek(0)
            result = subprocess.run(command, stdin=stream, capture_output=True, timeout=self.timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return destinations

    def download(self, source, destination):
        import shutil

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.host:
            with destination.open("wb") as stream:
                result = subprocess.run(
                    ["ssh", *self.ssh_options, self.host, "cat " + shlex.quote(str(source))],
                    stdout=stream,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                )
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        elif Path(source).resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

"""Replace a stable result directory only after retaining the new outcome."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil


class OutputRun:
    def __init__(self, destination):
        self.destination = Path(destination).expanduser().absolute()
        if self.destination.is_symlink():
            raise ValueError("a managed output destination cannot be a symlink")
        if self.destination.exists() and not self.destination.is_dir():
            raise ValueError("a managed output destination must be a directory")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.lock = (self.destination.parent / ("." + self.destination.name + ".lock")).open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(f"another run is updating {self.destination}") from None
        self.stage = self.destination.parent / ("." + self.destination.name + ".pending")
        if self.stage.exists():
            self.lock.close()
            raise ValueError(f"an interrupted result remains at {self.stage}; inspect it before retrying")
        self.stage.mkdir()

    def finish(self, error=None):
        try:
            self._finish(error)
        finally:
            self.lock.close()

    def _finish(self, error):
        original = str(self.stage)
        retained = str(self.destination)

        def relocate(value):
            if isinstance(value, str) and (value == original or value.startswith(original + "/")):
                return retained + value[len(original):]
            if isinstance(value, list):
                return [relocate(v) for v in value]
            if isinstance(value, dict):
                return {key: relocate(v) for key, v in value.items()}
            return value

        result = self.stage / "results.json"
        if result.exists():
            data = json.loads(result.read_text())
            for row in data["results"]:
                details = row["details"]
                details["output_locations"] = {"measured": original, "retained": retained}
                locations = [details]
                jit = details.get("jit")
                if jit:
                    locations.extend([jit, jit.get("runtime", {}), jit.get("disk_cache", {})])
                locations.extend(details[name] for name in ("replacement_plan", "model_validation")
                                 if isinstance(details.get(name), dict))
                for record in locations:
                    for key in ("artifact", "artifacts", "worker", "replay", "diagnostics", "paper_evidence",
                                "paper_validation_recheck", "validation", "model_paths", "package",
                                "source_package", "replacement_package", "candidate_package", "profile_path", "dot_inputs", "outputs"):
                        if key in record:
                            record[key] = relocate(record[key])
            result.write_text(json.dumps(data, indent=2))
        for manifest in self.stage.rglob("replacements.json"):
            data = json.loads(manifest.read_text())
            for key in ("package", "source_package", "replacements"):
                if key in data:
                    data[key] = relocate(data[key])
            manifest.write_text(json.dumps(data, indent=2))
        for captured in self.stage.rglob("dot-inputs.json"):
            data = json.loads(captured.read_text())
            data["buffers"] = relocate(data["buffers"])
            captured.write_text(json.dumps(data, indent=2) + "\n")
        for validation in self.stage.rglob("validation.json"):
            data = json.loads(validation.read_text())
            if "dot_inputs" in data:
                data["dot_inputs"] = relocate(data["dot_inputs"])
                if data.get("error", "").startswith("model dot mismatch; reproduce "):
                    data["error"] = data["error"].replace(original + "/", retained + "/")
                validation.write_text(json.dumps(data, indent=2) + "\n")
        (self.stage / "run-state.json").write_text(json.dumps({
            "status": "failed" if error else "completed", "error": str(error) if error else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "measurement_output": original, "retained_output": retained,
        }, indent=2))
        previous = self.destination.parent / ("." + self.destination.name + ".replaced")
        if previous.exists():
            raise ValueError(f"an interrupted replacement remains at {previous}")
        if self.destination.exists():
            if self.destination.is_symlink():
                raise ValueError("a managed output destination cannot be a symlink")
            self.destination.rename(previous)
        try:
            self.stage.rename(self.destination)
        except OSError:
            if previous.exists():
                previous.rename(self.destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        self.lock.close()

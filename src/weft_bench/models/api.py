"""The supported product path: normal llama.cpp versus parameter-selected Weft."""

from dataclasses import replace
from pathlib import Path
import tempfile

from .scenarios import ModelInputs, LlamaCpp
from .replacements import prepare_replacements, prepare_observer
from .llama import benchmark_model
from ..results import apply_baselines, read, save


def compare_model(
    model,
    *,
    target,
    candidate=None,
    baseline=None,
    replacements=None,
    inputs=None,
    phases="both",
    profile=False,
    reference=None,
    output=None,
    build=False,
    implementations=None,
    validate=False,
    validation_reference=None,
):
    selected = inputs or ModelInputs()
    if isinstance(selected, str):
        selected = ModelInputs.preset(selected)
    phase_inputs = selected.phases()
    output = Path(output or tempfile.mkdtemp(prefix="weft-model-"))
    settings = dict(target.settings.get("llama", {}))
    selected_names = implementations
    implementations = dict(settings.get("implementations", {}))
    if model not in settings.get("models", {}) and str(model).endswith(".gguf"):
        path = str(model)
        model = Path(path).stem
        settings["models"] = {**settings.get("models", {}), model: path}
    for name, value in (("normal", baseline), ("weft", candidate)):
        if value is not None:
            if not isinstance(value, LlamaCpp):
                raise TypeError(name + " must be LlamaCpp")
            implementations[name] = {
                "binary": value.binary,
                "env": value.env,
                **{
                    k: getattr(value, k)
                    for k in ("package", "source", "build_directory", "weft_include")
                    if getattr(value, k)
                },
            }
    mapping = replacements or (candidate.replacements if candidate else {})
    names = selected_names or (["weft"] if reference else ["normal", "weft"])
    if model not in settings.get("models", {}):
        raise ValueError(
            f"unknown model {model!r}; configure a model or pass its target-side GGUF path"
        )
    for name in names:
        if not implementations.get(name, {}).get("binary"):
            raise ValueError(f"configure the {name} llama.cpp binary")
    if "weft" in names and not mapping and not implementations["weft"].get("package"):
        raise ValueError(
            "select --replace, --replacement-file or a configured Weft package"
        )
    if mapping and "weft" not in names:
        raise ValueError("replacement mappings require a Weft candidate")
    manifest = None
    if mapping:
        manifest = prepare_replacements(mapping, target, output / "replacement")
        implementations["weft"] = {
            **implementations.get("weft", {}),
            "package": manifest["package"],
        }
    observer = (
        prepare_observer(
            implementations.get("weft", {}), target, output / "model-prepare"
        )
        if "weft" in names
        else None
    )
    settings.update(implementations=implementations, phases=phase_inputs)
    if selected.threads is not None:
        settings["threads"] = selected.threads
    target = replace(target, settings={**target.settings, "llama": settings})
    rows = benchmark_model(
        model,
        names,
        target,
        phases=phases,
        repetitions=selected.repetitions,
        profile=profile,
        output=output,
        build=build,
        validate=validate,
        validation_reference=validation_reference,
    )
    baselines = (
        read(reference)
        if reference
        else [
            replace(r, details=dict(r.details))
            for r in rows
            if r.implementation == "normal"
        ]
    )
    apply_baselines(rows, baselines, "normal")
    for row in rows:
        if row.implementation == "weft" and manifest:
            row.details["replacement_plan"] = manifest
        if row.implementation == "weft":
            row.details["observer"] = observer
    save(rows, output)
    return rows

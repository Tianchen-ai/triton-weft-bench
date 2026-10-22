"""Portable selected programs, workload suites and reference contracts."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).parent


def target_names():
    return sorted(path.stem for path in (ROOT / "targets").glob("*.json"))


def target_catalog(name):
    if name not in target_names():
        raise ValueError(f"unknown catalog target {name!r}; choose from {', '.join(target_names())}")
    return json.loads((ROOT / "targets" / (name + ".json")).read_text())


def resolve_source(source):
    if not source.startswith("package:"):
        return source
    relative = Path(source.removeprefix("package:"))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "kernels":
        raise ValueError("catalog source must name a kernel inside the Bench package")
    result = ROOT.parent / relative
    if not result.is_file():
        raise FileNotFoundError(result)
    return str(result)


def apply_catalog(settings, implementations):
    name = settings.get("catalog")
    if name is None:
        return settings, implementations
    selected = target_catalog(name)
    cases = selected["cases"]
    suites = {key: [{"case": case, "shape": cases[case]["shape"]} for case in names]
              for key, names in selected["suites"].items()}
    settings = {**settings, "operator_suites": {**suites, **settings.get("operator_suites", {})},
                "case_catalog": cases}
    implementations = deepcopy(implementations)
    candidate = implementations.setdefault("weft", {"factory": "weft_bench.adapters.triton_jit:TritonJitAdapter"})
    bindings = {}
    for case, record in cases.items():
        binding = {**record["candidate"], **candidate.get("bindings", {}).get(case, {})}
        if "source" in binding:
            binding["source"] = resolve_source(binding["source"])
        bindings[case] = binding
    candidate["bindings"] = {**candidate.get("bindings", {}), **bindings}
    return settings, implementations


def references():
    return json.loads((ROOT / "references.json").read_text())


def models():
    return json.loads((ROOT / "models.json").read_text())


def package_recipe(name):
    recipes = json.loads((ROOT / "packages.json").read_text())
    if name not in recipes:
        raise ValueError(f"unknown native package recipe {name!r}")
    recipe = recipes[name]
    for binding in recipe["bindings"].values():
        binding["source"] = resolve_source(binding["source"])
    return recipe


def describe_case(case, target):
    selected = target.settings.get("case_catalog", {}).get(case.name)
    reference_id = selected.get("reference_catalog") if selected else None
    candidate = target.implementations.get("weft", {})
    binding = {**candidate, **candidate.get("bindings", {}).get(case.name, {})}
    return {
        "case": case.name, "target": target.name,
        "workload": {"shape": selected["shape"] if selected else case.shape,
                     "scope": selected["scope"] if selected else case.scope, "dtype": case.dtype},
        "candidate": {key: binding[key] for key in (
            "source", "symbol", "signature", "constexprs", "backend_options", "compile_target",
            "workspaces", "persistent_inputs", "entry", "artifact") if key in binding},
        "retained_artifact_evidence": selected["artifact_evidence"] if selected else None,
        "reference": selected["reference"] if selected else None,
        "reference_details": {"id": reference_id, **references()[reference_id]} if reference_id else None,
        "baseline_result": target.settings.get("baselines", {}).get("operators"),
    }

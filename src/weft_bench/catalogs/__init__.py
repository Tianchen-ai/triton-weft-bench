"""Portable selected programs, workload suites and reference contracts."""
from copy import deepcopy
from importlib.util import find_spec
from importlib import import_module
import json
from pathlib import Path

ROOT = Path(__file__).parent


def provider_module(provider):
    return import_module(provider) if provider else None


def target_names(*, provider=None):
    if provider:
        return provider_module(provider).target_names()
    return sorted(path.stem for path in (ROOT / "targets").glob("*.json"))


def target_catalog(name, *, provider=None):
    if provider:
        return provider_module(provider).target_catalog(name)
    if name not in target_names():
        raise ValueError(f"unknown catalog target {name!r}; choose from {', '.join(target_names())}")
    return json.loads((ROOT / "targets" / (name + ".json")).read_text())


def resolve_source(source):
    if not source.startswith("package:"):
        return source
    relative = Path(source.removeprefix("package:"))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "kernels":
        raise ValueError("catalog source must name a kernel inside the Triton–Weft product")
    product = find_spec("triton")
    if product is None or not product.submodule_search_locations:
        raise ModuleNotFoundError("Install Triton–Weft to resolve the product kernel catalog")
    result = Path(next(iter(product.submodule_search_locations))).joinpath("kernels", "weft", *relative.parts[1:])
    if not result.is_file():
        raise FileNotFoundError(result)
    return str(result)


def apply_catalog(settings, implementations):
    name = settings.get("catalog")
    if name is None:
        return settings, implementations
    provider = settings.get("catalog_provider")
    selected = target_catalog(name, provider=provider)
    extension = provider_module(provider)
    candidate_name = extension.CANDIDATE if extension else "weft"
    candidate_factory = extension.FACTORY if extension else "weft_bench.adapters.triton_jit:TritonJitAdapter"
    source_resolver = extension.resolve_source if extension else resolve_source
    cases = selected["cases"]
    suites = {key: [{"case": case, "shape": cases[case]["shape"]} for case in names]
              for key, names in selected["suites"].items()}
    settings = {**settings, "operator_suites": {**suites, **settings.get("operator_suites", {})},
                "case_catalog": cases,
                "source_implementation": candidate_name,
                "verification": {"source_implementation": candidate_name, **settings.get("verification", {})}}
    if selected.get("projection_activation_contracts"):
        settings['projection_activation_contracts'] = {
            **selected['projection_activation_contracts'],
            **settings.get('projection_activation_contracts', {}),
        }
    implementations = deepcopy(implementations)
    candidate = implementations.setdefault(candidate_name, {"factory": candidate_factory})
    bindings = {}
    for case, record in cases.items():
        binding = {**record["candidate"], **candidate.get("bindings", {}).get(case, {})}
        if "source" in binding:
            binding["source"] = source_resolver(binding["source"])
        bindings[case] = binding
    candidate["bindings"] = {**candidate.get("bindings", {}), **bindings}
    for name, variant in selected.get("implementations", {}).items():
        base = {key: value for key, value in candidate.items() if key != "bindings"}
        implementations[name] = {**base, **deepcopy(variant), **implementations.get(name, {})}
    return settings, implementations


def references(*, provider=None):
    if provider:
        return provider_module(provider).references()
    return json.loads((ROOT / "references.json").read_text())


def models(*, provider=None):
    if provider:
        return provider_module(provider).models()
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
    candidate = target.implementations.get(target.settings.get("source_implementation", "weft"), {})
    binding = {**candidate, **candidate.get("bindings", {}).get(case.name, {})}
    return {
        "case": case.name, "target": target.name,
        "workload": {"shape": selected["shape"] if selected else case.shape,
                     "scope": selected["scope"] if selected else case.scope, "dtype": case.dtype},
        "candidate": {key: binding[key] for key in (
            "factory", "python", "compilation_host", "source", "symbol", "signature", "constexprs",
            "backend_options", "compile_target", "toolchain", "workspaces", "persistent_inputs", "entry", "grid",
            "num_cpu_threads", "artifact") if key in binding},
        "retained_artifact_evidence": selected["artifact_evidence"] if selected else None,
        "reference": selected["reference"] if selected else None,
        "reference_details": {"id": reference_id, **references(provider=target.settings.get("catalog_provider"))[reference_id]} if reference_id else None,
        "baseline_result": target.settings.get("baselines", {}).get("operators"),
    }

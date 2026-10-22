"""Directed source specialization checks in the existing runtime process."""
import json
import math

import numpy as np
import torch
import triton

from dense_view_probe import SENTINEL, strided_fixture


def bindings(compiled, selected):
    abi = compiled.metadata.kernel_abi
    assert compiled.metadata.source_bindings == selected, "compiler metadata lost source selection"
    assert abi["bindings"] == selected, "native ABI lost source selection"
    assert set(abi["shape_parameters"]) == {"M", "N"}, "source tile leaked into dynamic ABI geometry"
    assert not (set(selected) & {arg["name"] for arg in abi["arguments"]}), "source selection became a runtime argument"
    return {"source_bindings": selected, "kernel_abi": abi,
            "native_build": compiled.metadata.native_build}


def snapshot(root, name, **arrays):
    np.savez(root / (name + ".npz"), **{
        key: value.detach().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in arrays.items()})


def affine_call(root, label, shape, invoke, close):
    source, output, x, y, expected, guard = strided_fixture(*shape)
    original = source.clone()
    snapshot(root, label, source=original, expected=expected)
    compiled = invoke(x, y)
    snapshot(root, label, source=original, actual_source=source, expected=expected, actual=output)
    close(output, expected)
    assert torch.equal(source, original), "affine changed its input storage"
    assert bool(torch.all(output[guard] == SENTINEL)), "affine overwrote padding"
    return compiled, {"shape": list(shape), "x_stride": list(x.stride()), "y_stride": list(y.stride()),
                      "x_offset": x.storage_offset(), "y_offset": y.storage_offset()}


def export_bound(compiled, selected, directory):
    compiled.export(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["options"]["meta"] == selected, "export lost selected source parameters"
    for abi in (manifest["kernel"], manifest["grid"]["kernel"]):
        assert abi["bindings"] == selected, "exported ABI lost source binding"
        assert set(abi["shape_parameters"]) == {"M", "N"}
        assert not (set(selected) & {arg["name"] for arg in abi["arguments"]})
    return triton.load(directory)


def tiles_cache(root, options, close):
    from public_affine_strided import affine_strided

    kernels, records = {}, []
    for tile in (32, 64, 32):
        for shape in ((3, 65), (5, 97)):
            compiled, layout = affine_call(
                root, f"affine-b{tile}-{shape[0]}x{shape[1]}", shape,
                lambda x, y: affine_strided[(1,)](x, y, 1.25, bias=-0.5, B=tile, **options), close)
            if tile in kernels:
                assert compiled is kernels[tile], "dynamic geometry or repeated selection missed JIT cache"
            else:
                kernels[tile] = compiled
            records.append({"B": tile, **layout})
    assert kernels[32] is not kernels[64], "distinct source selections shared a compiled kernel"
    metadata = [bindings(kernels[tile], {"B": tile}) for tile in (32, 64)]
    artifacts = []
    for tile in (32, 64):
        directory = root / f"source-auto-affine-b{tile}"
        loaded = export_bound(kernels[tile], {"B": tile}, directory)
        for shape in ((3, 65), (5, 97)):
            affine_call(root, f"aot-b{tile}-{shape[0]}x{shape[1]}", shape,
                        lambda x, y: loaded[(1,)](x, y, 1.25, bias=-0.5), close)
        artifacts.append(str(directory))
    return {"public_source": affine_strided.fn.__code__.co_filename, "layouts": records,
            "compile_cache_reused_across_geometry": True, "source_selections_isolated": True,
            "return_to_b32_reused": True, "aot_without_runtime_B": True,
            "artifacts": artifacts, "candidates": metadata}


def binding_rejections(root, options, close, target):
    from triton.backends.compiler import CPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.weft.interface import describe
    from compile_triton import save_backend_failures
    from source_auto_kernels import unbound, finite_domain
    from kernels import affine

    interface = describe(finite_domain, {}, signature={})
    domains = interface["source_parameters"]
    assert len(domains) == 1, "finite-domain probe should expose one source parameter"
    parameter, domain = next(iter(domains.items()))
    assert list(domain) == [32, 64], "frontend did not preserve the declared finite domain"
    (root / "finite-domain-interface.json").write_text(json.dumps(interface, indent=2))

    def compile_selection(kernel, meta, constexprs=None):
        return triton.compile(ASTSource(kernel, {}, constexprs=constexprs), target=CPUTarget(backend="weft", **target),
                              options={**options, "meta": meta})

    records = []
    for name, kernel, meta, constexprs, phrase in (
        ("missing", unbound, {}, None, "requires one concrete meta binding"),
        ("nonpositive", unbound, {"B": 0}, None, "source meta binding must be name=positive_integer"),
        ("outside-domain", finite_domain, {parameter: 48}, None, "source auto binding is outside its declared finite domain"),
        ("constexpr-conflict", affine, {"BLOCK": 64}, {"BLOCK": 32}, "conflicts with constexpr value"),
        ("constexpr-nonpositive", affine, {}, {"BLOCK": 0}, "source meta binding must be name=positive_integer"),
    ):
        try:
            compile_selection(kernel, meta, constexprs)
        except Exception as error:
            diagnostic = str(error)
            saved = save_backend_failures(error, root, "source-auto-" + name)
            if phrase not in diagnostic.lower():
                raise AssertionError(f"{name} rejected at an unexpected boundary: {diagnostic}") from error
            records.append({"case": name, "meta": meta, "constexprs": constexprs,
                            "diagnostic": diagnostic, "backend_diagnostics": saved})
        else:
            raise AssertionError(name + " source binding was accepted")
    compiled = compile_selection(finite_domain, {parameter: 32})
    details = bindings(compiled, {parameter: 32})
    x = torch.arange(195, dtype=torch.float32).reshape(3, 65) / 2048
    original, y = x.clone(), torch.full_like(x, SENTINEL)
    snapshot(root, "finite-domain-legal", source=original, expected=original)
    compiled[(1,)](x, y)
    snapshot(root, "finite-domain-legal", source=original, actual_source=x, expected=original, actual=y)
    close(y, original)
    assert torch.equal(x, original)
    return {"finite_domain": domains, "legal_selection": details, "rejections": records}


def autotune_state(root, options, close):
    from public_state_update import state_update

    configs = [triton.Config({"B": tile}, num_cpu_threads=1) for tile in (32, 64)]
    tuned = triton.autotune(configs=configs, key=[])(state_update)
    # Config owns num_cpu_threads; do not pass the same keyword twice to Autotuner.run.
    tune_options = {key: value for key, value in options.items() if key != "num_cpu_threads"}
    records = []
    for m, n in ((3, 65), (5, 97)):
        storage = torch.arange(m*(n+1), dtype=torch.float32).reshape(m, n+1)/1000
        initial = storage.clone()
        state, delta = storage[:, :-1], storage[:, 1:]
        expected = initial.clone()
        expected[:, :-1] = initial[:, :-1]*0.75 + initial[:, 1:]
        snapshot(root, f"state-{m}x{n}", initial=initial, expected=expected)
        for tile in (32, 64):
            storage.copy_(initial)
            candidate = state_update[(1,)](state, delta, 0.75, B=tile, **options)
            snapshot(root, f"state-b{tile}-{m}x{n}", initial=initial, expected=expected, actual=storage)
            close(storage, expected)
            bindings(candidate, {"B": tile})
        storage.copy_(initial)
        before = len(tuned.cache)
        winner = tuned[(1,)](state, delta, 0.75, **tune_options)
        snapshot(root, f"state-{m}x{n}", initial=initial, expected=expected, actual=storage)
        close(storage, expected)
        assert len(tuned.cache) == before + 1, "changed geometry reused an unrelated tuning entry"
        assert not tuned.candidate_errors, "a declared legal source candidate was rejected"
        assert set(tuned.configs_timings) == set(configs)
        assert all(math.isfinite(values[0]) for values in tuned.configs_timings.values())
        chosen = dict(tuned.best_config.kwargs)
        details = bindings(winner, chosen)
        timings = tuned.configs_timings
        storage.copy_(initial)
        cached = tuned[(1,)](state, delta, 0.75, **tune_options)
        snapshot(root, f"state-cached-{m}x{n}", initial=initial, expected=expected, actual=storage)
        close(storage, expected)
        assert cached is winner and len(tuned.cache) == before+1
        assert tuned.configs_timings is timings, "identical workload repeated tuning"
        records.append({"shape": [m, n], "relative_alias_offset": 1, "winner": chosen,
                        "cache_entries": len(tuned.cache), "caller_updated_once": True,
                        "legal_candidates_checked_independently": [32, 64], **details})
    artifact = root / "source-auto-autotune-winner"
    loaded = export_bound(winner, chosen, artifact)
    storage.copy_(initial)
    loaded[(1,)](state, delta, 0.75)
    snapshot(root, "state-aot", initial=initial, expected=expected, actual=storage)
    close(storage, expected)
    return {"public_source": state_update.fn.__code__.co_filename, "scenarios": records,
            "real_triton_autotune": True, "performance_claim": False,
            "loaded_winner_updates_once": True, "artifact": str(artifact)}


def run_scenario(name, root, options, close, target):
    if name == "tiles_cache":
        return tiles_cache(root, options, close)
    if name == "bindings":
        return binding_rejections(root, options, close, target)
    if name == "autotune":
        return autotune_state(root, options, close)
    raise ValueError("unknown source-auto scenario " + name)

"""Public GGML/Weft comparisons; deployment is delegated to the outer runner."""

from dataclasses import replace

from .inputs import Inputs, resolve_inputs
from .libraries import GGML, WeftKernel, candidate_configuration
from .types import Measurement


def benchmark(
    case,
    implementations,
    target,
    measurement=None,
    *,
    output=None,
    inputs=None,
    save_outputs=False,
):
    """Compatibility entry for check/history clients using implementation names."""
    from .runner import mode, run_operator

    if inputs is not None:
        inputs = resolve_inputs(inputs)
    if any(mode(name, target) is None for name in implementations):
        if measurement and measurement.check_only:
            raise NotImplementedError("check_only requires a native worker implementation")
        from .compat import benchmark as command_benchmark

        rows = []
        for name in implementations:
            if mode(name, target) is None:
                rows.extend(
                    command_benchmark(case, [name], target, measurement, output=output)
                )
            else:
                rows.extend(
                    run_operator(
                        case,
                        [name],
                        target,
                        measurement,
                        output=output,
                        inputs=inputs,
                        save_outputs=save_outputs,
                    )
                )
        return rows
    if inputs is None and case.data.get("inputs"):
        saved = case.data["inputs"]
        inputs = Inputs(
            preset=saved["preset"],
            shape=saved["shape"],
            seed=saved["seed"],
            parameters=saved["parameters"],
            buffers={k: v["path"] for k, v in saved["buffers"].items()},
        )
    return run_operator(
        case,
        implementations,
        target,
        measurement,
        output=output,
        inputs=inputs,
        save_outputs=save_outputs,
    )


def compare(
    operation,
    *,
    target,
    candidate=None,
    baseline=None,
    inputs=None,
    measurement=None,
    output=None,
    save_outputs=False,
):
    """Compare a Weft source/artifact with a configured reference and shared inputs.

    candidate is a WeftKernel or file.py:symbol/artifact path. baseline selects a
    GGML build or a configured implementation name; omit it to use GGML. Target supplies external
    compilation/deployment parameters, not workload semantics.
    """
    from .operators import get
    from .results import apply_baselines

    if isinstance(candidate, str):
        candidate = WeftKernel.parse(candidate)
    if candidate is not None and not isinstance(candidate, WeftKernel):
        raise TypeError("candidate must be a WeftKernel")
    selected = resolve_inputs(inputs)
    case = selected.resolve(get(operation) if isinstance(operation, str) else operation)
    settings = dict(target.settings)
    reference_name = "ggml"
    if baseline is not None:
        if isinstance(baseline, str):
            reference_name = baseline
            if reference_name == "weft":
                raise ValueError("reference implementation must have a distinct name from weft")
        elif isinstance(baseline, GGML):
            settings["ggml"] = baseline.configuration()
        else:
            raise TypeError("baseline must be a GGML library or configured implementation name")
    implementations = dict(target.implementations)
    if candidate:
        implementations["weft"] = candidate_configuration(case, implementations.get("weft", {}), candidate)
    target = replace(target, settings=settings, implementations=implementations)
    rows = benchmark(
        case,
        [reference_name, "weft"],
        target,
        measurement,
        output=output,
        inputs=selected,
        save_outputs=save_outputs,
    )
    return apply_baselines(
        rows,
        [
            replace(r, details=dict(r.details))
            for r in rows
            if r.implementation == reference_name
        ],
        reference_name,
    )

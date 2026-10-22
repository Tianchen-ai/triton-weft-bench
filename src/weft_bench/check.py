"""One daily comparison: fixed reference, explicit checkpoint, current candidate."""

import csv
import json
from pathlib import Path
import shlex

from .api import benchmark
from .diagnostics import differences
from .operators import get, SUITES
from .results import comparison_basis, contracts_match, execution_issue, read, save
from .types import Measurement


def match(row, candidates, implementation=None):
    return [
        other
        for other in candidates
        if contracts_match(row, other)
        and (implementation is None or other.implementation == implementation)
        and other.status == "ok"
        and other.median_ms is not None
    ]


def observed(row):
    build = row.details.get("build", {})
    artifact = row.details.get("artifacts", {}).get("weft", "")
    if not build and artifact:
        request = Path(artifact).parent / "compile.json"
        if request.is_file():
            build = json.loads(request.read_text())
    if artifact:
        effective = Path(artifact).parent / "effective-options.json"
        if effective.is_file():
            build = {**build, "effective_options": json.loads(effective.read_text())}
    return {
        "implementation": row.implementation,
        "ms": row.median_ms,
        "status": row.status,
        "build": build,
        "artifact": artifact,
        "error": row.error,
        "measurement": row.details.get("measurement", {}),
        "native_build": row.details.get("native_build", {}),
        "origin": row.details.get("imported_from", row.details.get("replay")),
    }


def compare(
    row,
    references,
    previous,
    *,
    threshold=0.03,
    reference_impl=None,
    previous_impl=None,
):
    row.details.pop("comparison_basis", None)
    row.details.pop("comparison_note", None)
    row.details.pop("comparison_differences", None)
    row.details.pop("baseline_median_ms", None)
    row.details.pop("baseline_scope", None)
    comparison = {
        "reference": None,
        "previous": None,
        "ref_speedup": None,
        "previous_speedup": None,
    }
    for label, candidates, impl in (
        ("reference", references, reference_impl),
        ("previous", previous, previous_impl),
    ):
        matches = match(row, candidates, impl)
        if len(matches) == 1:
            comparison[label] = observed(matches[0])
            basis = comparison_basis(row, matches[0])
            if basis:
                comparison[label]["comparison_basis"] = basis
            if row.median_ms and row.status == "ok":
                comparison[
                    "ref_speedup" if label == "reference" else "previous_speedup"
                ] = (matches[0].median_ms / row.median_ms)
                if label == "reference":
                    row.details["baseline_median_ms"] = matches[0].median_ms
                    row.details["baseline_scope"] = matches[0].scope
                    if basis:
                        row.details["comparison_basis"] = basis
        else:
            related = [
                x
                for x in candidates
                if x.case == row.case
                and x.target == row.target
                and (impl is None or x.implementation == impl)
            ]
            comparison[label + "_note"] = (
                "ambiguous checkpoint" if len(matches) > 1 else "missing comparison"
            )
            failed = [
                x for x in related if contracts_match(row, x) and x.status != "ok"
            ]
            if failed and not matches:
                comparison[label + "_note"] = failed[-1].status
                comparison[label + "_failure"] = observed(failed[-1])
            elif related and not matches:
                comparison[label + "_note"] = execution_issue(related[0]) or "scenario changed"
                comparison[label + "_differences"] = differences(
                    {
                        "shape": related[0].shape,
                        "scope": related[0].scope,
                        "measurement": related[0].details.get("measurement"),
                        "execution": related[0].details.get("execution"),
                        "input_policy": related[0].details.get("input_policy"),
                        "dtype": related[0].details.get("dtype"),
                        "phase": related[0].details.get("phase"),
                    },
                    {
                        "shape": row.shape,
                        "scope": row.scope,
                        "measurement": row.details.get("measurement"),
                        "execution": row.details.get("execution"),
                        "input_policy": row.details.get("input_policy"),
                        "dtype": row.details.get("dtype"),
                        "phase": row.details.get("phase"),
                    },
                )
    issue = execution_issue(row)
    if issue:
        comparison["reference_note"] = issue
        comparison["previous_note"] = issue
    if row.status != "ok":
        conclusion = row.status
    elif issue:
        conclusion = "execution_mismatch"
    elif comparison["previous_speedup"] is not None:
        value = comparison["previous_speedup"]
        conclusion = (
            "improved"
            if value > 1 + threshold
            else "regressed" if value < 1 - threshold else "close"
        )
    else:
        conclusion = "previous_missing"
    comparison["conclusion"] = conclusion
    comparison["notes"] = "; ".join(
        f"{key}: {comparison[key+'_note']}"
        for key in ("reference", "previous")
        if key + "_note" in comparison
    )
    row.details["comparison"] = comparison
    if comparison["notes"]:
        row.details["comparison_note"] = comparison["notes"]
    row.speedup = comparison["ref_speedup"]
    row.baseline = (
        comparison["reference"]["implementation"] if comparison["reference"] else None
    )
    return row


def check(target, *, output, cases=None, compiler=None):
    from dataclasses import replace

    settings = target.settings.get("check", {})
    if not settings.get("reference") or not settings.get("previous"):
        raise ValueError(
            "check needs explicit reference and previous result paths in target.check"
        )
    reference_path = Path(settings["reference"]).expanduser()
    references = read(reference_path) if reference_path.exists() else []
    previous = read(settings["previous"])
    reference_impl = settings.get("reference_impl", "ggml")
    previous_impl = settings.get("previous_impl", "weft")
    current_impl = settings.get("current_impl", "weft")
    if compiler:
        target = replace(
            target,
            implementations={
                **target.implementations,
                current_impl: {
                    **target.implementations.get(current_impl, {}),
                    "compiler": compiler,
                },
            },
        )
    names = cases or [
        *SUITES["quick"],
        "q4_k_dot_rows",
        *(
            ["ime_centered"]
            if "spacemit-ime1" in target.settings.get("capabilities", [])
            else []
        ),
    ]
    rows = []
    for name in names:
        measurement = Measurement(
            **settings.get("measurements", {}).get(
                name,
                (
                    {"cache": "cold", "warmup": 1, "samples": 10}
                    if name == "q4_k_dot_rows"
                    else {}
                ),
            )
        )
        case = get(name, settings.get("shapes", {}).get(name))
        print(f"Checking {name} on {target.name}", flush=True)
        current = benchmark(
            case, [current_impl], target, measurement, output=Path(output) / "artifacts"
        )[0]
        same = match(current, references, reference_impl)
        if not same:
            reference = benchmark(
                case,
                [reference_impl],
                target,
                measurement,
                output=Path(output) / "reference-artifacts",
            )[0]
            references.append(reference)
            reference.details["reference_action"] = "measured_missing"
            save(references, reference_path)
        current.details["reference_action"] = "reused" if same else "measured_missing"
        rows.append(
            compare(
                current,
                references,
                previous,
                threshold=settings.get("threshold", 0.03),
                reference_impl=reference_impl,
                previous_impl=previous_impl,
            )
        )
        save(rows, output)
        save_comparison(rows, output)
    from .models.hotspots import attach_saved_profiles

    attach_saved_profiles(rows, target)
    save(rows, output)
    return rows


def columns(row):
    c = row.details.get("comparison", {})
    number = lambda v: f"{v:.6f}" if v is not None else "—"
    ratio = lambda v: f"{v:.3f}x" if v is not None else "—"
    return [
        row.case,
        row.target,
        ",".join(f"{k}={v}" for k, v in row.shape.items()),
        number((c.get("reference") or {}).get("ms")),
        number((c.get("previous") or {}).get("ms")),
        number(row.median_ms),
        ratio(c.get("ref_speedup")),
        ratio(c.get("previous_speedup")),
        c.get("conclusion", row.status),
        c.get("notes", ""),
        row.details.get("compiler_stage") or row.details.get("failure_stage") or "",
    ]


HEADERS = [
    "case",
    "target",
    "shape",
    "ref_ms",
    "previous_ms",
    "current_ms",
    "vs_ref",
    "vs_previous",
    "conclusion",
    "notes",
    "failure_stage",
]


def table(rows):
    all_rows = [HEADERS, *[columns(r) for r in rows]]
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(HEADERS))]
    return "\n".join(
        "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
        for row in all_rows
    )


def save_comparison(rows, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([*HEADERS, "replay"])
        for row in rows:
            writer.writerow(
                [
                    *columns(row),
                    shlex.join(
                        [
                            "weft-bench",
                            "replay",
                            str(directory.resolve()),
                            "--case",
                            row.case,
                            "--impl",
                            row.implementation,
                        ]
                    ),
                ]
            )

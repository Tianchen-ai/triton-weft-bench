from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

from .api import benchmark
from .loading import load_config, select_target, select_model
from .operators import SUITES, OperatorSelection, catalog, get, configured_suites
from .results import apply_baselines, read, save, table
from .types import Measurement
from .execution import ExecutionError


def parser():
    root = argparse.ArgumentParser(
        description="Compare Weft kernels with GGML or replace selected llama.cpp operations"
    )
    commands = root.add_subparsers(dest="command", required=True)
    environment = commands.add_parser("environment", help="observe CPU/OS state, affinity and NUMA without timing an operator")
    environment.add_argument("--target", required=True)
    environment.add_argument("--config")
    environment.add_argument("--output")
    capture = commands.add_parser("capture-model-reference", help="capture reusable normal logits without model timing or self-comparison")
    capture.add_argument("model")
    capture.add_argument("--config")
    capture.add_argument("--target", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--attach-run", help="associate the reference with an existing model result, preserving timings")
    validate = commands.add_parser("validate-model", help="check selected model logits without performance timing")
    validate.add_argument("model")
    validate.add_argument("--config")
    validate.add_argument("--target", required=True)
    validate.add_argument("--impl", default="weft")
    validate.add_argument("--output", required=True)
    validate.add_argument("--validation-reference", required=True)
    validate.add_argument("--check-dots", action="store_true", help="compare replaced native rows and capture the first mismatch")
    listing = commands.add_parser(
        "list",
        help="select GGML workloads, input presets or llama.cpp replacement roles",
    )
    listing.add_argument(
        "--kind", choices=("operators", "inputs", "replacements", "suites", "targets", "kernels", "baselines", "models"), default="operators"
    )
    listing.add_argument("--config", help="include suites from an external target configuration")
    listing.add_argument("--target")
    show = commands.add_parser("show", help="show a workload input/output contract")
    show.add_argument("case")
    show.add_argument("--inputs", default="default")
    show.add_argument("--target")
    show.add_argument("--config")
    prepare = commands.add_parser(
        "prepare-model",
        help="prepare a llama.cpp package from explicit replacement parameters",
    )
    prepare.add_argument("--target", required=True)
    prepare.add_argument("--config")
    prepare.add_argument("--model", help="select the configured model's package build environment")
    prepare.add_argument("--model-profile", help="select a normal backend and replacement policy")
    prepare.add_argument("--replace", action="append", default=[])
    prepare.add_argument("--replacement-file")
    prepare.add_argument("--output", required=True)
    daily = commands.add_parser(
        "check", help="compare current, fixed ref and selected previous checkpoint"
    )
    daily.add_argument("--target", required=True)
    daily.add_argument("--config")
    daily.add_argument("--case", action="append")
    daily.add_argument("--compiler")
    daily.add_argument("--output")
    verify = commands.add_parser("verify", help="enumerate and resume configured compiler, numerical and runtime verification")
    verify.add_argument("--target", required=True)
    verify.add_argument("--config")
    verify.add_argument("--list", action="store_true")
    verify.add_argument("--coverage", action="store_true", help="map the configured source inventory to cases and helper callers")
    verify.add_argument("--group", action="append")
    verify.add_argument("--case", action="append", help="stable verification id or glob, e.g. operators/q4_*")
    verify.add_argument("--output")
    verify.add_argument("--resume", action="store_true")
    verify.add_argument("--rerun", choices=("failed", "all"))
    verify.add_argument("--prepare-only", action="store_true", help="compile selected sources without executing on the target")
    verify.add_argument("--workers", type=int, default=1, help="host compilation workers for --prepare-only; device execution stays serial")
    verify.add_argument("--stop-after-current", action="store_true",
                        help="ask the run in --output to stop after its current case finishes")
    for action in ("explain", "replay"):
        command = commands.add_parser(action)
        command.add_argument("run")
        command.add_argument("--case", required=True)
        command.add_argument(
            "--impl", help="select one implementation when a run contains several"
        )
        if action == "replay":
            command.add_argument("--compiler")
            command.add_argument(
                "--source",
                help="use current author source instead of saved canonical IR",
            )
            command.add_argument("--output")
    history = commands.add_parser(
        "import-history", help="import old Weft and reference observations together"
    )
    history.add_argument("source")
    history.add_argument("--measurement")
    history.add_argument("--output", required=True)
    checkpoint = commands.add_parser(
        "checkpoint",
        help="combine explicitly selected observations without remeasurement",
    )
    checkpoint.add_argument("runs", nargs="+")
    checkpoint.add_argument("--impl", required=True)
    checkpoint.add_argument("--output", required=True)
    info = commands.add_parser(
        "model-info", help="read target GGUF tensor metadata, without loading weights"
    )
    info.add_argument("model")
    info.add_argument("--config")
    info.add_argument("--target", required=True)
    info.add_argument("--output", required=True)
    hotspots = commands.add_parser(
        "hotspots", help="link saved profile symbols to cases without rerunning models"
    )
    hotspots.add_argument("profile")
    hotspots.add_argument("--target", required=True)
    hotspots.add_argument("--phase", choices=("prefill", "decode"), required=True)
    hotspots.add_argument("--model-info")
    hotspots.add_argument("--operators")
    hotspots.add_argument("--config")
    hotspots.add_argument("--model")
    hotspots.add_argument("--output")
    observe = commands.add_parser(
        "observe-llama",
        help="add optional phase/counter observers to an external llama-bench source tree",
    )
    observe.add_argument("source")
    observe.add_argument(
        "--weft-include",
        help="directory containing weft/ggml.h, only for an integrated build",
    )
    for kind in ("op", "e2e"):
        command = commands.add_parser(kind)
        command.add_argument("case", nargs="?")
        command.add_argument("--config")
        command.add_argument("--target", required=True)
        command.add_argument(
            "--impl", help="optional explicit implementation selection"
        )
        command.add_argument("--output")
        command.add_argument(
            "--baseline", help="reuse results JSON, without rerunning baseline"
        )
        command.add_argument(
            "--baseline-impl", help="which implementation to use as baseline"
        )
        if kind == "op":
            command.add_argument(
                "--weft", help="candidate file.py:symbol or artifact directory"
            )
            command.add_argument(
                "--inputs", default="default", help="preset name or input JSON file"
            )
            command.add_argument("--seed", type=int)
            command.add_argument("--save-outputs", action="store_true")
            command.add_argument("--check-only", action="store_true", help="prepare and check one native invocation without performance sampling")
            command.add_argument("--suite", help="built-in or configured operator_suites selection")
            command.add_argument(
                "--shape", help="explicit dimensions, e.g. rows=1,K=4096"
            )
            command.add_argument("--warmup", type=int, default=3)
            command.add_argument("--warmup-ms", type=float, default=200.0,
                                 help="minimum native warmup duration; --warmup 0 disables warmup")
            command.add_argument("--samples", type=int, default=5)
            command.add_argument("--batch-ms", type=float, default=10)
            command.add_argument("--cache", choices=("reuse", "cold"), default="reuse")
            command.add_argument("--eviction-bytes", type=int, default=64 * 1024 * 1024)
        else:
            command.add_argument(
                "--scenario", choices=("default", "smoke"), default=None
            )
            command.add_argument("--replace", action="append", default=[])
            command.add_argument("--replacement-file")
            command.add_argument("--prefill", type=int)
            command.add_argument("--decode", type=int)
            command.add_argument("--depth", type=int)
            command.add_argument("--threads", type=int)
            command.add_argument(
                "--phase", choices=("prefill", "decode", "both"), default="both"
            )
            command.add_argument("--model-profile", help="configured normal-backend and replacement policy")
            command.add_argument("--profile", action="store_true")
            command.add_argument(
                "--build",
                action="store_true",
                help="run externally configured build commands once before timing",
            )
            command.add_argument("--repetitions", type=int)
            command.add_argument("--validate", action="store_true", help="compare a small fixed-prompt logit sequence outside timing")
            command.add_argument("--validation-reference", help="reuse or create a fixed numerical reference directory")
    compare = commands.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--baseline-impl")
    compare.add_argument("--output")
    legacy = commands.add_parser(
        "import-baseline", help="import historical results without remeasurement"
    )
    legacy.add_argument("source")
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--kind", choices=("operators", "llama"), default="operators")
    legacy.add_argument("--target")
    legacy.add_argument("--profile", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    output_run = None
    run_error = None
    try:
        if args.command == "capture-model-reference":
            from .models.validation import capture_model_reference, attach_validation

            target = select_target(load_config(args.config), args.target)
            target, _ = select_model(target, args.model)
            report = capture_model_reference(args.model, target, output=args.output)
            if args.attach_run:
                rows = read(args.attach_run)
                selected = [r for r in rows if r.case in (args.model + "/prefill", args.model + "/decode") and r.implementation == "normal"]
                if not selected:
                    raise ValueError("the selected run has no normal results for this model")
                attach_validation(selected, report, "normal")
                save(rows, args.attach_run)
            print(json.dumps(report, indent=2))
            return int(report["status"] != "ok")
        if args.command == "validate-model":
            from .models.validation import validate_model

            target = select_target(load_config(args.config), args.target)
            target, _ = select_model(target, args.model)
            if args.check_dots:
                target.settings["llama"].setdefault("validation", {})["check_dots"] = True
            report = validate_model(args.model, target, args.impl, output=Path(args.output),
                                    reference_output=Path(args.validation_reference))
            print(json.dumps(report, indent=2))
            return int(report["status"] != "ok")
        if args.command == "environment":
            from .environment import inspect_environment

            target = select_target(load_config(args.config), args.target)
            output = args.output or Path(target.settings.get("output_root", "runs")) / (args.target + "-environment")
            print(json.dumps(inspect_environment(target, output), indent=2))
            return 0
        if args.command == "verify":
            from .verification import plan, verify
            if args.stop_after_current:
                if not args.output:
                    raise ValueError("--stop-after-current requires the active run's --output")
                directory = Path(args.output).expanduser().resolve()
                current = json.loads((directory / "verification.json").read_text())
                if current["target"] != args.target:
                    raise ValueError("verification output belongs to a different target")
                (directory / ".stop-after-current").write_text(args.target + "\n")
                print("Requested stop after the current case; no device call was interrupted.")
                return 0
            target = select_target(load_config(args.config), args.target)
            if args.coverage:
                from .verification.coverage import coverage
                result = coverage(target, output=args.output)
                print(json.dumps(result["counts"] if args.output else result, indent=2))
                if args.output:
                    print(Path(args.output) / "coverage.json")
                return 0
            if args.list:
                print(json.dumps({"target": target.name, "entries": plan(target, groups=args.group, cases=args.case)}, indent=2))
                return 0
            if not args.output:
                raise ValueError("verify requires a stable --output directory")
            if args.prepare_only:
                from .verification.prepare import prepare as prepare_verification
                rows = prepare_verification(target, output=args.output, groups=args.group, cases=args.case,
                                            workers=args.workers, resume=args.resume, rerun=args.rerun)
                print(Path(args.output) / "preparation.json")
                return int(any(row["status"] not in ("compiled", "rejected") for row in rows))
            if args.workers != 1:
                raise ValueError("--workers applies to --prepare-only; device execution is serial")
            ledger = verify(target, output=args.output, groups=args.group, cases=args.case,
                            resume=args.resume, rerun=args.rerun)
            print(json.dumps({"selected": ledger["selection_counts"], "all": ledger["counts"]}))
            print(Path(args.output) / "verification.json")
            if ledger.get("stopped_after_current"):
                print("Stopped at a case boundary. Use --resume to continue.")
                return 0
            return int(any(ledger["entries"][key]["status"] not in ("passed", "not_applicable") for key in ledger["selection"]))
        if args.command == "show":
            from dataclasses import asdict
            from .inputs import resolve_inputs
            from .workloads import parameter_defaults, ports

            inputs = resolve_inputs(args.inputs)
            case = inputs.resolve(get(args.case))
            if args.target:
                from .catalogs import describe_case
                target = select_target(load_config(args.config), args.target)
                print(json.dumps(describe_case(case, target), indent=2))
                return 0
            print(
                json.dumps(
                    {
                        "operation": case.name,
                        "scope": case.scope,
                        "shape": case.shape,
                        "ports": [asdict(p) for p in ports(case)],
                        "parameters": {**parameter_defaults(case), **inputs.parameters},
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "prepare-model":
            from .models.replacements import prepare_replacements, parse_mapping

            mapping = (
                json.loads(Path(args.replacement_file).read_text())
                if args.replacement_file
                else {}
            )
            mapping.update(parse_mapping(args.replace))
            target = select_target(load_config(args.config), args.target)
            if target.settings.get("model_profiles"):
                if not args.model:
                    raise ValueError("prepare-model requires --model to select a package build environment")
                target, _ = select_model(target, args.model, args.model_profile)
            if not mapping:
                mapping = {role: "default" for role in target.settings.get("llama", {}).get("triton", {}).get("bindings", {})}
            print(
                json.dumps(prepare_replacements(mapping, target, args.output), indent=2)
            )
            return 0
        if args.command == "model-info":
            from .execution import Executor
            from .models import gguf_info

            target = select_target(load_config(args.config), args.target)
            target, _ = select_model(target, args.model)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            executor = Executor(target, output.parent / "metadata-reader")
            script = executor.upload(Path(gguf_info.__file__), "gguf_info.py")
            path = target.settings["llama"]["models"][args.model]
            result = executor.command(
                [
                    target.settings.get("deployment", {}).get("python", "python3"),
                    script,
                    path,
                ]
            )
            data = json.loads(result.stdout)
            output.write_text(json.dumps(data, indent=2))
            print(output)
            return 0
        if args.command == "hotspots":
            from .models.hotspots import link_hotspots

            data = json.loads(Path(args.profile).read_text())
            symbols = (
                data.get("symbols", data.get("hotspots", []))
                if isinstance(data, dict)
                else data
            )
            info = (
                json.loads(Path(args.model_info).read_text())
                if args.model_info
                else None
            )
            sites = {}
            if args.config and args.model:
                target = select_target(load_config(args.config), args.target)
                sites = (
                    target.settings.get("llama", {})
                    .get("call_sites", {})
                    .get(args.model, {})
                    .get(args.phase, {})
                )
            rows = link_hotspots(
                symbols,
                target=args.target,
                phase=args.phase,
                model_info=info,
                call_sites=sites,
                operator_rows=read(args.operators) if args.operators else [],
            )
            text = json.dumps(rows, ensure_ascii=False, indent=2)
            if args.output:
                Path(args.output).write_text(text)
            print(text)
            return 0
        if args.command == "checkpoint":
            from .results import contract

            rows = []
            keys = set()
            for run in args.runs:
                for row in read(run):
                    if row.implementation != args.impl or row.status != "ok":
                        continue
                    key = contract(row)
                    if key in keys:
                        raise ValueError(
                            f"duplicate checkpoint observation {row.case}; select one source explicitly"
                        )
                    keys.add(key)
                    rows.append(row)
            if not rows:
                raise ValueError("no matching observations")
            print(save(rows, args.output))
            return 0
        if args.command == "import-history":
            from .legacy import import_history

            rows = import_history(args.source, measurement=args.measurement)
            print(save(rows, args.output))
            print(
                f"{sum(r.status=='ok' for r in rows)} mapped observations; {sum(r.status=='unmapped' for r in rows)} records preserved as unmapped"
            )
            return 0
        if args.command in ("explain", "replay"):
            from .diagnostics import explain, replay

            rows = [
                r
                for r in read(args.run)
                if r.case == args.case
                and (not args.impl or r.implementation == args.impl)
            ]
            if not rows:
                raise ValueError("case not found in saved run")
            if args.command == "explain":
                print(
                    json.dumps([explain(r) for r in rows], ensure_ascii=False, indent=2)
                )
                return 0
            if len(rows) != 1:
                raise ValueError("select one implementation with --impl")
            output = args.output or (
                "runs/replay-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            )
            results = replay(
                rows[0],
                compiler=args.compiler,
                source=args.source,
                output=Path(output) / "artifacts",
            )
            print(table(results))
            print(save(results, output))
            return int(any(r.status != "ok" for r in results))
        if args.command == "check":
            from .check import check, table as comparison_table

            target = select_target(load_config(args.config), args.target)
            output = args.output or (
                "runs/check-"
                + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                + "-"
                + args.target
            )
            rows = check(target, output=output, cases=args.case, compiler=args.compiler)
            print(comparison_table(rows))
            print(Path(output) / "comparison.csv")
            import shlex

            print(
                "Inspect: "
                + shlex.join(
                    [
                        "weft-bench",
                        "explain",
                        str(Path(output).resolve()),
                        "--case",
                        "CASE",
                    ]
                )
            )
            return int(any(r.status != "ok" for r in rows))
        if args.command == "observe-llama":
            from .models.observe import instrument

            print(
                json.dumps(
                    instrument(args.source, weft_include=args.weft_include), indent=2
                )
            )
            return 0
        if args.command == "list":
            from .catalogs import target_names, target_catalog, references, models
            if args.kind == "targets":
                configured = load_config(args.config)["targets"] if args.config else {}
                print(json.dumps({"catalogs": target_names(), "configured": list(configured)}, indent=2))
                return 0
            if args.kind == "baselines":
                print(json.dumps(references(), indent=2))
                return 0
            if args.kind == "models":
                entries = models()
                if args.target:
                    entries = {key: value for key, value in entries.items() if value["target"] == args.target}
                configured = {}
                if args.config:
                    config = load_config(args.config)
                    for name in ([args.target] if args.target else config["targets"]):
                        settings = select_target(config, name).settings
                        configured[name] = {model: {policy: {
                            "inputs": record["inputs"], "baseline": record.get("baseline"), "package_recipe": record.get("package_recipe"),
                            "model_path": record["llama"]["models"].get(model),
                            "implementations": {key: {k: v[k] for k in ("binary", "package") if k in v}
                                                for key, v in record["llama"]["implementations"].items()}}
                            for policy, record in choices.items()}
                            for model, choices in settings.get("model_profiles", {}).items()}
                print(json.dumps({"protocols": entries, "configured": configured}, indent=2))
                return 0
            if args.kind == "kernels":
                targets = [args.target] if args.target else target_names()
                for name in targets:
                    for case, record in target_catalog(name)["cases"].items():
                        candidate = record["candidate"]
                        engines = record["artifact_evidence"]["used_extensions"]
                        print(f"{name}\t{case}\t{candidate['source']}:{candidate['symbol']}\t{','.join(engines) or 'no matrix extension recorded'}")
                return 0
            if not args.target and (args.config or args.kind == "suites"):
                raise ValueError("listing configured suites requires --target")
            settings = select_target(load_config(args.config), args.target).settings if args.target else {}
            suites = configured_suites(settings)
            if args.kind == "suites":
                for name, cases in suites.items():
                    print(f"{name}: {len(cases)} workloads: {', '.join(case.describe() for case in cases)}")
                return 0
            if args.kind == "inputs":
                print(
                    "operators: default, decode, prefill, llama7b, or a user JSON/Inputs object"
                )
                print(
                    "models: default (PP128/TG32/depth512), smoke (PP16/TG4/depth32), or explicit lengths"
                )
                return 0
            if args.kind == "replacements":
                print(
                    "ggml.vec_dot/{q4_0,q4_1,q5_0,q5_1,q8_0,q2_K,q3_K,q4_K,q5_K,q6_K}"
                )
                print("ggml.from_float/{q8_0,q8_K}")
                print("ggml.ime1/{q4_centered_prefill,q4_affine_prefill,q4_centered_decode,q4_affine_decode}")
                print("ggml.ime2/{i8_prefill,i8_prefill_rows8,i8_decode}")
                print("ggml.amx/{q4_0_decode,q4_0_prefill,q4_K_decode,q4_K_prefill,q6_K_decode,q6_K_prefill}")
                return 0
            for case in catalog().values():
                print(f"{case.name:28} {case.scope:42} {case.shape}")
            for name, cases in suites.items():
                print(f"suite {name}: {', '.join(case.describe() for case in cases)}")
            print(
                "models: external config models mapping (llama2-7b-q4_0, deepseek8b-q4_k_m examples)"
            )
            return 0
        if args.command == "compare":
            rows = apply_baselines(
                read(args.after), read(args.before), args.baseline_impl
            )
            print(table(rows))
            if args.output:
                print(save(rows, args.output))
            return 0
        if args.command == "import-baseline":
            from .legacy import import_baseline

            rows = import_baseline(
                args.source, kind=args.kind, target=args.target, profile=args.profile
            )
            print(save(rows, args.output))
            return 0
        config = load_config(args.config)
        target = select_target(config, args.target)
        if args.command == "op" and bool(args.case) == bool(args.suite):
            raise ValueError("select one case or --suite")
        if args.command == "e2e" and not args.case:
            raise ValueError("select one configured model")
        model_profile = None
        if args.command == "e2e":
            target, model_profile = select_model(target, args.case, args.model_profile)
            if model_profile and not args.baseline:
                args.baseline = model_profile.get("baseline")
        elif args.command == "op" and not args.baseline:
            args.baseline = target.settings.get("baselines", {}).get("operators")
        destination = Path(args.output) if args.output else (
            Path(target.settings.get("output_root", "output/current")) / args.target /
            args.command / (args.case or args.suite))
        if args.baseline:
            baseline_path = Path(args.baseline).resolve()
            output_path = destination.resolve()
            if baseline_path == output_path or output_path in baseline_path.parents:
                raise ValueError("candidate output cannot replace its fixed baseline or a containing directory")
        from .output import OutputRun
        output_run = OutputRun(destination)
        output = output_run.stage
        default_impl = (
            "weft"
            if args.baseline
            else "ggml,weft" if args.command == "op" else "normal,weft"
        )
        configured_default = target.settings.get("default_implementations", {}).get(args.command)
        if configured_default:
            default_impl = ",".join(configured_default)
        implementations = [
            x.strip() for x in (args.impl or default_impl).split(",") if x.strip()
        ]
        rows = []
        if args.command == "op":
            from .inputs import resolve_inputs
            from .libraries import WeftKernel, candidate_configuration

            selected = resolve_inputs(args.inputs)
            if args.seed is not None:
                selected = replace(selected, seed=args.seed)
            candidate = WeftKernel.parse(args.weft) if args.weft else None
            if candidate and "weft" not in implementations:
                raise ValueError("--weft requires an implementation selection containing weft")
            if bool(args.case) == bool(args.suite):
                raise ValueError("select one case or --suite")
            shape = (
                dict(
                    (key, int(value))
                    for key, value in (
                        item.split("=", 1) for item in args.shape.split(",")
                    )
                )
                if args.shape
                else None
            )
            if args.suite and shape:
                raise ValueError("--shape is for a single case")
            if shape:
                selected = replace(selected, shape={**selected.shape, **shape})
            measurement = Measurement(
                args.warmup,
                args.samples,
                args.batch_ms,
                args.cache,
                args.eviction_bytes,
                check_only=args.check_only,
                warmup_ms=args.warmup_ms,
            )
            suites = configured_suites(target.settings)
            if args.suite and args.suite not in suites:
                raise ValueError(f"unknown suite {args.suite!r}; use list --kind suites with this config/target")
            selections = suites[args.suite] if args.suite else [OperatorSelection(
                args.case, target.settings.get("case_catalog", {}).get(args.case, {}).get("shape", {}))]
            for selection in selections:
                name = selection.name
                print(f"Running {name} on {target.name}", file=sys.stderr, flush=True)
                case = get(name)
                case_inputs = replace(selected, shape={**selection.shape, **selected.shape})
                case_target = target
                if candidate:
                    case_target = replace(target, implementations={
                        **target.implementations,
                        "weft": candidate_configuration(case, target.implementations.get("weft", {}), candidate),
                    })
                rows.extend(
                    benchmark(
                        case,
                        implementations,
                        case_target,
                        measurement,
                        output=output / "artifacts",
                        inputs=case_inputs,
                        save_outputs=args.save_outputs,
                    )
                )
                save(rows, output)
        else:
            if not args.case:
                raise ValueError("select one configured model")
            from .models.scenarios import ModelInputs
            from .models.api import compare_model
            from .models.replacements import parse_mapping

            selected = (ModelInputs(**model_profile["inputs"]) if model_profile and args.scenario is None
                        else ModelInputs.preset(args.scenario or "default"))
            overrides = {
                k: getattr(args, k)
                for k in ("prefill", "decode", "depth", "threads", "repetitions")
                if getattr(args, k) is not None
            }
            selected = replace(selected, **overrides)
            mapping = (
                json.loads(Path(args.replacement_file).read_text())
                if args.replacement_file
                else {}
            )
            mapping.update(parse_mapping(args.replace))
            if "weft" not in implementations:
                if mapping:
                    raise ValueError(
                        "--replace requires a weft candidate in the model comparison"
                    )
                from .models.llama import benchmark_model

                settings = {
                    **target.settings.get("llama", {}),
                    "phases": selected.phases(),
                }
                if selected.threads is not None:
                    settings["threads"] = selected.threads
                target = replace(
                    target, settings={**target.settings, "llama": settings}
                )
                rows = benchmark_model(
                    args.case,
                    implementations,
                    target,
                    phases=args.phase,
                    repetitions=selected.repetitions,
                    profile=args.profile,
                    output=output,
                    build=args.build,
                    validate=args.validate,
                    validation_reference=args.validation_reference,
                )
            else:
                rows = compare_model(
                    args.case,
                    target=target,
                    replacements=mapping,
                    inputs=selected,
                    phases=args.phase,
                    profile=args.profile,
                    reference=args.baseline,
                    output=output,
                    build=args.build,
                    implementations=implementations if args.impl else None,
                    validate=args.validate,
                    validation_reference=args.validation_reference,
                )
        if args.baseline:
            reference_rows = read(args.baseline)
            for row in rows:
                selected_reference = target.settings.get("case_catalog", {}).get(row.case, {}).get("reference")
                apply_baselines([row], reference_rows,
                    args.baseline_impl or (selected_reference or "ggml" if args.command == "op" else "normal"))
        elif args.baseline_impl or (
            len(implementations) > 1 and ("ggml" in implementations or "normal" in implementations)
        ):
            baseline = args.baseline_impl or (
                "ggml" if args.command == "op" else "normal"
            )
            apply_baselines(
                rows,
                [
                    replace(row, details=dict(row.details))
                    for row in rows
                    if row.implementation == baseline
                ],
                baseline,
            )
        print(table(rows))
        save(rows, output)
        print(output_run.destination / "results.json")
        for row in rows:
            if row.error:
                print(f"{row.case}/{row.implementation}: {row.error}", file=sys.stderr)
            for hot in row.details.get("hotspots", []):
                print("  hotspot:", hot)
        if any(row.status != "ok" for row in rows):
            run_error = "one or more selected cases failed; see results.json"
        return int(run_error is not None)
    except (ValueError, OSError, KeyError, TypeError, ExecutionError) as error:
        run_error = error
        print(f"weft-bench: {error}", file=sys.stderr)
        return 2
    finally:
        if output_run is not None:
            output_run.finish(run_error or sys.exc_info()[1])


if __name__ == "__main__":
    raise SystemExit(main())

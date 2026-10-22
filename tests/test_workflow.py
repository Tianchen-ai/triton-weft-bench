from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weft_bench import Case, Measurement, Result, Target
from weft_bench.check import check, compare
from weft_bench.results import save
from weft_bench.models.llama import replacement_counts
from weft_bench.models.hotspots import link_hotspots
from weft_bench.adapters.dense import bind_metadata


class WorkflowTests(unittest.TestCase):
    def test_catalog_verification_uses_jit_and_selected_shapes(self):
        from weft_bench.catalogs import apply_catalog
        from weft_bench.verification.catalog import plan

        settings, implementations = apply_catalog({"catalog": "x86-1t"}, {})
        target = Target("cpu", settings, implementations)
        selected = plan(target, cases=["operators/q4_k_projection_decode"])
        self.assertEqual(settings["verification"]["source_implementation"], "weft")
        self.assertEqual(selected[0]["execution_mode"], "jit")
        self.assertEqual(selected[0]["shape"], settings["case_catalog"]["q4_k_projection_decode"]["shape"])
        settings["verification"]["operators"] = {"q4_k_projection_decode": {"artifact": "/unused/precompiled"}}
        with self.assertRaisesRegex(ValueError, "JIT verification requires source"):
            plan(target, cases=["operators/q4_k_projection_decode"])

    def test_jit_verification_cannot_prepare_host_artifacts(self):
        from weft_bench.catalogs import apply_catalog
        from weft_bench.verification.prepare import prepare

        settings, implementations = apply_catalog({"catalog": "x86-1t"}, {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-create"
            with self.assertRaisesRegex(ValueError, "--prepare-only cannot verify JIT"):
                prepare(Target("cpu", settings, implementations), output=output,
                        cases=["operators/q4_k_projection_decode"])
            self.assertFalse(output.exists())

    def row(self, implementation="weft", ms=1):
        return (
            Result(
                "q4_k_dot",
                "weft",
                "board",
                {"rows": 1, "K": 4096},
                "vec-dot-leaf",
                "ok",
                ms,
                details={
                    "measurement": asdict(Measurement()),
                    "input_policy": "seed410",
                    "dtype": "q4_k*q8_k",
                    "phase": "",
                    "execution": {"cpus": [0], "threads": 1, "env": {}, "isa": {}},
                },
            )
            if implementation == "weft"
            else replace(self.row(ms=ms), implementation=implementation)
        )

    def test_three_way_comparison_and_scenario_changes(self):
        row = self.row(ms=2)
        compare(row, [self.row("ggml", 3)], [self.row(ms=2.5)])
        self.assertEqual(row.details["comparison"]["ref_speedup"], 1.5)
        self.assertEqual(row.details["comparison"]["previous_speedup"], 1.25)
        changed = replace(
            row, shape={"rows": 4096, "K": 4096}, details=dict(row.details)
        )
        compare(changed, [self.row("ggml", 3)], [self.row(ms=2.5)])
        self.assertEqual(
            changed.details["comparison"]["previous_note"], "scenario changed"
        )
        self.assertTrue(changed.details["comparison"]["previous_differences"])

    def test_fixed_reference_reuse_and_missing_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save([self.row("ggml", 3)], p / "ref")
            save([self.row(ms=2)], p / "old")
            target = Target(
                "board",
                {
                    "check": {
                        "reference": str(p / "ref/results.json"),
                        "previous": str(p / "old"),
                    }
                },
            )
            calls = []

            def run(case, names, *args, **kwargs):
                calls.append(names)
                return [self.row(names[0])]

            with patch("weft_bench.check.benchmark", side_effect=run):
                rows = check(target, output=p / "first", cases=["q4_k_dot"])
            self.assertEqual(calls, [["weft"]])
            self.assertEqual(rows[0].details["reference_action"], "reused")
            save([], p / "ref")
            calls.clear()
            with patch("weft_bench.check.benchmark", side_effect=run):
                check(target, output=p / "second", cases=["q4_k_dot"])
            self.assertEqual(calls, [["weft"], ["ggml"]])
            calls.clear()
            with patch("weft_bench.check.benchmark", side_effect=run):
                check(target, output=p / "third", cases=["q4_k_dot"])
            self.assertEqual(calls, [["weft"]])

    def test_failed_reference_is_visible_without_false_scenario_change(self):
        failed = replace(
            self.row("ggml"),
            status="compile_failed",
            median_ms=None,
            error="native link failed",
        )
        row = compare(self.row(), [failed], [self.row()])
        from weft_bench.check import table

        self.assertIsNone(row.speedup)
        self.assertIn("reference: compile_failed", table([row]))
        self.assertEqual(
            row.details["comparison"]["reference_failure"]["error"],
            "native link failed",
        )

    def test_replay_uses_canonical_and_effective_options_after_directory_copy(self):
        from weft_bench.diagnostics import replay
        from weft_bench.results import read

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work = root / "artifacts/q4_k_dot/weft"
            work.mkdir(parents=True)
            target = Target(
                "board", {}, {"weft": {"selection": "missing-old-selection.json"}}
            )
            from weft_bench.operators import get

            request = {
                "case": asdict(get("q4_k_dot")),
                "target": asdict(target),
                "implementation": "weft",
                "measurement": asdict(Measurement()),
            }
            (work / "replay.json").write_text(json.dumps(request))
            (work / "compile.json").write_text(
                json.dumps(
                    {
                        "root": "/compiler",
                        "compiler": "old",
                        "source": "deleted.py",
                        "symbol": "dot",
                        "options": {},
                    }
                )
            )
            (work / "canonical.mlir").write_text("saved-canonical")
            (work / "effective-options.json").write_text('{"lmul_eighths":32}')
            row = self.row()
            row.details["replay"] = "/unavailable/original/replay.json"
            save([row], root)
            with patch("weft_bench.api.benchmark", return_value=[]) as run:
                replay(read(root)[0], compiler="new", output=root / "replay")
            options = run.call_args.args[2].implementations["weft"]
            self.assertEqual(options["compiler"], "new")
            self.assertEqual(Path(options["canonical"]).read_text(), "saved-canonical")
            self.assertEqual(
                options["bindings"]["q4_k_dot"]["options"]["lmul_eighths"], 32
            )
            self.assertNotIn("selection", options)

    def test_metadata_binding_handles_names_shapes_scalars_and_strides(self):
        metadata = {
            "symbol": "another_name",
            "bindings": {},
            "shape_parameters": ["M", "N"],
            "arguments": [
                {
                    "name": "Input", "kind": "view", "dtype": "f32",
                    "access": "read", "c_type": "const float *",
                    "encoding": "dense.f32", "strided": True, "shape": ["M", "N"],
                },
                {
                    "name": "scale", "kind": "scalar", "dtype": "f32",
                    "access": "none", "c_type": "float",
                    "encoding": "", "strided": False, "shape": [],
                },
                {
                    "name": "Out", "kind": "view", "dtype": "f32",
                    "access": "write", "c_type": "float *",
                    "encoding": "dense.f32", "strided": True, "shape": ["M", "N"],
                },
            ],
        }
        call = bind_metadata(
            metadata,
            {"X": "x", "Y": "y"},
            {"M": 3, "N": 7},
            {"scale": "2.0f"},
            {"Input": "X", "Out": "Y"},
        )
        self.assertEqual(
            call,
            "another_name(reinterpret_cast<const float *>(x), 2.0f, "
            "reinterpret_cast<float *>(y), 3, 7, 7, 1, 7, 1);",
        )

    def test_counter_windows_exclude_warmup(self):
        log = "\n".join(
            [
                "WB_COUNTER sample=0 point=begin type=q4_0 dots=100 quantizes=10 ime=0",
                "WB_COUNTER sample=0 point=end type=q4_0 dots=120 quantizes=12 ime=0",
                "WB_COUNTER sample=1 point=begin type=q4_0 dots=120 quantizes=12 ime=0",
                "WB_COUNTER sample=1 point=end type=q4_0 dots=140 quantizes=14 ime=0",
            ]
        )
        counts = replacement_counts(log)
        self.assertEqual(counts["vec_dot"], 40)
        self.assertEqual(counts["quantize"], 4)
        self.assertEqual(len(counts["samples"]), 2)

    def test_normal_model_reference_capture_reuses_files_without_self_comparison(self):
        import struct
        from weft_bench.models.validation import capture_model_reference, attach_validation

        target = Target("cpu", {"llama": {"models": {"model": "/model.gguf"}, "threads": 1,
                        "implementations": {"normal": {"binary": "/llama-bench"}}}})

        def capture(name, selected, implementation, parameters, directory):
            self.assertEqual(implementation, "normal")
            (directory / "logits.bin").write_bytes(struct.pack("<II6f", 3, 2, 1, 2, 3, 4, 5, 6))
            (directory / "tokens.bin").write_bytes(struct.pack("<II18i", 16, 18, *([0] * 18)))
            return {"rows": 3, "vocabulary": 2}

        with tempfile.TemporaryDirectory() as directory, patch(
            "weft_bench.models.validation._run", side_effect=capture
        ) as run:
            first = capture_model_reference("model", target, output=directory)
            second = capture_model_reference("model", target, output=directory)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(first["correctness"], "reference-logits-captured")
            self.assertFalse(first["self_comparison"])
            self.assertTrue(second["reference_reused"])
            row = Result("model/prefill", "normal", "cpu", {}, "llama-prefill", "ok", 12.5, [12.5])
            attach_validation([row], second, "normal")
            self.assertEqual((row.median_ms, row.samples_ms), (12.5, [12.5]))
            self.assertEqual(row.correctness, "reference-logits-captured")
            (Path(directory) / "logits.bin").write_bytes(struct.pack("<II6f", 3, 2, float("nan"), 2, 3, 4, 5, 6))
            invalid = capture_model_reference("model", target, output=directory)
            self.assertEqual(invalid["status"], "validation_failed")
            self.assertIn("nonfinite", invalid["error"])
            self.assertEqual(run.call_count, 1)

    def test_instrumented_native_windows_exclude_actual_warmup(self):
        import subprocess
        from weft_bench.models.observe import instrument

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "include/weft/ggml.h"
            header.parent.mkdir(parents=True)
            header.write_text("""#pragma once
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_COUNT };
inline unsigned long long calls = 100;
struct ggml_weft_counters { unsigned long long vec_dot_calls, weft_quantize_calls; };
inline bool ggml_weft_counters_get(ggml_type, ggml_weft_counters *c) { *c={calls,0}; return true; }
inline const char *ggml_type_name(ggml_type) { return "f32"; }
inline unsigned long long ggml_weft_ime1_calls() { return 0; }
""")
            source = root / "tools/llama-bench/llama-bench.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("""#include <vector>
#include <cstdint>
#include <chrono>
#include "weft/ggml.h"
uint64_t get_time_ns() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
void run() { calls += 3; }
int main() {
    struct { int n_prompt=128,n_gen=0; std::vector<uint64_t> samples_ns; } t;
    run(); // warmup must not enter the measured counters
    for(int i=0;i<3;++i) {
            uint64_t t_start = get_time_ns();
            run();
            uint64_t t_ns = get_time_ns() - t_start;
            t.samples_ns.push_back(t_ns);
    }
}
""")
            instrument(root, weft_include=header.parent.parent)
            self.assertFalse(
                instrument(root, weft_include=header.parent.parent)["changed"]
            )
            executable = root / "observer"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-I" + str(header.parent.parent),
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            completed = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True
            )
            counts = replacement_counts(completed.stderr)
            self.assertEqual(counts["vec_dot"], 9)
            self.assertEqual(len(counts["samples"]), 3)

    def test_model_mapping_has_source_and_unknowns(self):
        rows = link_hotspots(
            [
                {"symbol": "ggml_vec_dot_q4_K_q8_K_vl128", "percent": 75},
                {"symbol": "unknown_leaf", "percent": 5},
            ],
            target="board",
            phase="decode",
            model_info={"tensor_shapes": [{"ggml_type": 12, "shape": [4096, 11008]}]},
        )
        self.assertEqual(rows[0]["shape"], {"K": 4096, "rows": 11008})
        self.assertEqual(rows[0]["shape_origin"], "model-tensor-metadata")
        self.assertEqual(rows[0]["mapping"], "model_tensor_case")
        self.assertEqual(rows[1]["mapping"], "unmapped")

    def test_history_retains_both_sides_and_unmapped(self):
        from weft_bench.legacy import import_history

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.json"
            original = {
                "family": "vec_dot",
                "kernel": "quantized_vec_dot_q4_k_q8_k",
                "hardware": "K1/X60",
                "shape": "M=1;N=4096;K=4096",
                "compiler_flags": "-march=rv64gcv",
                "isa": "VLEN256",
                "comparison_median_ms": "13.379481",
                "median_ms": "8.426079",
                "configuration": json.dumps(
                    {
                        "input_policy": "independent-quantized-rows-seed410",
                        "native_cpu": [3],
                    }
                ),
            }
            p.write_text(json.dumps([original, {"kernel": "unmapped"}]))
            rows = import_history(p)
            self.assertEqual(
                [(r.implementation, r.status) for r in rows],
                [("ggml", "ok"), ("weft", "ok"), ("historical", "unmapped")],
            )
            self.assertAlmostEqual(
                rows[0].median_ms / rows[1].median_ms, 1.587865, places=5
            )


if __name__ == "__main__":
    unittest.main()

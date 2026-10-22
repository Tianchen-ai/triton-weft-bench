from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from weft_bench import Case, Measurement, Target, Result, benchmark
from weft_bench.loading import load_config, select_target
from weft_bench.models.llama import replacement_counts, summarize_samples
from weft_bench.results import apply_baselines, read, save

ROOT = Path(__file__).resolve().parents[1]


class ToolTests(unittest.TestCase):
    def target(self):
        return Target(
            "host",
            {"toolchain": {"cc": ["cc"], "cxx": ["c++"]}},
            {"custom": {"factory": str(ROOT / "examples/custom.py") + ":Adapter"}},
        )

    def test_external_adapter_native_timer_and_fixed_baseline(self):
        from weft_bench.loading import load_object

        case = load_object(str(ROOT / "examples/custom.py") + ":case")
        with tempfile.TemporaryDirectory() as directory:
            rows = benchmark(
                case,
                ["custom"],
                self.target(),
                Measurement(samples=3, batch_ms=1),
                output=directory,
            )
            self.assertEqual(rows[0].status, "ok", rows[0].error)
            self.assertEqual(len(rows[0].samples_ms), 3)
            self.assertEqual(rows[0].correctness, "exact")
            path = save(rows, Path(directory) / "results")
            loaded = read(path)
            after = replace(
                loaded[0],
                median_ms=loaded[0].median_ms / 2,
                details=dict(loaded[0].details),
            )
            apply_baselines([after], loaded)
            self.assertEqual(after.speedup, 2)
            self.assertTrue(path.with_suffix(".csv").is_file())

    def test_scope_shape_and_profile_prevent_wrong_ratio(self):
        baseline = Result(
            "dot",
            "ggml",
            "board",
            {"K": 4096},
            "leaf",
            "ok",
            2,
            details={"measurement": {"cache": "reuse"}},
        )
        for changed in (
            {"scope": "projection"},
            {"shape": {"K": 8192}},
            {"details": {"measurement": {"cache": "cold"}}},
            {"details": {"measurement": {"cache": "reuse", "profile": True}}},
        ):
            candidate = replace(
                baseline,
                implementation="weft",
                speedup=1.5,
                baseline="older",
                **changed,
            )
            apply_baselines([candidate], [baseline])
            self.assertIsNone(candidate.speedup)

    def test_failures_are_results_and_not_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "adapter.py"
            script.write_text("""from weft_bench import PreparedRun
class Adapter:
 def __init__(self, options): self.options=options
 def prepare(self, case, target, executor):
  return PreparedRun(["sh","-c",self.options["command"]],executor.workspace)
""")
            target = replace(
                self.target(),
                implementations={
                    "numeric": {
                        "factory": str(script) + ":Adapter",
                        "command": "echo 'numeric mismatch' >&2; exit 1",
                    },
                    "missing": {
                        "factory": str(script) + ":Adapter",
                        "command": "exit 2",
                    },
                },
            )
            rows = benchmark(
                Case("test", "custom", {}, "call"),
                ["numeric", "missing", "unknown"],
                target,
                output=Path(directory) / "run",
            )
            self.assertEqual(
                [r.status for r in rows],
                ["check_failed", "run_failed", "prepare_failed"],
            )
            self.assertTrue(all(r.median_ms is None for r in rows))

    def test_compiler_failure_is_distinct(self):
        target = replace(
            self.target(), settings={"toolchain": {"cc": ["cc"], "cxx": ["false"]}}
        )
        case = Case("custom_affine", "custom", {"N": 8}, "native-affine")
        with tempfile.TemporaryDirectory() as directory:
            row = benchmark(case, ["custom"], target, output=directory)[0]
        self.assertEqual(row.status, "compile_failed")

    def test_missing_configuration_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "unknown target"):
            select_target({"targets": {}}, "missing")

    def test_external_bindings_are_explicit_and_overridable(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "bindings.json").write_text(
                json.dumps(
                    {"bindings": {"a": {"symbol": "old"}, "b": {"symbol": "keep"}}}
                )
            )
            (p / "config.json").write_text(
                json.dumps(
                    {
                        "targets": {
                            "board": {
                                "bindings_file": "bindings.json",
                                "bindings": {"a": {"symbol": "new"}},
                            }
                        }
                    }
                )
            )
            target = select_target(load_config(p / "config.json"), "board")
            self.assertEqual(target.settings["bindings"]["a"]["symbol"], "new")
            self.assertEqual(target.settings["bindings"]["b"]["symbol"], "keep")

    def test_profile_excludes_loading_and_counts_cumulative_once(self):
        text = "1.0: 900 ff loading (lib)\n2.0: 50 ab dot (lib)\n2.5: 30 ac quant (lib)\n3.0: 20 ad dot (lib)\n"
        rows = summarize_samples(text, [(2.0, 3.0)])
        self.assertEqual(rows[0]["symbol"], "dot")
        self.assertEqual(rows[0]["percent"], 70)
        counts = replacement_counts(
            "WEFT_CALLS type=q4_0 dots=2 quantizes=0\nWEFT_CALLS type=q4_0 dots=5 quantizes=1\nWEFT_IME_CALLS 3\nWEFT_IME_CALLS 8\n"
        )
        self.assertEqual(counts["vec_dot"], 3)
        self.assertEqual(counts["ime"], 5)

    def test_stateful_reset_excluded_from_native_clock(self):
        from weft_bench.adapters.native import NATIVE

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.cpp"
            source.write_text("""#include "timer.h"
#include <thread>
int main() { int x=0; auto run=[&]{ if(x!=0) std::abort(); x=1; };
auto reset=[&]{ std::this_thread::sleep_for(std::chrono::milliseconds(2)); x=0; };
wb::measure(run,reset,"exact",true); }
""")
            binary = Path(directory) / "test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-I" + str(NATIVE),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            import os

            result = subprocess.run(
                [str(binary)],
                env={
                    **os.environ,
                    "WB_SAMPLES": "2",
                    "WB_CACHE": "cold",
                    "WB_WARMUP": "1",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(result.stdout.removeprefix("WB_RESULT "))
            self.assertTrue(all(v < 1 for v in row["samples_ms"]))

    def test_observer_is_idempotent_and_outside_timer(self):
        from weft_bench.models.observe import instrument

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tools/llama-bench/llama-bench.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                "            uint64_t t_start = get_time_ns();\nrun();\n            uint64_t t_ns = get_time_ns() - t_start;\n            t.samples_ns.push_back(t_ns);\n"
            )
            self.assertTrue(instrument(directory)["changed"])
            self.assertFalse(instrument(directory)["changed"])
            text = source.read_text()
            self.assertLess(text.index("clock_gettime"), text.index("t_start ="))
            self.assertGreater(text.index("PROFILE_WINDOW"), text.index("t_ns ="))


if __name__ == "__main__":
    unittest.main()

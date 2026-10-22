"""Publication-critical catalog, retained evidence and output replacement contracts."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weft_bench.catalogs import target_catalog, target_names, resolve_source
from weft_bench.cli import main
from weft_bench.output import OutputRun
from weft_bench.results import contracts_match
from weft_bench.types import Result


class ReleaseStructureTests(unittest.TestCase):
    def test_source_override_does_not_inherit_another_author_program(self):
        from weft_bench.adapters.triton_native import TritonAdapter
        from weft_bench.libraries import WeftKernel
        from weft_bench.operators import get
        base = {"python": "/environment/python", "toolchain": {}, "bindings": {
            "q8_0_dot": {"source": "/old.py", "symbol": "old", "constexprs": {"GROUP": 32},
                         "workspaces": [{"name": "scratch"}], "signature": {"W": "*i8"}}}}
        configured = TritonAdapter.configure_candidate(get("q8_0_dot"), base, WeftKernel(source="/new.py", symbol="new"))
        self.assertEqual(configured["constexprs"], {})
        self.assertEqual(configured["signature"], {})
        self.assertNotIn("workspaces", configured)
        self.assertEqual(configured["python"], base["python"])

    def test_catalog_sources_and_explicit_suite_membership(self):
        expected = {"sg2044": 133, "k1": 137, "k3-a100": 135, "x86-1t": 146}
        self.assertEqual(set(target_names()), set(expected))
        for target, count in expected.items():
            catalog = target_catalog(target)
            self.assertEqual(len(catalog["cases"]), count)
            selected = {name for names in catalog["suites"].values() for name in names}
            self.assertEqual(selected, set(catalog["cases"]))
            for name, record in catalog["cases"].items():
                with self.subTest(target=target, case=name):
                    self.assertTrue(Path(resolve_source(record["candidate"]["source"])).is_file())
                    self.assertNotIn("/home/", json.dumps(record["candidate"]))

    def test_update_retains_failure_and_locator_without_rewriting_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current"
            path.mkdir()
            (path / "old").write_text("obsolete")
            run = OutputRun(path)
            original = str(run.stage / "artifact")
            (run.stage / "results.json").write_text(json.dumps({"results": [{"details": {
                "artifacts": {"weft": original}, "command": ["run", original]}}]}))
            with self.assertRaisesRegex(ValueError, "another run"):
                OutputRun(path)
            run.finish(KeyboardInterrupt())
            self.assertFalse((path / "old").exists())
            self.assertEqual(json.loads((path / "run-state.json").read_text())["status"], "failed")
            detail = json.loads((path / "results.json").read_text())["results"][0]["details"]
            self.assertEqual(detail["artifacts"]["weft"], str(path / "artifact"))
            self.assertEqual(detail["command"], ["run", original])
            retry = OutputRun(path)
            retry.finish()
            self.assertEqual(json.loads((path / "run-state.json").read_text())["status"], "completed")

    def test_unexpected_error_never_publishes_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"targets": {"host": {}}}))
            with patch("weft_bench.cli.benchmark", side_effect=RuntimeError("interrupted compiler")):
                with self.assertRaisesRegex(RuntimeError, "interrupted compiler"), redirect_stderr(io.StringIO()):
                    main(["op", "q4_0_dot", "--impl", "weft", "--target", "host", "--config", str(config),
                          "--output", str(root / "current")])
            state = json.loads((root / "current/run-state.json").read_text())
            self.assertEqual(state["status"], "failed")

    def test_relocated_launcher_alias_preserves_identity_only_for_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "numactl"
            binary.touch()
            alias = root / "old-numactl"
            alias.symlink_to(binary)
            row = Result("case", "weft", "cpu", {}, "native", "ok", 1,
                         details={"execution": {"cpus": [1], "numa": {"command": str(binary)}}})
            reference = replace(row, details={"execution": {"cpus": [1], "numa": {"command": str(alias)}}})
            self.assertTrue(contracts_match(row, reference))
            self.assertEqual(reference.details["execution"]["numa"]["command"], str(alias))
            reference.details["execution"]["cpus"] = [2]
            self.assertFalse(contracts_match(row, reference))


if __name__ == "__main__":
    unittest.main()

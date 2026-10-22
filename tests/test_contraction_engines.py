"""Focused native compiler checks; run with the selected Triton installation."""
import json
import os
from pathlib import Path
import unittest

import numpy as np
import triton
from triton import cpu
from weft_bench.verification.probes import contraction_engine_kernels as kernels


class ContractionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["WEFT_BENCH_ENGINE_OUTPUT"])
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        (cls.root / "checks.json").write_text(json.dumps(cls.records, indent=2))

    def run_kernel(self, fn, engine, shape=(16, 16, 64)):
        m, n, k = shape
        a = ((np.arange(m*k) % 251) - 125).astype(np.int8).reshape(m, k)
        b = ((np.arange(k*n) % 253) - 126).astype(np.int8).reshape(k, n)
        storage = np.full(m*n+2, -9999, dtype=np.int32)
        c = storage[1:-1].reshape(m, n)
        parameters = {} if fn is kernels.mixed else {"M": m, "N": n, "K": k, "ENGINE": engine}
        compiled = fn[(1,)](a, b.T.copy() if fn is kernels.named else b, c,
                            num_cpu_threads=1, **parameters)
        expected = a.astype(np.int32) @ b.astype(np.int32)
        if fn is kernels.mixed:
            expected *= 2
        np.testing.assert_array_equal(c, expected)
        np.testing.assert_array_equal(storage[[0, -1]], [-9999, -9999])
        self.assertIs(compiled, fn[(1,)](a, b.T.copy() if fn is kernels.named else b, c,
                                        num_cpu_threads=1, **parameters))
        extensions = list(compiled.metadata.kernel_abi["used_extensions"])
        if engine == "vector":
            self.assertFalse(any("amx" in name or "ime" in name for name in extensions))
        else:
            self.assertTrue(any("amx" in name or "ime" in name for name in extensions))
        self.records.append({"kernel": fn.__name__, "engine": engine, "shape": shape,
                             "correctness": "exact", "used_extensions": extensions,
                             "warm_kernel_reused": True, "guards_unchanged": True})

    def test_explicit_and_default_routes(self):
        for engine in ("auto", "vector", "matrix"):
            with self.subTest(engine=engine):
                self.run_kernel(kernels.positional, engine)

    def test_masked_tail(self):
        for engine in ("vector", "matrix"):
            with self.subTest(engine=engine):
                self.run_kernel(kernels.positional, engine, (13, 11, 59))

    def test_named_axes(self):
        for engine in ("vector", "matrix"):
            with self.subTest(engine=engine):
                self.run_kernel(kernels.named, engine)

    def test_mixed_routes(self):
        self.run_kernel(kernels.mixed, "mixed")

    def test_invalid_engine_and_unavailable_matrix(self):
        from triton.compiler import ASTSource
        signature = {"A": "*fp32", "B": "*fp32", "C": "*fp32"}
        constants = {"M": 16, "N": 16, "K": 64}
        for engine, diagnostic in (("unknown", "engine"), ("matrix", "no legal x86 matrix")):
            with self.subTest(engine=engine), self.assertRaisesRegex(Exception, diagnostic):
                triton.compile(ASTSource(kernels.positional, signature, {**constants, "ENGINE": engine}),
                               target=cpu.target("x86-avx2"))


if __name__ == "__main__":
    unittest.main()

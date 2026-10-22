# Weft Bench

Compare Triton–Weft or Weft kernels with GGML, or compare normal llama.cpp with
explicitly selected kernel replacements. Pick an operation, inputs and implementation;
Bench prepares the shared inputs, checks outputs and measures the native call.

Reference libraries can also run independently: GGML operators and normal llama.cpp,
OpenBLAS F32 GEMM, and oneDNN strict F32, BF16/F32 or S8/S32 matmul. See the
[x86 reference workflow](docs/x86-references.md) to establish fixed baselines before
a compiler candidate is available.

```sh
git clone https://github.com/Tianchen-ai/Weft-Bench.git
cd Weft-Bench
python3 -m venv .venv
.venv/bin/python -m pip install -e .
export PATH="$PWD/.venv/bin:$PATH"
export WEFT_BENCH_CONFIG=/path/to/bench.local.json

# Preset kernel and input, compared with the configured GGML library.
weft-bench op q4_k_dot --target rvv128

# Your kernel, existing workload and reference. No benchmark driver to write.
weft-bench op rms_norm --weft /work/norm.py:my_norm --inputs decode --target rvv128

# Select exactly which llama.cpp entry to replace.
weft-bench e2e llama2-7b-q4_0 --target rvv128 \
  --replace ggml.vec_dot/q4_0=/work/dot.py:my_dot --phase both
```

The supported scope is intentionally concrete: GGML operator references and preset
llama.cpp workloads, with Triton–Weft or Weft source/artifact candidates. Runtime/library paths,
compiler targets and local/SSH deployment are external configuration.
The examples assume that configuration already exists. See [installation](docs/install.md)
to choose the compiler adapter, native SDK and target environment. Target names are
local aliases; installing Bench does not install a compiler, GGML or model weights.

| Select independently | Meaning |
|---|---|
| Operation | Computation, input/output contract and numerical oracle |
| Inputs | Preset dimensions, seed, custom arrays/files and scalar parameters |
| Weft kernel | Source + symbol, compiler options, or an existing artifact |
| GGML library | Native reference build and its headers |
| Model replacements | Explicit GGML role → Weft kernel bindings |
| Execution environment | Toolchains, ISA/VLEN, local or SSH worker, affinity |

Both operator implementations consume the same prepared input instance and have
separate output buffers in one native worker. Implementations supply calls; the
worker supplies checking, warmup and timing. Setup, compilation and transport are
outside the timed region.

```python
from weft_bench import compare, Inputs, WeftKernel
from weft_bench.loading import load_config, select_target

target = select_target(load_config("/path/to/bench.local.json"), "rvv128")
rows = compare(
    "rms_norm",
    candidate=WeftKernel(source="/work/norm.py", symbol="my_norm"),
    inputs=Inputs(shape={"M": 1, "N": 4096}, seed=410),
    target=target,
)
```

Terminal tables, CSV and JSON are the normal outputs. Optional output tensors,
compiler diagnostics and captured IR support inspection and replay. History comparison
and fixed result reuse are outer conveniences; no previous result is required to bench.
For compiler or installed-runtime changes, `verify` runs only the selected configured
correctness cases and saves diagnostics. It reports no latency or speedup; use `op`
and `e2e` for performance. Neither workflow requires a per-run benchmark script.

| Guide | Contents |
|---|---|
| [Installation](docs/install.md) | Bench package, compiler/runtime dependencies and first checks |
| [Quick start](docs/quickstart.md) | Choose workloads, kernels, inputs and replacements |
| [Input and implementation API](docs/extending.md) | IO contracts, custom data, Python usage |
| [Environment setup](docs/targets.md) | GGML, Weft, llama.cpp, RVV/IME and SSH |
| [x86 references](docs/x86-references.md) | GGML/OpenBLAS/oneDNN, NUMA, actual dispatch and reference-only models |
| [Triton–Weft integration](docs/triton-weft-integration.md) | Source/AOT, native grid, Python JIT and model packages |
| [Ownership and migration](docs/migration.md) | Core versus orchestration; older saved results |
| [Codex plugin](docs/codex-plugin.md) | Install the agent workflow and connect local configuration |
| [Maturity verification](docs/verification.md) | Enumerate configured source/AOT, Program, runtime and framework checks |
| [中文入口](README.zh-CN.md) | 产品与日常使用 |

Requires Python 3.10+ and the selected target's native C/C++ toolchain. GGML-only
measurement does not import a compiler. NumPy is optional for NumPy input files/arrays;
target-side Python runtime/framework verification requires its configured NumPy/Torch.
Models, machine configuration and generated runs remain outside version control.
Licensed under Apache-2.0.

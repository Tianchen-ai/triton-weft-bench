# Triton–Weft Bench 1.0

[中文](README.zh-CN.md) · [Download 1.0](https://github.com/Tianchen-ai/triton-weft-bench/releases/tag/v1.0.0) · [Bench website](https://tianchen-ai.github.io/triton-weft-bench/) · [All paper figures](https://tianchen-ai.github.io/triton-weft-bench/paper/gallery/) · [Result tables](docs/paper/README.md) · [Triton–Weft](https://github.com/Tianchen-ai/triton-weft)

**From a Weft program to a matched CPU experiment.**

Weft gives authors control over representation, arithmetic, sharing and state.
Bench connects those programs to concrete workloads: the same inputs, optimized
references, numerical checks and a common native timer. Select an installed
kernel or your own implementation, then compare a complete operator or its
effect inside llama.cpp.

The paper's results tell the same story at three scales: performance across
operators and applications, the effect of composable author choices, and the
compiler mechanisms that realize them. **[The complete result gallery](docs/paper/gallery.md)**
includes all 12 result figures (23 vector panels), matching data links and
timing explanations. The [reproduction workflow](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)
takes you from product installation to native and application measurements.

## Paper results

The paper evaluates **553 operator–platform cases**, including **543 matched
native comparisons**, and **18 model stages**. These figures and tables use the
[published paper dataset](docs/paper/README.md); speedup is reference time divided
by Triton–Weft time.

### Operators

[![Representative kernels and all 543 matched native comparisons](docs/paper/operators-ecdf.svg)](docs/paper/operators-ecdf.svg)

| Platform | Registered cases | Matched comparisons | Geometric mean speedup |
|---|---:|---:|---:|
| SG2044 · RVV128 | 133 | 132 | **1.60×** |
| K1 · RVV256 / IME1 | 137 | 136 | **1.54×** |
| K3 A100 · RVV1024 / IME2 | 136 | 135 | **1.83×** |
| Xeon Platinum 8470Q · AVX-512 / VNNI / AMX | 147 | 140 | **1.32×** |

References comprise 541 GGML cases and two oneDNN cases.
[Operator CSV](docs/paper/operators.csv) records workloads, numerical checks,
timings, references and JIT observations.

### Model inference

[![Prefill and decode for all nine paired model configurations](docs/paper/models.svg)](docs/paper/models.svg)

| Platform | Model | Reference | Prefill | Decode |
|---|---|---|---:|---:|
| x86 | Llama 2 7B Q4_0 | llama.cpp | 1.705× | 1.217× |
| x86 | DeepSeek 8B Q4_K_M | llama.cpp | 1.186× | 1.194× |
| x86 | Qwen 2.5 14B Q4_K_M | llama.cpp | 1.154× | 1.179× |
| SG2044 | Llama 2 7B Q4_0 | llama.cpp | 1.260× | 1.011× |
| SG2044 | DeepSeek 8B Q4_K_M | llama.cpp | 1.167× | 1.016× |
| K1 | Llama 2 7B Q4_0 | llama.cpp | 1.432× | 1.026× |
| K1 | DeepSeek 8B Q4_K_M | llama.cpp | 1.188× | 1.168× |
| K3 A100 | DeepSeek 8B Q4_K_M | vendor IME2 | 1.157× | 1.361× |
| K3 A100 | DeepSeek 8B Q4_K_M | CPU-vector | 1.939× | 1.976× |

SG2044 model runs use 64 threads; K1, K3 and x86 use eight. SG2044/K1/x86 use 128-token prefill and
32-token decode after a 512-token context; K3 uses 32-token prefill and
8-token decode after a 32-token context. [Model CSV](docs/paper/applications.csv)
contains all 18 measured stages, reference policies and replacement counters.

### What produces the performance?

| Evidence | What to inspect |
|---|---|
| [Vector/matrix execution, FP32 libraries and thread scaling](docs/paper/gallery.md#vector-and-matrix-execution) | Hardware use, complete-call cost and multi-core execution |
| [Five-route author controls](docs/paper/gallery.md#five-routes-for-the-same-control-tasks) | Q4_K/Q6_K preparation and precision choices in C, Triton CPU, Exo 2, TensorIR and Weft |
| [Arithmetic and state scope](docs/paper/gallery.md#arithmetic-organization-and-output-state) | Source organization under the same compiler and physical binding |
| [Source increments and reuse](docs/paper/gallery.md#source-increments-and-reuse) | New formats, a second operator, attention and streaming statistics |
| [Compiler mechanisms](docs/paper/gallery.md#compiler-mechanisms-under-fixed-source-semantics) | Encoding, forwarding, fragment retention and source ranges with fixed source semantics |

[![Source added and reused across two operators](docs/paper/increment-measurements.svg)](docs/paper/increment-measurements.svg)

The [data guide](docs/paper/README.md) links every result table and the exact
figure provenance. The [Triton–Weft handbook](https://tianchen-ai.github.io/triton-weft/)
explains the model and compiler behind these experiments.

Version **1.0.0**. First install the product using the [Triton–Weft installation guide](https://tianchen-ai.github.io/triton-weft/getting-started/installation/). Normal kernels come from `triton.kernels.weft`.

## Install and reproduce

Reference libraries can also run independently: GGML operators and normal llama.cpp,
OpenBLAS F32 GEMM, and oneDNN strict F32, BF16/F32 or S8/S32 matmul. See the
[x86 reference workflow](docs/x86-references.md) to establish fixed baselines before
a compiler candidate is available.

```sh
git clone https://github.com/Tianchen-ai/triton-weft-bench.git
cd triton-weft-bench
python -m pip install '.[verification]'
export WEFT_BENCH_CONFIG=/path/to/bench.local.json

# See the selected author program, reference and artifact evidence.
weft-bench list --kind targets
weft-bench show q4_k_dot --target k1
weft-bench list --kind baselines
weft-bench list --kind models --target k1 --config "$WEFT_BENCH_CONFIG"

# Reuse the configured fixed reference; only the candidate is measured.
weft-bench op q4_k_dot --target k1

# Your kernel, existing workload and reference. No benchmark driver to write.
weft-bench op rms_norm --weft /work/norm.py:my_norm --inputs decode --target k1

# Select exactly which llama.cpp entry to replace.
weft-bench e2e llama2-7b-q4_0 --target k1 \
  --replace ggml.vec_dot/q4_0=/work/dot.py:my_dot --phase both
```

Bench provides GGML operator references and preset llama.cpp workloads, with
Triton–Weft source/artifact candidates. Historical Weft results
remain readable. Runtime/library paths,
compiler targets and local/SSH deployment are external configuration.
The examples assume that configuration already exists. See [installation](docs/install.md)
to choose the compiler adapter, native SDK and target environment. Target names are
local aliases; installing Bench does not install a compiler, GGML or model weights.

`main` is the product branch. Normal kernels are installed by Triton–Weft as
`triton.kernels.weft`; Bench's compatibility modules forward to that library. Public catalogs contain the selected DSL, shape,
physical parameters, reference provider and recorded hardware evidence:

DSL code uses the standard `tl` alias. GGML base Encodings come from the product's
`triton.integrations.ggml_formats`; explicit candidate layouts are collected in
`triton.kernels.weft.ggml.layouts`. The [source map](docs/catalogs.md#author-programs-and-encoding-declarations)
distinguishes declarations, numerical programs, preparation and reference adapters.

| Target catalog | Registered cases |
|---|---:|
| SG2044 / RVV128 | 133 |
| K1 / RVV256 + IME | 137 |
| K3 A100 / RVV1024 + IME2 | 136 |
| x86 single-thread | 147 |

The [catalog guide](docs/catalogs.md) explains these selections and the 9 model
package recipes. A GEMM name does not imply matrix hardware: author constraints,
instructions in an artifact and an observed conditional execution path are separate.

Results default to stable target/case locations under `output/`. Fixed baselines,
current results and reports are separate; `.local/` contains private dependencies
and environments. Neither directory is published. A failed update becomes the
current result and cannot masquerade as the preceding successful run.

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

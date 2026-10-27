# Programs, references and retained evidence

The Python package includes `catalogs/targets/*.json`, `references.json`,
`models.json` and `packages.json`. These are explicit selections, not searches
for the fastest observation in an output directory.

```sh
weft-bench list --kind kernels --target x86-1t
weft-bench list --kind suites --target k3-a100 --config bench.local.json
weft-bench show ime2_i8_prefill --target k3-a100 --config bench.local.json
weft-bench list --kind baselines
weft-bench list --kind models --target k1 --config bench.local.json
```

Each case specifies computation, shape, dtype, timing scope, source file and
symbol, constexpr choices, backend options and reference provider. The selected
artifact records actual extension requirements. Conditional paths without a
counter remain `not_observed`; a source filename, a GEMM workload or an AMX
buffer name is not instruction-execution evidence.

The four operator catalogs contain SG2044 133, K1 137, K3 A100 136 (including two IME2),
and x86 147 entries. Unmatched reference cases remain unmatched. `full` suites
explicitly list their members; `show` and a single-case `op` use the same selected
shape. User `--shape` overrides it explicitly.

## Author programs and Encoding declarations

Every kernel uses `import triton.language as tl`. A GGML workload's numerical
program is independent of its reference adapter: `ggml_formats.Q4_K` denotes a
storage Encoding, while the `ggml` implementation in a catalog is a native
reference library.

| Source location | Responsibility |
|---|---|
| Product `triton.integrations.ggml_formats` | Shared GGML base records, fields and byte layout contracts |
| `weft_bench.kernels.ggml.layouts` | Author-owned IME1/IME2, AMX and row-interleaved Encodings and derive builders |
| `weft_bench.kernels.ggml.blocks` | Encoded block arithmetic and composed full projections |
| `weft_bench.kernels.ggml.vec_dot`, `mul_mat`, `quantize` | Explicit candidate algorithms and target-oriented source variants |
| `weft_bench.kernels.programs` | Normalization, attention, MoE, Mamba and other ordered model programs |
| `weft_bench.libraries` and `adapters` | Native reference and candidate invocation contracts |

Each layout consumer imports its declaration from `kernels.ggml.layouts`.
The layout package does not re-export computation kernels. Importing an Encoding
does not pack inputs; a catalog's persistent preparation or explicit kernel
produces the storage before it is used. The small `dense/low_bit_contraction.py`
example retains its two self-contained declarations beside its author program.

## Local configuration

Select a public catalog using `targets.<name>.catalog`. Configure toolchains,
CPU sets, native runtime, SSH and model locations outside Git. See
`configs/example.json`. Compiler/library inputs remain explicit; no
machine is contacted by `list` or `show`.

`implementations.weft` supplies the installed Triton Python and target toolchain.
The catalog supplies per-program settings. `--weft other.py:symbol` replaces the
author program and does not inherit the old program's constexprs or workspace.
Use an explicit local binding when the new program needs those choices.

## Fixed references

The reference catalog documents source families and available revision evidence:
GGML CPU/RVV and x86, vendor IME1/IME2, OpenBLAS, oneDNN, and normal llama.cpp.
The local configuration points to retained observations. Moving a directory
does not change their original command, toolchain or measurements.

A result directory may contain `index.json` instead of copied measurements:

```json
{"sources": [
  {"path": "../../current/paper/x86-1t/reference"},
  {"path": "../library-reference/results.json", "implementation": "onednn",
   "target": "x86-1t", "cases": ["gemm_bf16_f32", "gemm_s8_s32"]}
]}
```

Paths resolve relative to the index. Matching still requires the same workload,
inputs, execution and timing contract. Multiple matching references are an error
in selection, not permission to take their minimum time. Missing commit metadata
is reported as missing rather than inferred from a repository's current tip.

## Models and native packages

`packages.json` contains nine explicitly selected model/backend recipes and
27 role bindings. They distinguish normal CPU/RVV and vendor IME2 policies and
preserve independent prefill/decode protocols. A local model profile names its
`package_recipe`, toolchain, normal binary, integrated binary, model file, fixed
reference and deployed package.

```sh
weft-bench prepare-model --target k1 --model deepseek8b-q4_k_m \
  --config bench.local.json --output output/current/k1/model-package
weft-bench e2e deepseek8b-q4_k_m --target k1 --config bench.local.json --phase both
```

After preparation, set the profile's candidate `package` to the returned native
deployment location before the model run. This explicit selection avoids changing
a deployed package merely because an operator microbenchmark was faster.
The adapter loads already-linked artifacts; timing does not include compilation
or an additional thread team. Numeric validation and actual replacement counts
are separate from throughput.

## Output ownership

Use `output/baselines`, `output/current`, and `output/reports`; keep native
artifacts next to the observation that owns them. The CLI's stable transaction
writes a pending result and then replaces current, including failure status.
It refuses concurrent writers and refuses to overwrite a configured fixed
baseline. Interrupted pending data remains inspectable. Retain necessary failure
evidence and fixed baselines, not one directory per tuning attempt.
## Reference identity

Each selected case records both its executable adapter name (`reference`, such as
`ggml`) and its reference catalog identity (`reference_catalog`). `show` expands
the latter as `reference_details`, including provider, source and call boundary.
The four K1 IME1 and two K3 IME2 cases explicitly point to the vendor matrix
entries; ordinary GGML/RVV rows remain separate. An unmatched case keeps a null
reference rather than receiving a provider inferred from its kernel name.

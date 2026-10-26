# Triton–Weft integration

Bench calls the installed Triton–Weft compiler through its public interfaces:
`@triton.jit(model="weft")`, `triton.compile`, `CompiledKernel.export`, `triton.load`
and `triton.Package.create`. The compiler owns lowering, autotuning, caching and
native code generation. Bench owns workload inputs, numerical oracles, measurement
and results. [Installation](install.md) and [target configuration](targets.md)
describe the environment setup.

## Native deployment and explicit cross compilation

Set the implementation factory to
`weft_bench.adapters.triton_native:TritonAdapter`. An implementation name such as
`weft` or `candidate` is only a label; the factory selects the compiler path.

| Input | Configuration |
|---|---|
| Source | Host `python`, optional `pythonpath`, `source`, `symbol`, `signature`, `constexprs`, `backend_options`, `toolchain` |
| Exported AOT | `artifact` directory with native library, manifest, header and runtime dependencies |
| Single instance | `entry: "single"` (default) |
| Native grid | `entry: "grid"`, `grid`, `num_cpu_threads` |

The target declares its ISA/VLEN and execution-side `triton_runtime.include` and
`triton_runtime.library`. Cross compilation uses the configured host toolchain and
target sysroot. Bench loads the exported candidate library directly; its C++ compiler
builds the measurement worker, not a second version of the candidate C.

An implementation can specify `compile_target` independently of the measurement
machine's `target`. For example, `{"arch":"x86_64", "march":"x86-64-v4",
"abi":"sysv", "vlen_bits":0}` selects the compiler ISA while an execution target
can describe the actual host as `sapphirerapids`. The exported manifest is checked
against the compiler descriptor, which is saved for replay. Fixed references keep
their original machine and measurement contracts. `architecture` is accepted as
an alias for `arch`; legacy configurations with an `rv64` march still select
`riscv64`. Other architectures must be explicit. The product SDK checks whether
the executing CPU supports the artifact.

The typed artifact ABI supplies argument order, encoding, access, dimensions and
strides. View arguments must match the declared logical shape. Pointer arguments
use the workload's actual storage and explicit size/stride arguments; an empty
pointer shape does not mean one element. No adapter parses C parameter names.

Native grid execution invokes the product launcher once for the whole workload.
Its typed ABI must consume the whole workload. A packed-row workload can use an
author grid kernel with batch-shaped Views; Bench then binds its complete batch
buffers once. A leaf ABI that still requires an outer row/tile loop is rejected
for grid measurement. Runtime capability checks remain active. The configured CPU
affinity must cover the intended worker set.

## Timing scopes

| Scope | Included |
|---|---|
| Native single instance | Complete candidate C ABI call |
| Native grid | Complete native grid launch, worker scheduling and synchronization |
| Python JIT grid | Warm `kernel[grid]` call, including Python binding and runtime guards |
| Model phase | The selected llama.cpp prefill or decode measurement window |

Compilation, allocation, input generation, numerical checking, state restoration
and first runtime initialization are outside normal timing. Work required by the
algorithm, such as projection quantization, stays inside the call. Existing multi-row
dot and packed IME cases retain their original boundaries. A grid time cannot be
divided by its instance count and presented as a leaf-call measurement.

An adapter change does not invalidate an otherwise identical fixed baseline.
Shape, inputs, cache mode, call scope and execution settings determine whether a
comparison is meaningful. Mismatches remain visible in results and `explain`.

## Target-side Python JIT

`weft_bench.adapters.triton_jit:TritonJitAdapter` uses a configured target-side
`python`, optional `environment` and `toolchain`, with source/symbol, signature,
constexprs, grid and thread options. Install both packages on the target. Catalog
sources are imported from that installed Bench package; custom source packages are
transferred with their relative imports. No host-produced kernel library is an
input to this route. An empty toolchain configuration selects the target Triton
wheel's bundled Clang and discovers the local system runtime.

The target's Canonical interface describes the caller storage before any native
compilation. The existing workload supplies typed, packed and strided storage,
lookup tables, workspaces, persistent layouts and the numerical oracle. Multiple
outputs and overlapping aliases borrow their original addresses. Mutable state
uses the workload's reset outside every measured invocation. Leaf programs keep
the existing row/tile traversal; Bench does not silently convert them to a different
whole-workload algorithm. Buffer construction and fixture preparation are outside
Python call timing.

`--cache cold` selects cold **data**, independently of the compilation cache.
The native fixture restores mutable state and traverses the configured eviction
buffer before each timed Python invocation, using the same eviction routine as
the native comparison. Cold-data samples contain one complete workload call;
eviction and state restoration are excluded. The returned `data_cache` and
`eviction_bytes` describe this boundary. First compilation, in-process kernel
reuse and fresh-process disk-cache reuse are still checked separately. Reuse
mode retains batched hot calls; both modes honor the warmup time budget.

For one Python call over a packed row batch, select an author grid entry whose
weight View has shape `(rows, K)` and whose output View has shape `(rows,)`.
The corpus provides `iq1_s_grid`, `iq1_m_grid`, `iq2_xs_grid` and `iq2_xxs_grid`
in `weft_bench.kernels.ggml.vec_dot.grouped_lookup`. Each uses `tl.program_id(0)`
to select its row and calls the same numerical helper as the leaf program.
Configure `grid: [rows]`, `entry: "grid"` and an explicit `num_cpu_threads`;
the source also guards excess grid instances. Both JIT and native grid measurement
bind the complete batch once, without a Bench-generated row loop. Their
`launches_per_workload` is one. This explicit variant does not change the retained
leaf/native reference boundary or silently replace a configured author program.

The target calls the normal `kernel[grid](...)`. Compiler specializations, dependency
capture, tuning, loading and launching belong to Triton. A decorated autotuner stays
the same Triton autotuner; Bench requires its configurations to have a compatible
caller-visible storage contract. A failed JIT invocation remains failed.

Results retain these separate measurements:

- `details.jit.runtime`: first compile-and-call, first in-process cache hit,
  compilation events and the number of launches covering the workload.
- `details.jit.python_jit`: warm Python calls, including normal binding and guards.
- `details.jit.disk_cache`: an independent Python process importing the same source
  and calling JIT again. Triton's compilation listener must report cache hits.
- The main native timing row: the original worker calls the artifact exported by
  that successful JIT invocation, with the unchanged native measurement boundary.

Only the main native row is compared with a fixed native GGML reference. Neither
Python overhead nor cold compilation is included in that speedup. This additional
native measurement consumes the JIT result; it cannot rescue a failed JIT case.

For a stateless workload with one JIT launch, `profile_runtime: true` in the
implementation configuration records an additional component profile under
`details.jit.runtime.runtime_profile`. It instruments that `CompiledKernel`'s
existing native launcher, including C++ argument preparation, worker capability
checks, grid dispatch and computation. The remainder is Python dispatch and
specialization. Instrumented totals and their overhead remain separate from the
plain hot-call samples; no component is subtracted from the reported latency.
The original callable is restored even when a launch fails.

Python JIT uses the product's native target detection on RISC-V or x86. Bench
checks the requested execution architecture before launch and records the detected
compiler target and actual affinity. Registration and CPU binding precede detection.
An optional `compile_target` constrains architecture, ABI, VLEN and required
extensions; additional discovered capabilities are retained. It does not override
the product driver with a machine name. Explicit cross compilation remains a
separate deployment operation.

Optional `check_python_aot` checks exported/reloaded execution.
`check_torch_compile` checks first and cached `torch.compile(fullgraph=True)` calls
in a compatible configured framework environment. These checks are outside warm
timing and default to off. Torch capture checks use dense CPU Tensors borrowing
the same storage. Packed Encoding cases use the public `tl.View` interface. Data
cache mode is currently `reuse`; compiler cold starts and compiler cache reuse are
recorded separately. Further framework contracts use explicitly selected
[verification cases](verification.md).

Optional `profile_runtime` retains normal uninstrumented samples and records
available Python runtime segments separately. If the installed runtime uses a C++
launcher that cannot expose those segments, they are `unavailable`; unobserved
internal work is not reported as zero.

For a buffer-boundary investigation, `guard_input_ends: ["W", "X"]` places
the named read-only input buffers immediately before inaccessible memory pages.
Use `--check-only` and an entry that consumes the complete workload in one call.
The adapter copies the existing workload bytes, preserving Encoding, shape and
strides; it does not generate new inputs or change the numerical tolerance.
Overlapping arguments and incompatible alignment are rejected. Input data pages
are read-only, and the applied bounds are printed before launch so a process crash
retains evidence of the check. Guarded runs provide correctness evidence only,
never a performance ratio.

## llama.cpp packages

Set `llama.package_backend` to `triton` and configure `llama.triton` with the
target Python, `compilation_host: "target"` and its native toolchain. Source roles
are compiled on the selected execution cores, using discovered target capabilities,
then packaged there through `triton.integrations.llama_cpp.create_package`.
Source bindings specify the role's signature, constexprs and backend options.
`--replace ROLE=ARTIFACT` can also deploy an explicitly exported native artifact.
The package and its single-instance ABI are bound before inference; GGML workers
call the native functions without Python, compilation or another thread team.
Model source compilation defaults to `compilation_host: "target"`, including a
local x86 target. The recipe's ISA requirements constrain the discovered CPU;
they do not replace its capabilities with a reduced feature list. An explicit
`compilation_host: "host"` with a cross toolchain retains the separate
cross-deployment path. Neither deployment mode counts as Python JIT corpus coverage.

Use compatible normal and integrated llama.cpp builds with matching ordinary
backend settings. Replacement roles and measurement phases are independent:
`--phase prefill|decode|both` selects the windows; the package selects the hooks.
Consult `list --kind replacements` for the current role catalog, including distinct
prefill/decode IME entries. Only measured-window call-count increments establish a
hit. `not_hit` means an observed zero count; `not_observed` means no usable observation.

`e2e --validate --validation-reference PATH` checks full-vocabulary logits for the
configured prompt and teacher-forced decode sample outside phase timing. Its saved
normal-model numerical reference is separate from the timing baseline. Select
`llama.validation` prompt length, decode steps and tolerances before execution;
this sample does not establish correctness for every model input.

## Diagnosis and reusable workloads

Native replay defaults to a saved artifact. Source-origin runs can use `--source`
with the recorded Triton environment/options; the legacy `--compiler` flag does
not select a Triton compiler. Historical flat snapshots with relative imports
require the matching package checkout. Preserve available IR, C, compiler diagnostics
and the actual executed artifact; compilation alone is not numerical evidence.

Optional `shared-semantics` and `pointer-semantics` workloads share the existing
worker. They cover ordinary array operations, masked/strided pointers, reductions,
helper control flow and typed operations. Alias cases restore state before each
call; Linux protected-page cases detect forbidden masked accesses. Scan and atomic
cases also use this worker, with explicit mutable state and returned-old-value
checks. These workloads do not enter `quick` or invent GGML performance references.

`causal_attention` is an optional complete attention workload with grouped-query
heads, a causal prefix and an independent numerical oracle. Its full candidate call
is timed, including required internal state updates. It currently has no bound GGML
performance implementation.

The [Codex plugin](codex-plugin.md) exposes these same paths. Its environment names,
compiler installations, machine addresses and model paths remain external settings.

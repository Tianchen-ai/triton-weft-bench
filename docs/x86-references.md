# x86 reference libraries and model baselines

Use the same inputs, native worker, timer and result files on x86. A reference-only
environment needs a native compiler and the selected libraries; it does not need
Weft, Triton, PyTorch or a working Weft x86 compiler backend.

Copy [the configuration example](../configs/x86-reference.example.json) outside the
checkout and set library, compiler, model and output paths. Choose physical cores
from one socket for the first comparison. `deployment.numa.command` may name a
privately installed `numactl`; omit the NUMA object when no policy is requested.
The example's core numbers and paths are placeholders for your machine.

```sh
export WEFT_BENCH_CONFIG=/work/x86/bench.local.json
weft-bench environment --target x86-1t --output /work/x86/capabilities
weft-bench op --suite quick --impl ggml --target x86-1t --output /work/x86/ref-ggml
weft-bench op gemm_f32 --impl openblas,onednn --target x86-8t --output /work/x86/ref-f32
weft-bench op gemm_bf16_f32 --impl onednn --target x86-8t --output /work/x86/ref-bf16
weft-bench op gemm_s8_s32 --impl onednn --target x86-8t --output /work/x86/ref-s8
weft-bench e2e llama2-7b-q4_0 --impl normal --target x86-8t --phase both --output /work/x86/model-ref
```

`default_implementations` can select `ggml` for `op` and `normal` for `e2e`, as in
the example. These commands prepare genuine references; they do not label a library
call as a Weft candidate. A Weft source/artifact, native grid or Python host call
can be added through its existing adapter when that compiler supports the target.
Their timing scopes remain distinct.

Native results also retain calling-thread CPU samples and, when Linux exposes
them, CPU frequency snapshots before and after sampling. The CSV includes their
median, scope and frequency fields. CPU time excludes descheduling and other
OpenMP/library workers; it is diagnostic data, not a replacement performance
score. Frequency snapshots are not an average clock and are never used to
normalize speedups. Warmup and timing windows retain their existing meaning;
fixed reference results without these diagnostic fields remain usable.

An output-only workload can use a native kernel whose `Y` effect is readwrite
when the kernel writes its initial partial sum before reading it. The worker
poisons that output before the numerical check (NaN for floating storage,
nonzero patterns for integers), so dependence on caller-provided zeros fails
the existing oracle. The kernel's initialization and intermediate output
reads/writes remain inside every timed invocation. Input and inout access
contracts are unchanged.

An author may declare extra dense Views for invocation scratch and list their
native argument names in the Triton adapter's `workspaces` option, for example
`"workspaces": ["Scratch", "Activation"]`. Bench allocates each buffer from the
artifact's typed ABI and bound dimensions before timing, then poisons its initial
contents. The kernel must populate every workspace element it consumes inside
each invocation. Packing is therefore included in the operator's measured time.
`invocation_workspaces` records each shape, byte size and these two boundaries.
These are separate writable contiguous dense buffers; aliasing workload ports,
encoded layouts and stronger-than-element alignment are not supported by this
allocator. Persistent weight preparation remains a separate declared contract.

## Choose the reference by computation

| Reference | Workloads | Timed boundary |
|---|---|---|
| GGML | Quantized dot/rows, quantize/dequantize, projection, norms, RoPE, softmax and F32 GEMM | The selected native function or prepared CPU graph |
| OpenBLAS | `gemm_f32` | LP64 CBLAS SGEMM, X[M,K] × W[N,K]ᵀ → Y[M,N] |
| oneDNN | `gemm_f32`, `gemm_bf16_f32`, `gemm_s8_s32` | Plain-stride matmul `execute` + `wait`, including internal packing |
| Normal llama.cpp | Configured GGUF models | Separate prefill/decode phases, excluding loading and process startup |

oneDNN uses explicit strides `{K,1}` for X, `{1,K}` for the logical K×N weight
view, and `{N,1}` for Y. The primitive and user scratchpad are prepared before
timing. No external reorder or `format::any` prepacking is hidden outside the
call. Such a future prepared-weight implementation needs its own timing scope.
F32 math is explicitly strict; BF16/F32 and S8/S32 are separate typed workloads.

OpenBLAS requires `cblas.h`, `openblas_config.h` and its shared library. oneDNN
requires `oneapi/dnnl/dnnl.hpp` and its shared library with OMP or sequential CPU
runtime. Keep the worker's compiler and OpenMP headers/library compatible with
the selected builds. `references` entries can retain source/build/version fields,
which are saved with the run. GGML retains its normal library and compile database;
the existing RoPE bridge compiles with the original `ops.cpp` options.

## Inputs and numerical checks

Every normal `op` invocation checks outputs before sampling. `op --check-only`
uses that same worker for one checked call and returns no timing samples. The
configured `verify` route can select these native implementations by their real
names; it needs no compiler runtime in a reference-only target.

BF16 inputs are rounded to stored BF16 before the independent scalar oracle runs.
Python lists contain numeric values; NumPy inputs for BF16 use uint16 bit storage,
and raw files use the same two-byte representation. S8 generation spans the full
signed range and includes -128, 127, zero and ±1. S8 products accumulate in an
int64 oracle and must fit the declared S32 output. Typed matmul checks every
output for at most 4096 outputs; larger shapes explicitly report a 64-output
sample. Existing F32 GEMM keeps its established sampled oracle.

GGML quantization storage rules can differ across native paths. The example
selects `quantization_contracts.q8_k = "first-absolute-max-rtne"`: an independent
scalar byte oracle keeps the first absolute extremum and round-to-nearest-even.
Legacy RVV configuration retains its existing extrema/tie rule. The chosen rule
is part of the input contract, so unlike rules cannot produce a comparison ratio.

The existing GGML dequantization reference checks consistency against the same
library decoder. Its result is labeled `library-reference-self-consistency` with
`oracle.independent=false`; this is not an independent numerical proof. A genuine
Weft dequantizer can still be compared against this library reference. Dot,
projection, dense and the new typed matmuls retain their separate mathematical
checks; no tolerance is relaxed for reference-only execution.

## Persistent model projection weights

Set `ggml.projection_weights` to `model-default` for the CPU model-projection
reference. Bench queries the CPU backend's extra buffer types in their normal
order, tests each with `ggml_backend_dev_supports_op`, and falls back to the CPU
default buffer. Weights have a separate context/buffer marked `WEIGHTS`;
`tensor_set` performs the library's normal weight preparation before timing.
Activation and output storage remain ordinary CPU buffers. Graph execution,
including activation quantization, stays inside the timed call.

The result records the selected buffer type and weight-preparation boundary, with
scope suffix `/persistent-model-default-weights`. Plain projection retains its
existing scope and remains the default for configurations that omit the setting.
An AMX-named weight buffer alone does not prove tile instructions: for example,
GGML can use VNNI for decode or AVX512 for F16 with such a buffer. Bench records
buffer selection plus successful graph execution without guessing that an AMX
instruction executed.

For complete projections, the native runner also records `projection_boundary`:
the prepared W/X buffers, setup and preparation statements, and the timed call.
Weights are persistent inputs prepared before timing; activation preparation and
the complete projection belong to the call. GGML's model-default weight layout
is an implementation choice within that boundary.

`op --baseline`, `compare`, and `check` accept the plain/suffixed scope pair only
when both records establish this complete-operator boundary. Existing fixed GGML
records can supply their observed model-weight buffer and preparation evidence;
they need no rewrite or remeasurement. The original scopes stay in JSON/CSV, and
`comparison_basis` explains the accepted equivalence. All other contract fields
remain exact, including input policy, dtype, shape, threads, CPU/NUMA and cache
protocol. Missing boundary evidence or a leaf/grid/host-call scope is not treated
as equivalent. Numerical failures remain failures; this timing rule does not
make different activation rounding policies numerically interchangeable.

### Mixed F16 weights and F32 activations

`f16_f32_projection_decode` (M=1) and `f16_f32_projection_prefill` (M=128)
use N=K=4096 by default. Their inputs are stored F16 W and original F32 X;
the independent reference decodes W, multiplies by the unchanged X, and sums in
F64 before storing F32. The candidate ABI has W/X/Y, with no Xh/Xq scratch.
The complete projection is timed under `projection-preserves-f32-activation`.
The existing `f16_projection_*` cases retain their explicit X-to-F16 conversion
and original oracle; their results are not interchangeable with these cases.

Keep `ggml.projection_weights=model-default`. The mixed reference currently
recognizes GGML's AMX-buffer F16 path, which uses AVX-512 F32 accumulation.
Bench records the upstream source/build evidence, selected weight buffer,
resolved compute symbol and DSO, then checks that this dispatch preserves F32 X.
An unrecognized source is unsupported; an incompatible selected path reports
`contract_mismatch` before sampling. Normal GGML selection is not disabled or
replaced. The usual numerical check still applies with the existing tolerances.

```sh
weft-bench op f16_f32_projection_decode --impl ggml --target x86-1t --output /work/x86/mixed-ref
weft-bench op f16_f32_projection_decode --impl weft --target x86-1t \
  --baseline /work/x86/mixed-ref --baseline-impl ggml --output /work/x86/mixed-current
```

## Freeze a normal model's numerical reference

Normal-only `e2e --validate` captures fixed-prompt logits and the teacher-forced
token sequence with the existing native validation helper. It checks shape,
finite values and token ranges and labels the result `reference-logits-captured`;
it does not compare the library against itself. To associate a numerical
reference with phase times that already exist:

```sh
weft-bench capture-model-reference llama2-7b-q4_0 --target x86-8t \
  --output /work/x86/model-reference --attach-run /work/x86/model-ref
```

The public Python API is `capture_model_reference(model, target, output=...)`.
An existing matching reference is reused without running the model again. Later
real candidates use the same directory through `--validation-reference`, retaining
the existing logit tolerances. Capturing or attaching a reference does not remeasure
or overwrite phase timing samples.

## Threads, affinity and execution evidence

Dot leaves, multi-row serial dot loops, and direct Norm/RoPE/softmax calls keep
their one-thread boundary. GGML complete GEMM/projection, OpenBLAS and oneDNN
matmul accept explicit thread limits. Unsupported combinations return an explicit
result instead of labeling a serial leaf as an eight-thread kernel. Model thread
counts and CPU sets are separately configurable.

A native candidate with `entry=single` also uses one thread for complete GEMM or
projection. An 8/52-thread target or OpenMP environment does not create parallel
work inside that entry: Bench reports unsupported before compiling or running it.
Use the existing explicit grid/framework route for a parallel candidate. Grid
results retain their actual `num_cpu_threads` and distinct launch scope; saved
records whose single/grid entry contradicts the recorded threads cannot produce
a comparison ratio.

`environment` records CPUID, XCR0, OS extended-state permissions, actual affinity,
CPU core/socket/SMT mapping and effective NUMA policy. It only observes those
facts; it does not execute AMX or request tile permissions. Each native worker can
record the same facts with `record_environment=true`. Model observations come
from a separate probe process under the same deployment policy.

Results also capture the actual reference entry and loaded DSO. OpenBLAS reports
its selected core/configuration and thread setting. oneDNN reports its primitive
implementation, strict math setting, thread limit and post-execution tile state.
`amx_executed` is set only after an AMX-named primitive has actually executed and
passed its oracle. An advertised AMX flag or an enabled tile permission alone is
not an AMX execution result.

## Keep and reuse a fixed baseline

After establishing a result, retain its JSON/CSV and artifacts. Subsequent runs
can use `--baseline RUN --baseline-impl NAME`, or compare two saved runs without
executing either library again:

```sh
weft-bench compare /work/x86/ref-f32 /work/x86/current-f32 --baseline-impl openblas
weft-bench explain /work/x86/current-f32 --case gemm_f32 --impl onednn
weft-bench replay /work/x86/ref-bf16 --case gemm_bf16_f32 --impl onednn --output /work/x86/replay
```

Scope, stored dtype, shape, cache protocol, CPU/thread/NUMA settings and input
contract determine whether a ratio is meaningful. Separate worker directories
retain each implementation's source, headers, commands and raw output when a
comparison uses multiple native providers. No complete performance report is
required to continue development.

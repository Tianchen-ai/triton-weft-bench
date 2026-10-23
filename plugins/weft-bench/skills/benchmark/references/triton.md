# Triton–Weft JIT and native deployment

Use `TritonJitAdapter` for normal source-corpus validation. It runs the installed
target Triton and subsequently measures that same JIT-generated native artifact.
`TritonAdapter` is the explicit source/AOT deployment path; running an existing
library is not evidence of JIT coverage. Both retain the existing native workload
and fixed reference boundary. See the target-native section below for the records.

Use the task's explicitly configured Bench environment. In the local plugin settings,
an environment has the same `command` argument array and `config` path as the default.
This selects an installed Bench build/configuration; compilation still belongs to Triton.
The plugin name remains Weft Bench. An explicit environment selects the actual
Triton–Weft installation; neither the plugin name nor a `weft` implementation label
establishes which compiler is configured. Inspect `--plugin-info` and the selected
implementation before changing compiler environments.

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft op affine \
  --target k1 --impl triton_source --baseline /path/to/affine-ref --output /path/to/runs/affine
```

Environment and implementation names come from local configuration, not from the
runner. The examples use `triton-weft`, `triton_source` and `triton_aot`; inspect
the selected config to find the actual names and source/artifact bindings.

The explicit native deployment factory is `weft_bench.adapters.triton_native:TritonAdapter`.
Source configuration supplies the selected Triton `python`, optional `pythonpath`
list, `source`, `symbol`, ASTSource `signature`, `constexprs`, `backend_options` and
native `toolchain`. The public compile route is `triton.compile(ASTSource(...),
target=CPUTarget(...), options=...)`, followed by `CompiledKernel.export`.
An explicit empty `signature: {}` lets the Weft frontend use the kernel's typed
annotations. `--weft SOURCE:SYMBOL` overrides the chosen workload binding, and
`--weft ARTIFACT` preserves the configured Triton native provider. These overrides
also work through the Python comparison API; an artifact does not fall back to
the legacy artifact compiler.
AOT configuration supplies `artifact` instead of `source`. Both consume the actual
exported shared library; the worker never recompiles candidate C.

Multiple single-instance Triton candidates in `--impl` share one immutable input
preparation. Each artifact is loaded independently and has its own output,
workspace, numerical check and timing samples. Grid and other provider candidates
retain their separate workers.

For Q8_K activation projections, an explicit target setting
`projection_activation_contracts.q8_k` selects `extrema-lo-rtne` or
`first-absolute-max-rtne` for the independent packed-byte oracle. Before timing,
Bench checks every oracle byte against the actual GGML CPU quantizer and every
candidate workspace byte against that oracle. This avoids treating a scalar
reference's fused rounding trick as the SIMD rounding contract. It changes the
untimed oracle, not the input data, numerical tolerance or measured call.

Set implementation `compile_target` to select the compiler descriptor independently
of the measurement machine's `target`. For example:

```json
{"compile_target": {"arch": "x86_64", "march": "x86-64-v4", "abi": "sysv", "vlen_bits": 0}}
```

The execution machine may remain `architecture: x86_64, march: sapphirerapids`
with its existing affinity/thread settings. Bench validates the exported target
against `compile_target` and retains it for replay; the SDK checks execution
support. Do not rewrite fixed reference contracts to match a compiler ISA name.
`architecture` is an alias for `arch`; legacy `rv64` configurations still select
`riscv64`. Use an explicit architecture for other targets. Installed candidates
select the installed Python and matching native SDK; set `pythonpath: []` when no
development overlay is intended.

A configured `program` names a product catalog Program. Its source is prepared
through `Program.signature`, `Config` and `Program.compile`; ordinary kernels keep
the ASTSource route. For configured numerical and installed-runtime validation,
use [verify](verification.md). This is separate from native timing below.

The external target config supplies `triton_runtime.include` and
`triton_runtime.library` on the execution machine. They must expose the real native
SDK's `weft_kernel_open_native`, `weft_kernel_get_entry` and `weft_kernel_check_worker`.
Missing runtime support is a setup/unsupported result, not permission to bypass checks.

`entry: "single"` is the default. It uses the single-instance ABI in manifest.json,
checks the bounded measurement thread, and directly calls the native pointer without
creating an OpenMP team. Existing compatible workloads and references remain usable.
Use a one-thread target for this entry, including complete GEMM/projection. Setting
target threads or `OMP_NUM_THREADS` to 8/52 does not parallelize a direct instance;
Bench returns unsupported and does not label it as a multithread candidate.

`entry: "grid"` also requires explicit `grid: [x, y, z]` and `num_cpu_threads`.
Use the exported `manifest.grid` and launcher declaration, not the `*_instance`
pointer. The generated native launcher calls the SDK guard on every worker before
writes. The result is a separate `grid-launch` scope with grid/thread/CPU parameters;
do not divide that time into a dot-leaf speedup or compare it automatically with
a single-instance baseline. The author kernel must partition its work by program id.
Grid support requires a whole-workload ABI. Packed-row workloads can use an author
grid entry with batch-shaped Views, which bind the existing complete batch buffers
once. Leaf ABIs that still need an outer row/tile loop report unsupported instead
of wrapping a second dispatch loop around a grid. The installed corpus includes
`iq1_s_grid`, `iq1_m_grid`, `iq2_xs_grid` and `iq2_xxs_grid` in
`weft_bench.kernels.ggml.vec_dot.grouped_lookup`; these select rows with
`tl.program_id(0)` and reuse the corresponding leaf's numerical helper.
The recorded grid worker count comes from `num_cpu_threads`, not the target's
nominal thread setting. Keep its launch scope; saved single/grid records with
contradictory thread metadata cannot produce a comparison ratio. External native
or framework adapters retain their existing explicit execution contracts.

`replay` uses the saved native artifact. For a source-origin run, `--source FILE`
recompiles through its recorded Triton environment/signature/options. The old
`--compiler /path/to/weft-compile` override does not apply to Triton candidates.
When fixed reference protocols or inputs differ, use `explain` to read the specific
comparison differences. Preserve the existing reference rather than changing its labels.

## Source parameters and Triton Config

`tl.auto("B")` is a compile-time source parameter. Supply its value through the
kernel's declared constexpr argument, `ASTSource(..., constexprs={"B": 32})`,
or explicit compilation `options.meta` when the source exposes an unbound symbol.
For real tuning, `triton.Config({"B": 32}, num_cpu_threads=1)` supplies a source
candidate; `backend_options` supplies target-specific physical choices. Triton owns
compilation, candidate timing and caching. Bench does not invent another tuner or
add RVV-specific options to x86 configurations.

Named `auto("B")` does not itself declare a finite domain. Two tuning configs B32
and B64 do not make B48 invalid. For `auto(32, 64)`, obtain the generated parameter
name and declared choices from the actual Canonical interface. A missing required
binding, nonpositive selection or out-of-domain choice must produce its specific
diagnostic. Explicit meta that conflicts with an already fixed constexpr must also
be rejected; never treat an unrelated compiler exception as an expected rejection.

Each selected tile produces a fixed compiled candidate. Dynamic View ABI shapes,
strides and offsets can reuse that candidate; changing the source tile must select
distinct compiled code. The tuning cache additionally distinguishes workload
geometry and execution conditions. Tuning trials use private state and the winning
call updates caller state once. Internal tuning measurements are not a native
operator speedup result.

Export the selected compiled kernel. Its manifest `options.meta` and ABI `bindings`
retain the source selection; `triton.load` does not retune or require that tile as
a runtime argument. Runtime View dimensions remain in `shape_parameters`.
Development-overlay evidence and installed-wheel evidence describe their respective
compiler environments; use the task's verified installation for normal work.
View extents may refer to declared constexpr parameters when the author intends
that specialization. Leave ordinary runtime M/N/K dimensions dynamic; a workload
shape is not an implicit request to turn it into a source constexpr.

## Target-native Python JIT

The explicit factory `weft_bench.adapters.triton_jit:TritonJitAdapter` runs
`kernel[grid]` on a RISC-V or x86 machine, borrowing the existing workload storage.
Configure the target-side `python`, optional `environment`, native `toolchain`, author
`source`/`symbol`, `constexprs`, `grid` and `num_cpu_threads`. Compilation belongs to
Triton; Bench supplies inputs, the numerical oracle and outer execution.

JIT selects its target through the product's native detection. Bench records that
target and checks the requested architecture; an explicit `compile_target` is an
required ISA/ABI/VLEN constraint, not a driver override. Registration and affinity
precede discovery; extra real capabilities remain available. To compile v3 on a v4 host, use source/AOT
with an explicit compiler target. CPU Tensor eager/JIT/AOT support does not establish
`torch.compile`/Inductor support for the installed Torch version.

The primary timing row uses the original native boundary and the artifact exported
by the successful target JIT call. `details.jit.runtime` separately records cold
compile+call and in-process reuse; `details.jit.python_jit` records warm Python
calls. `details.jit.disk_cache` records a fresh process calling JIT from source,
whose Triton compilation-listener events must be cache hits. Python timings include
normal argument binding, guards and grid launch, with the number of calls covering
the workload recorded explicitly. Never compare them against native-only GGML.
Failed JIT cannot fall back to AOT. Installed package paths, discovered target and
affinity remain in the evidence. Compiler caches and dumps stay within Bench output.

Set `check_python_aot: true` to export and reload the just-compiled artifact through
`triton.load(...)[grid]`. Set `check_torch_compile: true` for an additional
`torch.compile(fullgraph=True)` numerical check when the selected private Torch
environment has the compiler's supported source integration. The latter is a
framework check, not the reported warm timing. Both options are off by default.
The fixture supports typed/packed Views, multiple outputs, explicit strides,
overlapping aliases, resettable state, workspaces and declared persistent layouts.
It reuses native workload traversal and oracles; each stateful trial resets outside
timing. Catalog kernels come from the installed target Bench package. The data-cache
mode remains `reuse`. Torch checks use dense CPU Tensors; this does not establish
arbitrary graph coverage. Read actual case results before asserting coverage.

JIT replay uses the saved source and recorded target environment, with the same
input snapshot. `--source FILE` selects a replacement source; the legacy compiler
override is rejected. Do not run saved fullgraph requests against shared Torch
unless that environment was explicitly configured for the integration.

## Native llama.cpp packages

Set `target.llama.package_backend` to `triton` and `llama.triton.compilation_host`
to `target`, with the installed target Python and native toolchain. `prepare-model`
compiles source roles on the selected target and packages them through Triton's
llama.cpp integration before inference. Explicit artifact replacements use their
already-linked libraries. The optional `host` compilation location is an explicit
cross-deployment route. Source roles retain signature, constexprs and physical
options; there is no legacy compiler fallback. GGML workers call the native entry
without Python, compilation or another thread team during measurement.

Use a compatible integrated llama.cpp binary and native SDK. A package role is
not a phase selection: `ggml.ime1/q4_centered_prefill` only matches that hook;
it cannot stand in for a decode IME kernel. `--phase` still selects which model
measurement to run. Report the measured-window role counters alongside timings;
`not_hit` means the observed phase did not call the replacement even if elapsed
time changed. Fixed normal llama.cpp results remain read-only baselines.

The K3 IME2 signed-byte slots are `ggml.ime2/i8_prefill` (M4) and
`ggml.ime2/i8_decode` (M1). `ime2_i8_prefill` and `ime2_i8_decode` measure
those packed native calls against the actual vendor kernel. They consume the
vendor block32 I8 weight/activation storage used for Q6_K and Q8_0; they do not
cover the separate scaled FP16 Q4 path. Select the same explicit M/N/K as the
saved reference rather than relying on the workload's default K.

Keep the standalone SpacemiT backend enabled in both model builds and runtime
environments. The integrated Triton SDK supplies `install_spacemit_source`
for its native call sites. Record IME2 counters and run `validate-model` against
the matched vendor-enabled logits reference before using new package timings.
CPU/RVV-only model results are a different baseline and cannot stand in for it.

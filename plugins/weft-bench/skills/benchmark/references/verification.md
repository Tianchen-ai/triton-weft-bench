# Configured compiler and runtime verification

Use `verify` when the task calls for numerical, compiler or installed-runtime
validation. For latency and comparisons use `op`, `check` or `e2e`. Verification
reuses the same workload inputs, numerical oracles, typed ABI and native adapters;
it does not remeasure fixed GGML or llama.cpp baselines.

Select the task's configured environment and verification config explicitly:

```sh
python3 "$BENCH_SKILL/scripts/bench.py" verify \
  --config /work/bench.local.json --target cpu --list
python3 "$BENCH_SKILL/scripts/bench.py" verify \
  --config /work/bench.local.json --target cpu --case 'operators/q4_k_dot' \
  --output /work/verify/cpu
```

Environment, target, group and case names come from configuration. `--list` does
not connect to a device. Select cases affected by the task; a complete configured
inventory is appropriate only when that is the requested scope.
Formal catalogs select their `weft` candidate and declared shapes for `verify`.
With the JIT adapter, source runs through the target's installed `kernel[grid]`;
the result retains in-process reuse, fresh-process cache reuse and the native
check of that same JIT artifact. Prepared artifacts and `--prepare-only` are
rejected for JIT entries. Use an explicit native adapter for AOT validation.

The repository's `configs/verification-riscv.example.json` lists optional scan,
atomic RMW/CAS/contention, noinline-helper and loop/context-return probes. Add
selected entries to the task's external config; keep the actual installed compiler
and target Python explicit. These probes do not imply support on older wheels.
For a framework upgrade, first verify the ordinary CPU Tensor and required Torch
interfaces in that private environment, then select the existing framework cases.
Use wheels matching architecture, Python ABI and libc, or build a compatible wheel;
do not copy a different target's installed package across incompatible ABIs.

Product examples can be configured as `kind: "usage"`: `contract: "triton_basics"`
selects one `operation`, `autotune_export` checks tuning/export/reload, and
`pytorch_cpu` checks the hand-written call and Inductor output. Supply the real
example source and its `imports` roots. The existing worker runs that source and
checks complete outputs; it does not treat printed previews as numerical evidence.
An entry may select its own `runtime` Python/environment/toolchain. The usage worker
honors `unset_environment` and loads the deployed author imports explicitly, so an
installed example can run without inheriting a compiler overlay.

For directed source-parameter work, `kind: "runtime", case: "source_auto"` uses
`request.source_auto_scenario` to select `tiles_cache`, `bindings` or `autotune`.
Supply unchanged public affine/state-update files with `runtime.source_files`
(relative worker filename to local source path); the worker deploys and records
them. `tiles_cache` checks two bound tiles, dynamic ABI reuse and AOT metadata;
`bindings` checks the actual missing/nonpositive/finite-domain/constexpr-conflict
diagnostics; `autotune` checks two real Triton configs against an independent oracle,
private trial state and the exported winner. Reuse the task's configuration, not
an expanded shape matrix. Installed checks select the installed Python, no overlay,
and `toolchain: {}` with external toolchain overrides cleared. These are separate
from development-overlay results; neither establishes operator latency.

Helper runtime cases reuse the same source on x86 and RVV: `helper_jit`,
`helper_noinline`, `helper_loop_returns` and `helper_context_return`. The last case
includes one-block and multi-block Views, preserving untouched or previously
completed blocks after a return. `helper_descriptors` combines fixed shaped tuple
returns, nested noinline calls and strided Views; `helper_level_capture` passes the
outer point and selected Views through a noinline boundary into public `subtiles`,
state/stage births and a nested helper. Both check complete outputs and padding
guards. Select the affected entries rather than expanding the suite. If an author
construct fails, retain its diagnostics and source; do not silently replace it with
an easier kernel. Installed checks use the same scenarios with the installed Python,
no overlay and `toolchain: {}`; wait for those results before treating them as a
validated installed workflow.

For common numerical primitives, select `case: "numerical_primitives"` and
`request.numerical_scenario`: `grouped_mac`, `sum_pairs`, `grouped_tail`, `lookup`
or `scalar_validity`. The first four cover actual public operations, multi-axis
and widening/wrap semantics, f32/group-three, partial-group validity and lookup
snapshot order. Integer oracles are exact; f32 uses the configured runtime tolerance.
The fifth follows tail scalars through a noinline helper's if/for/while returns and
the caller's structured carries/merge into pointer effects. It checks valid-input
histograms and inactive output guards without supplying an explicit effect mask.
The same five source cases can run in development or installed environments;
preserve which environment produced each result. They do not establish packed
storage lowering or a speedup.

Packed layouts use `case: "packed_layouts"` with `request.packed_scenario` set to
`natural`, `layered_joined` or `orders`. `case: "dynamic_rounding"` reuses one compiled
f32/f64-to-i8 kernel under the four target fenv modes, checks exact results and restores
the original floating-point environment. FE constants come from target `fenv.h`.
Its C shim needs a target C compiler: with an installed default candidate, use
`request.default_toolchain: true` and select the package's Clang for the shim in
`request.toolchain`; that shim setting is not passed to the candidate compiler.
Original dynamic Norm/projection, large stages, BF16 mixed dot and loop carries use
existing `program`/`native` entries. Keep their original source, shape and oracle.
Verification inputs accept the same preset, JSON path or mapping as the regular CLI.

## Prepare, execute and resume

- For an explicit native adapter, `--prepare-only --workers N` compiles source entries on the host using Triton's
  existing compiler. Its `preparation.json` reports `compiled`, an expected
  `rejected`, or the actual failure. It is not numerical evidence.
- `verification.prepared_entries` in the external target config selects one or
  more preparation manifests. Execution consumes their exported native libraries,
  without recompiling C. Program source entries use the product's
  `Program.signature`, `Config` and `Program.compile` APIs.
- Native operator validation invokes the prepared candidate once and checks its
  outputs. There is no warmup or timing calibration. Program and framework probes
  retain their configured sequences, such as two stateful calls or autotuning.
- Keep one execution queue per target. Host preparation can use the worker count
  suitable for the available machine; it does not parallelize device measurement.
- `--resume` continues pending entries. `--case ID --rerun failed` retries selected
  failures; `--case ID --rerun all` explicitly replaces a selected result.
  `--stop-after-current --output RUN` requests a stop at a case boundary.
- Preparation has its own ledger; it does not automatically skip cases passed in
  `verification.json`. Select the non-passed case IDs before parallel preparation,
  then use that same selection with the prepared config and original execution output.
  `--rerun failed` skips missing preparation records and pending execution entries;
  use selected `--rerun all` when those must be included. Keep manifests for same-ID
  scenarios with different source/shape bindings separate.
- Keep the selected compiler/runtime installation stable while a queue runs.
  A changed identity requires an explicit selection for rerun; an environment
  change during execution is recorded as `stale`, not a numerical failure.

## Read evidence and reproduce

`verification.json` includes both selected and whole-ledger counts. Use the selected
counts for a partial run: older pending entries may belong to a different selection.
Each entry links its current diagnostics under `cases/ID/current`; reruns retain
the first and latest failure. `failures.json` keeps the non-pass entries visible.

For a native operator, the entry's `result` points to ordinary Bench results:

```sh
weft-bench explain /work/verify/cpu/cases/operators/q4_k_dot/current/results.json
weft-bench replay /work/verify/cpu/cases/operators/q4_k_dot/current/results.json \
  --output /work/replay/q4_k_dot
```

Native replay defaults to the saved artifact when one exists. Source replay uses
the recorded Triton Python/toolchain, signature and options; `--compiler` is only
for the legacy Weft CLI. For a historical flat source snapshot with package-relative
imports, pass `--source /work/author/package/kernel.py` from the matching checkout;
the flat snapshot alone does not preserve its package dependencies. Framework,
runtime and usage entries are rerun through `verify --case ID --rerun all`.

Input/output snapshots and partial compiler IR are diagnostic evidence. A sampled
oracle covers only its declared output indices; exact unchanged-artifact evidence
may be reused with its original execution provenance, not relabeled as a new run.
`--coverage` maps an explicitly configured source inventory to cases and static
helper calls. Static reachability is not proof of runtime execution.
Track ordinary bindings, target-specific source/shape variants, other source families
and installed-wheel representatives separately. A passed subset does not establish
whole-repository coverage, framework support, or a newer compiler's execution results.

Report unsupported framework capabilities and hardware-inapplicable cases
explicitly. An expected compiler rejection has `checked: false`; its successful
expectation check is not a numerical pass. No correctness result implies a speedup.

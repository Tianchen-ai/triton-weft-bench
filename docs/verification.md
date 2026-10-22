# Configured maturity verification

`verify` enumerates and executes a selected product coverage plan. It uses the same
operator workloads, input contracts, numerical oracles and native adapters as `op`.
It does not time GGML references, rewrite fixed baselines or select model replacements.

```sh
weft-bench verify --config /work/bench.local.json --target cpu --list
weft-bench verify --config /work/bench.local.json --target cpu \
  --group operators --output /work/maturity/cpu
weft-bench verify --config /work/bench.local.json --target cpu \
  --output /work/maturity/cpu --resume
weft-bench verify --config /work/bench.local.json --target cpu \
  --case 'operators/q4_*' --output /work/maturity/cpu --rerun failed
```

`--list` is read-only and does not connect to a target. Groups and stable case IDs
come from configuration. By default, the `operators` group enumerates every binding
of `verification.source_implementation`, rather than the small performance suite.
One configured scenario is used for each binding. There is no shape/tuning sweep.

`--prepare-only --workers 2` compiles selected source entries on the host using
the same Triton adapter, without device execution. Its `preparation.json` distinguishes
`compiled` from failures. Set `verification.prepared_entries` to that file to execute
those exact artifacts later. `--resume` and `--rerun failed|all` also apply to preparation;
a targeted rebuild preserves the other entries in the manifest. Device queues stay serial.

For native operators, `Measurement(check_only=True)` prepares inputs, calls the
candidate once, collects its outputs and runs the existing oracle. It does not
warm up, calibrate or sample execution time. Results contain no median or speedup.
An oracle that samples outputs continues to describe its sampling scope explicitly.

Native workloads and implementation calls can declare `snapshots`, a mapping from
file-safe labels to C++ `std::vector` expressions. With `save_outputs=True`, the
shared worker exports them after its initial call, outside the timing interval,
including when the numerical check fails. Projection captures include the actual
`X`, `W` and candidate `Xq`, the independent `reference-Xq`, lookup tables and
`reference-Y`. Only the positions listed in the uint64 `reference-Y-indices`
buffer were computed by a sampled reference. These raw buffers are linked from
the result's `outputs`; exporting them does not change the numerical oracle.

## Configuration

The optional `verification` object belongs in a target's external configuration:

```json
{
  "source_implementation": "triton_source",
  "compiler_environment": "installed-candidate",
  "provenance_paths": ["/work/compiler/python/triton"],
  "runtime": {
    "python": "/opt/candidate/bin/python",
    "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
    "toolchain": {}
  },
  "operators": {
    "q4_k_dot": {
      "artifact": "/work/compiled/q4_k_dot/artifact",
      "compilation": {"source_snapshot": "/work/compiled/q4_k_dot/author-source.py"}
    }
  },
  "entries": [
    {
      "id": "programs/router",
      "group": "programs",
      "kind": "program",
      "case": "router",
      "artifact": "/work/compiled/router/artifact",
      "bindings": {}
    },
    {"id": "runtime/pid_grid", "group": "runtime", "kind": "runtime", "case": "pid_grid"},
    {"id": "framework/inductor_pointwise", "group": "framework", "kind": "framework", "case": "inductor_pointwise"}
  ]
}
```

An operator scenario can override `shape` and `inputs`. An explicit `artifact`
consumes the actual exported native library without recompiling its C output.
`compilation_failure` instead imports an existing failed compilation record, with
`error` and `diagnostics`; it remains a failure and does not trigger another compile.
Retain the producing compiler's records with imported artifacts. An artifact check
describes that artifact, not whatever source happens to be present later.

For source-backed catalog Programs, add `source`, `symbol` and
`options.program` to the `program` entry, with the existing Triton adapter options.
Preparation invokes the public `Program.signature`, `Config` and `Program.compile`
route. Other source kernels use `ASTSource`. `--coverage` reads `source_inventory`
or `inventory_roots` and maps definitions to configured cases and static helper
calls; helper reachability is not evidence that a device invocation executed it.

Runtime/framework entries can supply unchanged author files through
`runtime.source_files`: a mapping from a relative worker filename to its local
source path. These files are deployed alongside the probe and included in its
provenance snapshot. Destinations cannot replace worker files. For example:

```json
{
  "id": "source-auto/tiles_cache", "group": "source-auto",
  "kind": "runtime", "case": "source_auto",
  "request": {
    "architecture": "x86_64", "source_auto_scenario": "tiles_cache",
    "compile_target": {"arch": "x86_64", "march": "x86-64-v4", "abi": "sysv", "vlen_bits": 0}
  },
  "runtime": {
    "source_files": {
      "public_affine_strided.py": "/work/Triton-Weft/third_party/weft/source/weft/dense/affine_strided.py"
    }
  }
}
```

The directed `source_auto` probe has three scenarios. `tiles_cache` uses the
public affine source with B32/B64, two strided layouts, JIT reuse and exported
binding metadata. `bindings` checks missing/nonpositive bindings and a declared
finite domain; it requires the specific source-binding diagnostics. `autotune`
imports `public_state_update.py` through the same mapping, uses two real Triton
configs and checks private trial storage, caller updates, tuning reuse and the
exported winner against an independent oracle. Its internal tuning does not
produce a Bench latency or speedup claim. Choose only the requested scenarios.

Additional `native` entries specify an existing `case`, `source`/`symbol`, optional
`shape`/`inputs`, and implementation `options`. They reuse the same worker and ABI
binder. `program` entries load an exported catalog artifact through `triton.load`
on the target. Their NumPy scenarios check every output and read-only input;
MoE inputs cover stable ties and empty experts, and stateful Mamba programs check
two consecutive calls. The tolerance is the native worker's existing
`atol=1e-4, rtol=2e-3`; indices, Boolean results and unchanged inputs compare exactly.

The existing `helper_jit`, `helper_noinline`, `helper_loop_returns` and
`helper_context_return` runtime probes share their source across CPU targets.
They retain Canonical and target output (`x86ir`/`llir` or `riscvir`/C); the noinline
case requires a defined native noinline function and a call to it. The optional
`helper_descriptors` combines nested noinline helpers, fixed shaped tuple returns,
strided View descriptors, outer Level captures and ordered scratch/output effects
with a complete independent oracle and padding guards. `helper_context_return`
uses both one-block and multi-block Views to check that an early return skips
remaining Level instances. `helper_level_capture` passes a parent point and
already selected Views into a noinline helper, then uses public `level.subtiles`,
state/stage births and a nested helper; nonzero parent origins and tails retain
their ordinary semantics. A frontend rejection stays a diagnostic for that
source, rather than triggering an alternate algorithm. Configuring these probes
does not establish support until the selected compiler actually passes them.

`numerical_primitives` provides four directed scenarios through
`request.numerical_scenario`: `grouped_mac`, `sum_pairs`, `grouped_tail`, and
`lookup`. They call public `tl.mac_pairs/mac_groups`, `tl.sum_pairs` and `tl.lookup`
and retain the resulting Canonical/native output. Integer results use independent
exact oracles, including explicit accumulator wrap; the f32/group-three subcase
uses the existing runtime tolerance. A static block32 with N35 checks partial
groups and inactive pointer effects. Lookup uses only the supported
`bounds="in_bounds"` contract with unsigned active indices, plus table snapshot
ordering and View tail validity; it does not invent lookup mask/fill arguments or
send an active out-of-bounds index. The additional `scalar_validity` scenario
extracts View tail scalars, passes them through a noinline helper's if/for/while
returns and then performs caller for/while carries, runtime if merging, arithmetic,
stores and atomics. Its independent
histogram counts only valid inputs; no explicit effect mask hides validity loss.
These checks do not measure performance.

The optional `packed_layouts` runtime case selects `request.packed_scenario`:
`natural`, `layered_joined`, or `orders`. It uses public Encoding declarations and
independent serialized-byte fixtures for field reads/writes, a whole-record stage
snapshot, multiple records and a partial row group. The natural/layered cases use
the exported i4/u6 types; the order case uses a big-endian u16 and MSB-first u1
array. Existing GGML native and encoded Program entries can be selected alongside
these cases without changing their author source or oracle.

Framework and runtime probes are the migrated integration probes, not a separate
compiler. The target must provide NumPy and the selected Torch/Triton installation.
Missing framework capabilities are reported as `unsupported`. Hardware-inapplicable
examples may be explicit `not_applicable` entries with a concrete `reason`; they
are never counted as numerical passes.

Public product examples use the same `usage` worker. The default `contract: "use"`
keeps the original `use.py` execution/export/reload check. Additional contracts run
the real selected source file, observe its `main` outputs, and check complete
arrays rather than parsing the printed preview:

| Contract | Selection and evidence |
|---|---|
| `triton_basics` | `operation`: affine, softmax, matmul, scan or histogram; independent complete-output oracle |
| `autotune_export` | `size` and `threads`; selected tuned output, exported artifact and a reloaded call |
| `pytorch_cpu` | Hand-written kernel plus cached complete Inductor output and generated Weft code |

Each `usage` entry provides `source` and any `imports` package roots. For example,
`imports: {"source": "/work/Triton-Weft/third_party/weft/source"}` supplies the
product's author kernels. Imported files are copied into the run. The selected
target runtime must already be installed and provide the example's required
capabilities. New example semantics require an explicit observation contract;
successful process exit or a few printed values do not establish correctness.

`configs/verification-riscv.example.json` provides selectable runtime entries for
standard `tl.associative_scan` (both directions, several axes and tuple outputs),
atomic read/modify/write and compare/exchange, contended shared-address updates,
and helper calls with `noinline=True` or early returns from loops/level contexts.
Atomic checks validate the returned old values as well as final state. Contended
calls use permutations and a unique CAS winner, without assuming a thread order.
Use the configured legal CPU set and thread count; these are correctness calls,
not contention performance measurements. The installed compiler must implement
the selected feature; a configured case may still report a real compiler failure.

The runtime atomic probes retain a separate result and backend diagnostic for each
selected dtype/operation. They cover signed 32/64-bit and unsigned 32-bit RMW,
float16/32/64 add, and integer 16/32/64-bit CAS. A rejected type stays non-pass;
the presence of a verification case does not establish compiler support. Floating
add fixtures use exactly representable binary fractions, so no relaxed tolerance
is needed for narrower storage.

For host preparation followed by native execution, the same scan and atomic
semantics are registered as `pointer_scan_*` and `pointer_atomic_*` workloads.
They use the ordinary typed-ABI worker, including mutable `State` inputs and
reset outside timing. For example, an entry for the noncommutative reverse scan is:

```json
{
  "id": "native/scan_affine_reverse", "group": "native", "kind": "native",
  "case": "pointer_scan_affine_reverse",
  "source": "/work/Weft-Bench/src/weft_bench/verification/probes/riscv_kernels.py",
  "symbol": "scan_affine",
  "options": {
    "signature": {"X": "*i32", "Values": "*i32", "Y": "*i32", "Z": "*i32"},
    "constexprs": {"M": 4, "N": 8, "AXIS": 1, "REVERSE": true},
    "entry": "single"
  }
}
```

The source implementation supplies the host Triton Python and cross toolchain.
The affine tuple combiner is `(a,b)*(c,d)=(a*c,b*c+d)`; its reference evaluates
`flip(scan(flip(input)))`, so reversed operand order cannot hide behind a
commutative sum. `--prepare-only` emits the usual preparation manifest, and
`verification.prepared_entries` selects it for later native execution.

To consume a separately produced compilation batch without per-case scripts, set
`verification.compilations` to `{"results": "/work/compile/summary.json",
"bindings": "/work/compile/bindings.json", "root": "/work/compile"}`. Results are
a JSON array with unique `target`/`case`, `status` (`compiled` or a failure), and
`artifact` or `error`. Optional binding rows contain `target`, workload `case` and
`compiled_case`; several workloads can use one compiled artifact. Each invocation
reads the current manifest. Explicit per-operator artifact selections take precedence.

## Results and resuming

For a running queue, `verify --target cpu --output /work/maturity/cpu
--stop-after-current` requests a stop at the next case boundary. The active device
call finishes normally, and pending cases remain in the ledger for `--resume`.

`verification.json` is the current ledger; `failures.json` contains actionable
non-pass entries. Each case has a stable `cases/ID/current` directory containing
its request, outputs, logs and available diagnostics. Reruns retain at most the
first and latest failure directories plus a short status history. Superseded
successful artifacts are removed from this verification output.

The ledger reports `selection_counts` separately from counts for every retained
entry. Use selected counts when reporting a partial run. Native entries link
ordinary `results.json` files accepted by `explain` and `replay`; other entries
rerun through `verify --case ID --rerun all`. Triton replay uses saved artifacts
or the recorded Python/toolchain source route, not the legacy `--compiler` flag.
Historical flat source snapshots do not contain package-relative dependencies;
source replay of those kernels requires `--source` from a matching package checkout.

`--resume` continues pending/interrupted entries and skips unchanged completed
entries. It checks the plan, source/compiler file paths, sizes and modification
times, and the observed installed Triton runtime. It does not compute hashes.
Changed identities stop automatic resume; a changed environment during
execution marks an otherwise passing result `stale`. Use a stable installation for
the final complete run. Source overlays and installed wheels should have distinct
configuration labels and must not be described as one verified release.

When retaining earlier numerical evidence, keep its executed artifact and runtime
identity. An explicit byte-for-byte C/header comparison can associate a newly
compiled artifact with that evidence; record the comparison separately rather
than claiming the new binary was executed. Select changed cases for a rerun.

Compiler rejection, worker build failure, runtime failure, numerical failure,
unsupported coverage and hardware inapplicability remain distinct. Compilation
alone is never reported as numerical correctness, and correctness is not a claim
about performance.

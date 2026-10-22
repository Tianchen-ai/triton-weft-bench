# Environment and command recipes

These examples call the existing Weft Bench CLI through the plugin launcher.
Set `BENCH_SKILL` to the directory containing the loaded `SKILL.md`, `TARGET` to a
configured target name, and `RUNS` to the task's output directory. Weft Bench 0.3.0+
and Python 3.10+ are required. Use absolute source/artifact paths when switching
between Weft worktrees. Source files are host-side; model/library paths are target-side.

## Local installation and discovery

The plugin does not bundle the compiler, benchmark executable, GGML or models.
Install the bench from its checkout with `python3 -m pip install -e /path/to/weft_bench`
in the intended Python environment. A virtual environment is also supported.

Optional machine-only settings live at
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json`:

```json
{
  "command": ["/path/to/venv/bin/weft-bench"],
  "config": "/path/to/bench.local.json"
}
```

`command` is an argument array, never a shell expression. A Python module command
such as `["/path/to/python", "-m", "weft_bench.cli"]` is also accepted. Without local
settings, the launcher uses `weft-bench` from PATH and, when present,
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/bench.local.json`. Existing
`WEFT_BENCH_CONFIG` and explicit CLI `--config` still take priority.

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" list --kind inputs
python3 "$BENCH_SKILL/scripts/bench.py" show rms_norm --inputs decode
```

Target configuration selects compiler, RVV/VLEN/extensions, native toolchain,
GGML libraries, optional SSH deployment and normal/integrated llama.cpp builds.
Use the bench checkout's `configs/example.json` and `docs/targets.md` for initial
setup. Keep real settings outside the plugin and repository.

For heterogeneous RISC-V machines, keep one target per CPU class and first run
`environment --target TARGET --output DIR`. The observation reports actual VLEN
for every allowed CPU and the executing CPU's model. An explicit `vlen_bits`
must agree with all of them. `deployment.launch_prefix` is an argv wrapper run
before affinity and exec; use it for platform-required worker registration.
`deployment.ssh_options` applies to both SSH commands and transfers. Keep their
machine-specific values in the local configuration.

K3 A100 execution may require registration through `/proc/set_ai_thread` before
binding its CPU set. Its RVV leaf measurements remain distinct from the separate
IME2 backend. The matching SpacemiT GGML IQ2_S VLEN1024 reference needs the bundled
`native/patches/ggml-iq2-s-rvv1024.patch` (scale-nibble selection and safe sign
application). Preserve the original failed result, identify the correction in
`ggml.local_patches`, and establish a reference only after numerical checking.

## Operators and saved references

```sh
# Current source vs the target's GGML library, using an existing workload.
python3 "$BENCH_SKILL/scripts/bench.py" op rms_norm \
  --target "$TARGET" --weft /work/norm.py:my_norm --inputs decode \
  --output "$RUNS/norm-current"

# Use custom data or dimensions, retaining the same input for both implementations.
python3 "$BENCH_SKILL/scripts/bench.py" op rms_norm \
  --target "$TARGET" --weft /work/norm-artifact \
  --inputs /work/norm-inputs.json --save-outputs --output "$RUNS/norm-custom"

# Establish a reference once for this scenario.
python3 "$BENCH_SKILL/scripts/bench.py" op q4_k_dot \
  --target "$TARGET" --impl ggml --output "$RUNS/dot-ref"

# Reuse it. This does not re-time the reference implementation.
python3 "$BENCH_SKILL/scripts/bench.py" op q4_k_dot \
  --target "$TARGET" --weft /work/dot.py:kernel \
  --baseline "$RUNS/dot-ref" --output "$RUNS/dot-current"
```

Default short-call protocol: 3 warmups, 5 groups of approximately 10 ms, reused
inputs and one numerical check. `--cache cold` selects eviction before each call;
retain the same protocol when comparing history. Numerical oracle work is untimed.
Use `show` to obtain the actual port names, dtype, layout and shape before binding
a new kernel. Weft artifacts carry the ABI; no hand-built C benchmark is required.

External configurations can define `operator_suites` at the defaults or target
level, for example `{"operator_suites":{"focused":["q4_k_dot","f16_f32_projection_prefill"]}}`.
Members can specify dimensions with `{"case":"gemm_f32","shape":{"M":128,"N":256,"K":256}}`.
The listing shows these overrides; they apply to both implementations and replay.
Explicit input-file dimensions take precedence. A workload appears at most once
per suite, with one latest artifact directory. Use this form when a fixed baseline
has dimensions different from the workload defaults; do not relabel that baseline.
Discover the selected target's actual suites before using them:

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --environment "$BENCH_ENV" list --kind suites \
  --config /path/to/bench.local.json --target "$TARGET"
python3 "$BENCH_SKILL/scripts/bench.py" --environment "$BENCH_ENV" op --suite focused \
  --config /path/to/bench.local.json --target "$TARGET" --impl weft \
  --baseline "$RUNS/fixed-ref" --baseline-impl ggml --output "$RUNS/focused-current"
```

`BENCH_ENV` is a configured environment name from `--plugin-info`. Suite membership
selects workloads; it does not replace source/artifact bindings or change their
shape, precision or thread contracts. Keep the existing representative defaults;
a configured full suite is run only when requested.

## llama.cpp replacements

```sh
python3 "$BENCH_SKILL/scripts/bench.py" list --kind replacements
python3 "$BENCH_SKILL/scripts/bench.py" e2e llama2-7b-q4_0 \
  --target "$TARGET" --phase decode \
  --replace ggml.vec_dot/q4_0=/work/dot.py:kernel \
  --replace ggml.from_float/q8_0=/work/quant.py:kernel \
  --output "$RUNS/model-decode"
```

Repeat `--replace` for distinct roles or use `--replacement-file /work/replacements.json`:

```json
{
  "ggml.vec_dot/q4_0": "/work/dot.py:kernel",
  "ggml.from_float/q8_0": {"source": "/work/quant.py", "symbol": "kernel"}
}
```

The same role can have one implementation per invocation. A mapping creates a fresh
package containing exactly the requested roles. The tool delegates ABI validation,
packaging and native linking to Weft, then supplies the package to the integrated
llama.cpp build through `WEFT_GGML_PACKAGE`. Actual GGML dispatch determines whether
the hook is reached; repack and IME can take different paths from ordinary vec-dot.

Default phase is `both`: PP128/depth0 and TG32/depth512, each repeated 3 times.
`--scenario smoke` is a shorter integration check, not the default performance
comparison. `--prefill`, `--decode`, `--depth`, `--threads` and `--repetitions`
override the scenario. Keep prefill and decode as separate result rows.

`--impl normal` measures only normal llama.cpp without a replacement package.
Once its result is established, `--baseline MODEL_REF_RUN` measures only the
candidate by default. Read measured-window hit counts, not warmup totals. If the
binary lacks observers, the result reports `not_observed`. Configuring candidate
source/build/include lets the tool prepare observers and incrementally rebuild
the candidate harness. Source integration hooks are a prerequisite.

Configured build directories must provide `compile_commands.json` (enable
`CMAKE_EXPORT_COMPILE_COMMANDS`). Bench records actual CPU compiler commands and
rejects unoptimized builds before timing; a Release label alone is insufficient.
Model comparisons match the recorded compiler, optimization, ISA and floating-point
policy. A binary without a build directory reports unavailable inspection, rather
than claiming a verified build policy.

`prepare-model --target TARGET --replace ROLE=KERNEL --output DIR` prepares a
package without running a model. `--profile` is opt-in; use `hotspots --help` to
associate an existing profile with operator cases without re-running the model.
Perf collection uses 64 mmap pages by default. Configure `llama.perf_mmap_pages`
as a positive power of two if the target needs a different ring-buffer budget.
Check profile errors before interpreting an empty hotspot list.
The configured execution prefix and affinity wrap the profiled program inside
perf, so platform registration belongs to the process that runs the model.

For a numerical failure, use `validate-model MODEL --target TARGET --output DIR
--validation-reference REF_DIR` to check the configured package without repeating
model timing. `--check-dots` also compares each replaced vec-dot with its native
GGML function on the same live row, using the operator checker's tolerance.
This untimed interposer never replaces the candidate's output with the reference.
The first differing row is saved as `dot-inputs.json` and packed W/X buffers;
reproduce it with `op CASE --inputs DIR/dot-inputs.json --check-only`. Passing
per-row checks does not establish model correctness: downstream quantization can
amplify small floating-point differences, and the full logit check still applies.

## Compare and reproduce

```sh
python3 "$BENCH_SKILL/scripts/bench.py" compare "$RUNS/before" "$RUNS/after"
python3 "$BENCH_SKILL/scripts/bench.py" explain "$RUNS/after" --case q4_k_dot
python3 "$BENCH_SKILL/scripts/bench.py" replay "$RUNS/after" --case q4_k_dot \
  --compiler /build/new/weft-compile --output "$RUNS/replay"
python3 "$BENCH_SKILL/scripts/bench.py" check --target "$TARGET" --case q4_k_dot
```

`check` requires explicitly configured `check.reference` and `check.previous`.
Do not select a historical best result as the checkpoint. Failed or incompatible
rows remain results; inspect their diagnostics/scope before attributing a change
to the compiler. A replay uses saved inputs and compilation material; `--source`
selects current author source instead of the saved canonical IR when intended.

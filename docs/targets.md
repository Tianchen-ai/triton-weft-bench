# Environment setup

The benchmark core has no SSH or machine-name branches. CLI/Python convenience
calls use a selected external target for building and deploying a native worker.
Run `op` and `e2e` locally or select a configured SSH target without changing
operation, input or replacement definitions.

`bench.local.json` contains `defaults` and `targets`; see `configs/example.json`.

| Field | Meaning |
|---|---|
| `weft_root`, `compiler`, `weft_python` | Host-side legacy Weft source/compiler/Python |
| `implementations` | Selected adapters, source/artifact bindings and compiler options |
| `triton_runtime` | Execution-side exported Triton–Weft native SDK include/library paths |
| `target` | Target architecture, march, and architecture-specific ABI/VLEN |
| `toolchain` | Execution-side C/C++ commands, flags and libraries |
| `deployment` | Optional SSH alias, workspace, CPU affinity, NUMA policy and target Python |
| `ggml` | Target-side llama.cpp source, GGML libraries and optional compile database |
| `bindings` / `bindings_file` | Optional preset Weft source/symbol/physical options |
| `capabilities` | Explicit extension availability, such as spacemit-ime1 |
| `llama` | Model paths, normal/integrated builds, threads and optional profiles |
| `check` | Optional historical reference/checkpoint convenience |
| `references` | Named OpenBLAS/oneDNN include/shared-library paths and build provenance |
| `default_implementations` | Optional `op`/`e2e` implementation lists, including reference-only defaults |
| `record_environment`, `output_root` | Native CPU/OS observations and the default output directory |

Source paths for Weft kernels are host-side. GGML libraries, model weights,
llama.cpp binaries and toolchain commands are execution-side. Credentials stay in
the existing SSH environment. Configuration files and models are not bundled.

`deployment.ssh_options` is an optional argv list passed to SSH for commands and
file transfers, for example a task-scoped control socket. `deployment.launch_prefix`
is an optional argv list wrapping execution before CPU affinity and NUMA placement;
it does not wrap compilation or file transfers. A platform that requires thread
registration must perform it here and then `exec` the supplied command. This lets
the new process and its dynamic loader start on the selected CPU class. Keep
machine paths and registration policy in the external target configuration.

SSH command deadlines run on the target through `timeout`, which terminates
the command's process group and escalates after five seconds. The SSH transport
has an additional fifteen-second allowance for cleanup. A deadline is a failed
run; it must not leave a model worker overlapping the next measurement. Remote
Linux targets need coreutils `timeout` with `--signal` and `--kill-after` support.
Set the configured deadline for the full command,
including model loading, warmup and every timing repetition.

The native environment probe records the current CPU's model and ISA, and on
RISC-V reads actual `vlenb` on every allowed CPU after checking Linux vector state.
It restores affinity and reports per-CPU widths; heterogeneous widths must not be
represented as one machine-wide VLEN. Feature observation does not prove that an
IME or other optional instruction has executed.

For the SpacemiT llama.cpp reference at revision `6df9f75`, the VL1024 IQ2_S
dot leaf needs the bundled
`weft_bench/native/patches/ggml-iq2-s-rvv1024.patch`: its original nibble merge
selects the opposite scale, and applying the sign to Q8 bytes can overflow at
`-128`. Apply the patch to a separate reference checkout, rebuild GGML, and
record both `ggml.source_revision` and `ggml.local_patches` in the target
configuration. The corrected leaf still uses the vendor RVV implementation;
keep its numerical failure and the corrected reference distinguishable.

SpacemiT IME assembly can use upstream Clang with a vendor GNU assembler. Apply
`native/patches/ggml-spacemit-external-assembler.patch` to both reference and
candidate sources, and use `-fno-integrated-as -include
/path/to/weft_bench/native/spacemit_external_assembler.h`. The patch leaves the
IME instruction probes enabled; the external assembler must pass them. The
header restores the IME extension after Clang's assembly architecture attribute.
Declare any other required extensions in the compiler flags. Keep SPERT and the
vendor backend enabled when comparing against that backend; a CPU/RVV build is
a separate reference, not evidence of the vendor's full execution path.

The standalone SpacemiT backend also requires `GGML_SPACEMIT_BACKEND=1` at
runtime. Its library can report IME2 capability and appear in the compiled
backend list while exposing zero devices with that switch unset. Configure the
switch, fusion policy, worker count and preferred CPU list identically for normal
and candidate models. These controls are cleared from the inherited environment
before applying the model configuration, and are part of timing/reference
comparison and fixed-logit reference identities. Confirm actual tensor placement
and execution; build flags and initialization messages alone do not prove use.

For the three-operand IME1 form, Clang's integrated assembler can instead use
`-Xclang -target-feature -Xclang +xsmtvdot -include
/path/to/weft_bench/native/spacemit_ime1_llvm.h`. The header maps the legacy
`vmadot` spelling to `smt.vmadot`; it does not translate IME2's operand contract.
Use the same choice for reference and candidate builds and verify the native
instruction probe. Match the OpenMP library to the compiler: a Clang 23 build
can require `__kmpc_dispatch_deinit`, absent from older runtime bundles.

Model observers inspect the configured integration SDK's counter declarations.
Row and IME counters are required; AMX counters are emitted only when that SDK
provides them. An unavailable counter is recorded as unavailable, not as zero
calls. The instrumented llama-bench target needs both the integration include
directory and `triton_runtime.include`, since `weft/ggml.h` includes the native
runtime contract. Model validation uses the same declared counter capabilities.

Single-instance leaves use one CPU thread; complete GGML GEMM/projection and CPU
library matmuls accept explicit thread limits. Direct Norm/RoPE/softmax and serial
dot workloads reject a misleading multi-thread selection. Explicit grid/JIT calls
and models record their own thread counts. The legacy adapter builds generated C with
the configured native toolchain. Triton–Weft AOT consumes the exported native
library directly. Native timing needs the SDK; Python JIT and framework verification
also need a target-side Python installation.

For precomputed RoPE, provide the normal GGML build's `compile_commands.json`
(`CMAKE_EXPORT_COMPILE_COMMANDS=ON`). Its original `ops.cpp` flags compile a thin
bridge to `rotate_pairs<float>`. RMSNorm calls the existing fused multiply path;
LayerNorm times norm+mul+add; Softmax uses its native execution function; projection
uses a prepared normal GGML graph. GGML dequantization's existing decoder check is
explicitly labeled library-reference self-consistency; other workload checks use
their declared mathematical/reference oracle. See [x86 references](x86-references.md)
for the library, quantization storage and execution-evidence contracts.

## Triton–Weft candidates

Select `weft_bench.adapters.triton_native:TritonAdapter` as the implementation
`factory`. The usual implementation name `weft` may select either this adapter or
the legacy one; it is a configured label, not automatic compiler detection.

```json
{
  "implementations": {
    "weft": {
      "factory": "weft_bench.adapters.triton_native:TritonAdapter",
      "bindings": {"q4_k_dot": {"artifact": "/work/artifacts/q4_k_dot"}}
    }
  },
  "triton_runtime": {
    "include": "/opt/triton-weft/native/include",
    "library": "/opt/triton-weft/native/lib/libweft_runtime.so"
  }
}
```

This fragment belongs inside a target with its `target`, worker `toolchain`, GGML
and deployment settings. Artifacts are host-side; SDK paths are execution-side.
For source compilation, replace `artifact` with `source` and `symbol`, and provide
the selected host `python`, optional `pythonpath`, `signature`, `constexprs`,
`backend_options` and compiler `toolchain`. An empty `signature` uses typed source
annotations. Bench calls Triton compilation/export and loads the native artifact;
it does not compile the exported candidate C again.

`entry: "single"` uses the single-instance ABI. `entry: "grid"` additionally
requires `grid` and `num_cpu_threads`, and is recorded as a different timing scope.
`weft_bench.adapters.triton_jit:TritonJitAdapter` selects target-side Python JIT;
its call overhead is reported separately from native operator timing.

## llama.cpp setup

```json
{
  "llama": {
    "threads": 8,
    "models": {"llama2-7b-q4_0": "/models/llama-2-7b.Q4_0.gguf"},
    "implementations": {
      "normal": {"binary": "/llama-normal/build/bin/llama-bench"},
      "weft": {
        "binary": "/llama-weft/build/bin/llama-bench",
        "source": "/llama-weft",
        "build_directory": "/llama-weft/build",
        "weft_include": "/sdk/share/weft/llama.cpp/include",
        "env": {"LD_LIBRARY_PATH": "/sdk/lib"}
      }
    }
  }
}
```

The normal and integrated builds use matching normal backend settings. The candidate
tree is prepared with Weft's public llama.cpp integration once. Its existing hooks
accept parameter-selected packages; changing a kernel or replacement mapping does
not require patching the hooks again.

For a configured `build_directory`, enable `CMAKE_EXPORT_COMPILE_COMMANDS`.
Before timing, Bench reads the actual GGML backend compilation commands, rejects
unoptimized translation units, and saves them in `build-info.json`. Compiler version,
optimization, target and floating-point flags form the model comparison policy.
A `Release` label alone does not establish these flags. `compile_commands` can name
an explicit database path. Binary-only configurations report unavailable build
inspection and cannot match a reference with an observed compilation policy.
The logit check follows `llama.args` layer-offload selection (`-ngl` or
`--n-gpu-layers`, default `-1`) as well as the model thread count. It does not force
CPU-only numerical execution while timing an accelerator. Offload selection is
part of the saved numerical-reference contract.

When source/build/include are provided, the tool installs measurement-window
observers into that candidate benchmark harness and invokes an incremental
`cmake --build ... --target llama-bench`. This affects benchmark instrumentation,
not GGML kernel arithmetic. Supply a separate candidate source tree.

For legacy replacement package linking, `deployment.python` must provide Weft's public
runtime, or `deployment.weft_root` can select an existing execution-side checkout.
The tool calls `Package.create` on the host and `Package.link` on the target;
it does not invent a second package format or linker.

For Triton–Weft, set `llama.package_backend` to `triton` and configure
`llama.triton.python` plus the source role bindings if compiling replacements.
The existing role mappings call `triton.Package.create` with native artifacts;
this route does not use the legacy target-side linker. Both providers require a
compatible integrated llama.cpp build and report measured-window replacement hits.

Preset replacement groups:
- `ggml.vec_dot`: q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K.
- `ggml.from_float`: q8_0, q8_K.
- `ggml.ime1`: q4_centered_prefill, q4_affine_prefill.

IME requires an eligible target and explicit compiler extension options. It remains
a distinct packed call boundary. Replacing an ordinary dot does not automatically
replace an IME model path.

Saved profiles and optional `call_sites` mappings remain supported. They associate
model functions with workloads; they do not choose replacements automatically.
Profiling uses 64 mmap pages per perf ring by default. `llama.perf_mmap_pages`
can select another positive power of two when the target needs a different buffer
budget. This keeps collection independent of perf's machine-dependent default;
the selected event, frequency and buffer size are recorded with the result.
Execution registration, affinity and NUMA policy are applied inside perf, to the
profiled program. A per-process platform registration on the outer profiler does
not establish the child program's execution capabilities.

`validate-model MODEL --target TARGET --output DIR --validation-reference REF_DIR`
checks the configured package's full-vocabulary logits without model timing.
`--check-dots` (or `llama.validation.check_dots: true`) additionally compares live
replaced rows with the original GGML dot in the validation process. It requires
the integrated implementation's `weft_include`, shared GGML libraries and an unused
`LD_PRELOAD` setting. Its calls and overhead never enter model performance results.
On the first operator mismatch, it saves packed inputs, both outputs and an
available dequantized-dot diagnostic. `dot-inputs.json` can be passed directly to
the corresponding `op --check-only` invocation. Successful row checks record their
actual call count; the independent full-logit tolerance is still enforced.

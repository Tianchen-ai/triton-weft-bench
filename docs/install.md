# Install and choose an environment

Bench is a Python CLI/library. Install it in a virtual environment:

```sh
git clone https://github.com/Tianchen-ai/Weft-Bench.git
cd Weft-Bench
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/weft-bench list
```

Python 3.10 or newer is required. `list` and command help need no device connection
or compiler. Use `.venv/bin/weft-bench` directly, or activate that environment.

## Select the dependencies for the requested work

| Workflow | Host | Execution machine |
|---|---|---|
| GGML only | Bench, external configuration | C/C++ toolchain, GGML source/headers and built libraries |
| Triton–Weft source | Compatible installed Triton–Weft Python and configured native/cross toolchain | Exported library, native Weft SDK and worker toolchain |
| Triton–Weft AOT | Exported `manifest.json`, `triton.json`, `kernel.so` and accompanying files | Matching native SDK and worker toolchain |
| Python JIT / `verify` runtime | Bench and selected case definitions | Installed Triton–Weft, NumPy and Torch for the configured probes |
| llama.cpp | Selected package provider and replacement bindings | Normal/integrated llama.cpp builds, model and native SDK |

Install compiler wheels that match each machine's architecture and Python ABI.
An x86 host compiler and a RISC-V runtime may use different Python versions and
need separate wheels. Torch wheels must also match the target's libc requirement.
Keep development/compiler environments separate from shared system packages.
Use the compiler's supported PyTorch integration installer in the selected private
Torch environment; Bench does not patch framework or compiler implementations.

For the legacy Weft compiler, configure its existing compile/runtime paths instead.
It remains a separate implementation adapter, with the same workload and input API.

## Configure and make the first comparison

Keep configuration outside the checkout. Start with `configs/example.json` for the
legacy adapter; use the [Triton–Weft target fields](targets.md#tritonweft-candidates)
for source or AOT candidates. Replace example paths with your own compiler, library
and model paths, then select the file explicitly:

```sh
export WEFT_BENCH_CONFIG=/work/bench.local.json
weft-bench show q4_k_dot
weft-bench op q4_k_dot --impl ggml --target cpu --output /work/results/ref
weft-bench op q4_k_dot --target cpu --baseline /work/results/ref \
  --output /work/results/current
```

Here `cpu` is a target name from your configuration. The second invocation reuses
the matching saved reference and measures the candidate. Shape, input, cache and
call boundary remain part of that comparison; mismatches are shown explicitly.

For numerical/runtime validation, copy the entries from
`configs/verification-riscv.example.json` into your own target configuration and
set its Python/toolchain and legal CPU affinity. Inspect before selecting:

```sh
weft-bench verify --config /work/bench.local.json --target cpu --list
weft-bench verify --config /work/bench.local.json --target cpu \
  --case riscv/associative_scan --output /work/verification/cpu
```

These checks require a compiler implementing the selected features; the existence
of a Bench case is not a claim that every installed compiler version supports it.
See [verification](verification.md) for resuming and diagnosing selected failures.
See [Codex plugin setup](codex-plugin.md) to expose the same CLI to other agents.

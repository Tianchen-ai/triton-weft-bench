# Inputs and implementations

The shipped product compares Triton–Weft and Weft with GGML for predefined computations.
A new implementation of one of these computations requires a kernel binding, not
a new input generator, timer, remote launcher or result parser.

## Python interface

```python
from weft_bench import compare, GGML, Inputs, WeftKernel
from weft_bench.loading import load_config, select_target

target = select_target(load_config("/work/bench.local.json"), "rvv128")
rows = compare(
    "rms_norm",
    candidate=WeftKernel(
        source="/work/norm.py", symbol="my_norm",
        options={"lmul_eighths": 16},
        arguments={"Input": "X", "Scale": "Weight", "Output": "Y"},
    ),
    baseline=GGML("/device/llama.cpp", "/device/llama.cpp/build/bin"),
    inputs=Inputs(shape={"M": 2, "N": 64}, parameters={"epsilon": 1e-6}),
    target=target,
    save_outputs=True,
    output="/work/run/artifacts",
)
```

Argument aliases are only needed if the kernel's argument names differ from the
workload names. The artifact manifest determines argument order, runtime dimensions
and stride slots. The binding checks view shapes, encodings and access modes before
invocation. A kernel whose fixed dimension differs from the chosen input must be
compiled with matching options.

Workload ports may be `input`, `output` or `inout`. An `inout` port accepts a custom
initial buffer and requires the artifact's `readwrite` ABI access. Stateful workloads
declare a reset outside timing and check both the mutated state and returned outputs;
atomic workloads use this path. Implementations do not create their own timer.

Use `WeftKernel(artifact="/work/artifact")` to measure an existing artifact.
`baseline="onednn"` or another configured implementation name selects a native
reference through the same Python comparison API. Reference-only callers use
`benchmark(case, ["onednn"], target, ...)` without creating a compiler candidate.
Use `compiler="/build/new/weft-compile"` to select another compiler build.
Compilation, JIT, autotuning and artifact caching remain Weft's public runtime's job.

## Custom inputs

Python arrays, raw files and JSON use the same input selection:

```python
inputs = Inputs(
    shape={"M": 2, "N": 64},
    buffers={"X": [0.25] * 128, "Weight": [1.0] * 64},
    parameters={"epsilon": 1e-6},
)
```

Equivalent JSON:

```json
{
  "shape": {"M": 2, "N": 64},
  "seed": 410,
  "parameters": {"epsilon": 0.000001},
  "buffers": {"X": "x.npy", "Weight": "weight.bin"}
}
```

Paths in JSON are relative to the JSON file. Omitted buffers use the workload's
generator. Lists contain numeric values in the declared dtype. NumPy arrays/files must match the
declared floating dtype and shape, or contain packed bytes for encoded inputs;
raw bytes/files carry the storage shown by `show OP`.
BF16 NumPy arrays use uint16 bit storage; BF16 lists are rounded to stored BF16
with round-to-nearest-even before execution.
Encoded dot/dequantization weights use GGML block bytes, not unquantized floats.

Packed IME presets accept `W_f32` and `X_f32` as custom logical source tensors;
packing and activation quantization occur once before leaf timing on both sides.
Projection accepts encoded W and float32 X; its required activation quantization
occurs inside the timed call. Input generators, buffers and kernel implementations
are independently selectable.

Unknown roles, incompatible byte counts or incompatible artifact IO produce an
explicit preparation error. Output files are optional; their tensor descriptors
remain present in every result.

## Model interface

```python
from weft_bench import compare_model, LlamaCpp, ModelInputs, WeftKernel

candidate = LlamaCpp(
    "/device/integrated/bin/llama-bench",
    env={"LD_LIBRARY_PATH": "/device/sdk/lib"},
).replace({
    "ggml.vec_dot/q4_K": WeftKernel(source="/work/dot.py", symbol="my_dot"),
})
rows = compare_model(
    "deepseek8b-q4_k_m",
    candidate=candidate,
    inputs=ModelInputs(prefill=128, decode=32, depth=512, threads=8),
    target=target,
)
```

The mapping uses Weft's existing GGML roles. Package creation validates storage and
ABI contracts; linking uses the target's native toolchain. The integrated llama.cpp
build retains normal backend selection. Counters distinguish a loaded replacement
from one actually called during the measured phase.

## Internal extension boundary

`core.py` defines a native workload and native call, and composes the shared worker.
`workloads.py` supplies the fixed GGML computation/IO definitions and their oracles.
`inputs.py` owns input selection and snapshots. `libraries.py` binds Weft ABI and
GGML calls. `runner.py` builds and executes the composed worker through the outer
transport in `execution.py`.

An implementation contributes preparation plus invocation, never a standalone
benchmark program. New built-in workloads add their semantic inputs/oracle and
GGML binding. Native measurement, CSV/JSON formatting and SSH do not change.

Version 0.2 whole-command adapters remain isolated in `compat.py` for existing
clients. They are not the interface for adding a Weft kernel to the product.

## Native library or external implementation adapters

Configure an implementation with `kind: "native"` and a `module:factory` factory.
The factory returns an object exposing
`prepare_native(case, target, options, directory, executor, workload, name)`.
Return `weft_bench.core.PreparedNative` containing the `NativeCall`, include/link
flags, optional environment and result details. The shared worker continues to
own input preparation, output checking, state reset, timing and result parsing.
There is no new benchmark executable protocol to implement.

Read the workload's named `buffers` and `scalars` rather than inventing new inputs.
Preparation may allocate library handles/workspaces; the call's `invoke` contains
the complete declared computation. `collect` runs once outside sampling and can
emit `WB_DETAILS` with the implementation name and actual dispatch evidence.
Every other field belongs to that implementation's result. A different boundary
must supply an explicit `timing_scope` detail. The new OpenBLAS/oneDNN adapters
are examples of this interface; an external factory needs no runner changes.

# TileLang–Weft source JIT and native deployment

[中文](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/tilelang-weft-integration/)

Install TileLang–Weft, its matching `weft-core` and this Bench wheel in the target
environment. Select `weft_bench.adapters.triton_jit:TileLangJitAdapter` for ordinary
TileLang source JIT. It uses the same fixture, inputs, numerical checks, native
timer, SSH execution and result storage as the Triton adapter.

An implementation uses the existing source configuration:

```json
{
  "factory": "weft_bench.adapters.triton_jit:TileLangJitAdapter",
  "python": "/path/to/environment/bin/python",
  "source": "/path/to/tilelang/weft/kernels/activation.py",
  "symbol": "affine",
  "constexprs": {"N": 4096, "block": 128, "cpu_num_threads": 1},
  "grid": [32],
  "num_cpu_threads": 1,
  "backend_options": {}
}
```

Use `show affine --target TARGET` to inspect its workload. Then use
`op affine --impl tilelang --check-only --target TARGET --output OUTPUT` with
the existing local target configuration. Remove `--check-only` for measurement.
The worker reports first compilation, hot Python/grid calls, disk-cache restore
and exported native execution separately. The actual source grid is checked
against the measurement contract; tuning records the selected source grid.

For eager source functions, Tensor metadata is taken from the workload ports;
an explicit `signature` can provide shape/dtype/strides. Packed raw storage uses
the source's Encoding declaration, while its byte tensor geometry stays separate
from the logical encoded View. Runtime expressions in buffer dimensions are
evaluated by TileLang's existing expression binder. Multioutput, readwrite and
workspace parameters use the existing workload and `workspaces` configuration.

For deployment, select
`weft_bench.adapters.triton_native:TileLangArtifactAdapter`. A `source` is compiled
through TileLang's native source/AOT API; `artifact` loads an already exported
library. These are recorded as native deployment, separately from source JIT.

llama.cpp configurations select `producer: "tilelang"` and a `tilelang` compiler
configuration containing the existing `python`, `compile_target`, `toolchain`
and per-role `bindings`. `prepare-model` compiles before inference and packages
artifacts through the shared role validator. `validate-model` and `e2e` reuse
the existing runners and observed-hit checking. Use the `deployment.vec_dot`
programs for actual GGML row ABIs; a matrix-shaped dot has a different entry
contract and is not substituted as a vec-dot.

Product sources remain in TileLang–Weft. Bench owns comparison configuration
and outputs under its stable `output/` directory. Comparative TileLang
performance remains **Under evaluation / 测试中**.

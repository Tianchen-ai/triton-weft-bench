# Mainline and benchmark boundary

The upcoming embedded Triton route has a separate
[integration contract](triton-weft-integration.md). It is preparation for a real
native artifact/launcher interface; the current compiler path remains active.

The source extraction preserves algorithm definitions and public runtime ownership.

| Existing asset | Current owner |
|---|---|
| Weft compiler, passes, extensions, JIT, autotune, artifacts/packages | Weft mainline |
| Author kernels, format helpers, example algorithms | Weft mainline, selected by external source paths |
| Native SDK, Torch and JIT correctness repros, including IME JIT callbacks | Weft mainline |
| `examples/run` operator runners, configuration lookup and shell launchers | Replaced by `weft-bench op` and external bindings |
| Dot/quantization/dequantization/projection timing drivers | Replaced by workload inputs, native calls and one shared worker |
| Model runner and perf-window symbol aggregation | `weft-bench e2e`, optional `--profile` |
| Earlier performance CSVs, narratives and obsolete driver variants | Local ignored `history/`, preserved before source removal |
| Product roadmap and formal compiler contracts | Weft mainline |

The benchmark's standalone per-operation programs are removed. Their data/oracle
logic now lives in workload input fixtures; library providers contribute calls;
the core owns one measurement loop. The IME callback validation in mainline retains
its separate functional purpose. Runtime deployment/tuning APIs do not import bench.

Version 0.3 separates `Case` semantics from source/build fields, exposes `Inputs`,
`WeftKernel`, `GGML`, `ModelInputs` and `LlamaCpp`, and adds parameter-selected model
package preparation. `runner.py`/`execution.py` own compilation and deployment;
`check.py` and saved-result comparison remain outer workflows. Native measurement
does not contain SSH, host names, compiler selection or historical checkpoint logic.

Older whole-command adapters are isolated in `compat.py`. Old saved case dictionaries
are accepted when replaying. The new shared-input protocol is explicit; historical
timings are not silently renamed to the new input/worker protocol. A fresh reference
for that protocol can be measured once and reused.

Historical material is byte-checked during migration and kept locally, not bundled
in wheels or included in version control. Large experiment directories and machine
configuration are not copied wholesale. Current runs are separate from these old
records. No old performance table is overwritten by the tool's acceptance runs.

TileLangBench is downloaded under ignored `ref/` for source reference. Its compact
case scripts informed this tool; neither TileLangBench nor TileLang is a dependency.

The workload / implementation / input separation also follows the organization of
[Tritonbench](https://github.com/meta-pytorch/tritonbench/blob/main/tritonbench/utils/triton_op.py).
We keep the execution layer small for native CPU/RVV; tuning, backend lowering and
package selection stay in Weft. No benchmark framework is a runtime dependency.

Historical import explicitly maps `quantized_vec_dot_*` rows to `*_dot_rows`, keeping
the old cache policy, shape, native CPU, Weft time and ref time. IME packed leaves,
full projections and model stages remain separate scopes. Records that lack a known
equivalent boundary are retained with reasons, not silently dropped or converted to
new-protocol results. `checkpoint` assembles user-selected observations without
running either implementation or choosing historical maxima.

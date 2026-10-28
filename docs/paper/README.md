# Paper kernel and model data

[中文](README.zh-CN.md) · [Reproduction workflow](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)

These tables accompany the overall-performance figures in the Weft paper V10.
They retain values from its frozen `data/operators.csv` and `data/models.csv`,
with machine-local paths omitted.

- [operators.csv](operators.csv): 553 operator–platform rows, including 543 matched native comparisons.
- [models.csv](models.csv): 18 phases from nine paired model configurations.
- [operators.png](operators.png) and [models.png](models.png): the paper's original
  Figure 8 and Figure 10 PNG exports, embedded directly in the repository README.

## Read the columns

`case`, `shape`, `target` and `scope` identify the workload.
`median_ms` and `baseline_median_ms` are the matched native/model times;
`speedup` is reference divided by candidate. `correctness` states the actual
check scope. Warmup, sampling, cache and protocol columns preserve the setting.

JIT columns separately record first compilation/call, warm Python invocation and
disk-cache reuse. Model rows include normal-backend policy and observed replacement
counters. Matched operator references are 541 GGML and two oneDNN cases;
K3 model CPU–RVV and vendor IME2 policies are separate.

## Reproduce

Install Triton–Weft and this Bench package, configure the target and references,
inspect `show CASE --target TARGET`, then use `op`, `prepare-model` and `e2e`.
The [complete guide](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)
provides commands.

V10's product/Bench source snapshot is `369c22929` / `e17240a`.
The aggregate combines installed-JIT coverage and affected-case repairs;
each original observation retains its own source/settings.
Current catalogs provide author-program entries, shapes, parameters and reference
providers for new runs on the selected installation and machine.

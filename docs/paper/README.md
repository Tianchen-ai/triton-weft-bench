# Final paper data

[中文](README.zh-CN.md) · [Reproduction workflow](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)

Fourteen CSVs preserve the final paper measurements. Figures use the approved
vector SVGs. [Provenance](provenance.json) records the paper revision, hashes,
row counts and thread settings.

| Table | Rows | Evidence |
|---|---:|---|
| [operators.csv](operators.csv) | 553 | All operator/target cases and matched times |
| [operator-families.csv](operator-families.csv) | 5 | Family counts and geometric means |
| [applications.csv](applications.csv) | 18 | Complete inference phases and reference policies |
| [application-x86-windows.csv](application-x86-windows.csv) | 120 | x86 per-process stage observations |
| [application-roles.csv](application-roles.csv) | 18 | Computation and preparation role effects |
| [fp32-libraries.csv](fp32-libraries.csv) | 12 | FP32 GEMM against three libraries |
| [grid.csv](grid.csv) | 48 | Complete calls on 1/2/4/8 threads |
| [ggml-compute.csv](ggml-compute.csv) | 32 | Computation inside GGML workers |
| [author-controls.csv](author-controls.csv) | 240 | Five-route sharing and precision controls |
| [author-organization.csv](author-organization.csv) | 48 | Arithmetic and output-state scope |
| [five-route-bindings.csv](five-route-bindings.csv) | 30 | Selected task bindings |
| [format-addition.csv](format-addition.csv) | 5 | Q4 construction and Q6 source increments |
| [state-algorithms.csv](state-algorithms.csv) | 36 | Attention and Welford organizations |
| [compiler-mechanisms.csv](compiler-mechanisms.csv) | 214 | Paired physical mechanisms |

## Read the data

In `operators.csv`, `weft_ms` and `reference_ms` are matching native call times;
`speedup` divides reference by Weft. Shape, scope and correctness identify the
workload and check. There are 553 cases and 543 matched references.

`applications.csv` contains complete stages for eight model/platform deployments
and the additional K3 CPU-vector reference, totaling 18 rows. SG2044 uses
64 threads and the other targets use eight. `aggregation` identifies process
aggregation; the separate x86 table gives individual stage windows.

Control `s0/s1` selects local/shared preparation and `g0/g1` direct/grouped
accumulation. Compiler studies fix the author program and change physical
mechanisms. `disabled_over_enabled` and sample columns preserve paired values.

## Figures and reproduction

[![Representative kernels](operators-heatmap.svg)](operators-heatmap.svg)

[![Every matched distribution](operators-ecdf.svg)](operators-ecdf.svg)

[![Complete inference](models.svg)](models.svg)

Install [Triton–Weft](https://github.com/Tianchen-ai/triton-weft) and Bench,
configure references and targets, inspect `show`, then run `op`,
`prepare-model` and `e2e`. The [complete bilingual guide](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)
provides commands. The product supplies normal kernels; Bench owns inputs,
comparisons and measurement.

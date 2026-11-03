# Paper results and data

[中文](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/paper/) · [Complete figure gallery](gallery.md) · [Reproduction workflow](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)

The [complete gallery](gallery.md) presents all paper result figures (Figures
9–20): operator/application performance, author controls and reuse, and compiler
mechanisms. The tables below preserve the final measurements and source counts.
[Provenance](provenance.json) records the paper revision, hashes, row counts,
thread settings and the source of every vector panel.

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
| [increment-costs.csv](increment-costs.csv) | 3 | Figure 17: added and reused SLOC per format |
| [increment-definitions.csv](increment-definitions.csv) | 18 | Figure 17: individual source definitions and SLOC |

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

The **[complete figure gallery](gallery.md)** contains 23 SVG panels across all
12 result figures. Each figure has an explanation, timing boundary and data links.
The two `increment-*.csv` files retain the source-definition snapshot used for
Figure 17; identifiers name the measured definitions before their library move.
`first` is dequantization source and `second` is additional dot source; their
sum is the total source added for both operators. `format-addition.csv` measures
the separate five-route Q4_K/Q6_K task in tokens.

Install [Triton–Weft](https://github.com/Tianchen-ai/triton-weft) and Bench,
configure references and targets, inspect `show`, then run `op`,
`prepare-model` and `e2e`. The [complete bilingual guide](https://tianchen-ai.github.io/triton-weft/experiments/reproduce/)
provides commands. The product supplies normal kernels; Bench owns inputs,
comparisons and measurement.

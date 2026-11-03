# Triton–Weft Bench 1.0

[中文](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/)

**From a Weft program to a matched CPU experiment.**

Weft lets authors compose representation, arithmetic, sharing and state. Bench
connects those programs to matching inputs, optimized references, numerical
checks and native timing. The paper follows the result from individual kernels
to complete llama.cpp stages, then explains it through author controls, reuse
and the compiler's use of source relationships.

[![All 543 matched operator comparisons](paper/operators-ecdf.svg)](paper/operators-ecdf.svg)

Across SG2044, K1, K3 and Xeon 8470Q, geometric-mean speedups over optimized
references are **1.60×, 1.54×, 1.83× and 1.32×**, respectively.
**[Explore every paper result figure](paper/gallery.md)**: hardware execution,
FP32 libraries, thread scaling, complete inference, source controls, format
increments and compiler mechanisms. The [data guide](paper/README.md) links
their matching measurements and source counts.

## Run your own comparison

Install [Triton–Weft](https://tianchen-ai.github.io/triton-weft/getting-started/installation/)
and Bench, configure a CPU and reference library, then select a workload and kernel.
The guides below explain the complete path. The
[Triton–Weft handbook](https://tianchen-ai.github.io/triton-weft/) introduces the
programming model and installed kernel library.

| Task | Guide |
|---|---|
| Install the product and Bench | [Installation](install.md) |
| Make a matched comparison | [Quickstart](quickstart.md) |
| Choose the kernel, data and library independently | [Inputs and implementations](extending.md) |
| Configure native or SSH targets | [Targets](targets.md) |
| Follow normal kernel ownership | [Product and Bench](migration.md) |
| Inspect registered selections | [Catalogs](catalogs.md) |
| Use JIT, AOT, grids and models | [Product integration](triton-weft-integration.md) |
| Establish optimized native references | [x86 references](x86-references.md) |
| Check selected semantics and deployments | [Verification](verification.md) |
| Invoke the CLI from Codex | [Plugin](codex-plugin.md) |
| Explore every paper result figure | [Complete gallery](paper/gallery.md) |
| Inspect measurements and source counts | [Paper tables](paper/README.md) |

The product implements the model, compiler, runtime and normal kernel library.
Bench supplies shared inputs, reference calls, numerical checks and measurement.

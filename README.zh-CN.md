# Triton–Weft Bench 1.0

[English](README.md) · [下载 1.0](https://github.com/Tianchen-ai/triton-weft-bench/releases/tag/v1.0.0) · [Bench 网站](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/) · [全部论文图](https://tianchen-ai.github.io/triton-weft-bench/zh-CN/paper/gallery/) · [结果表](docs/paper/README.zh-CN.md) · [Triton–Weft](https://github.com/Tianchen-ai/triton-weft)

**从 Weft 程序，走到条件匹配的 CPU 实验。**

Weft 让作者控制表示、算术、共享和状态。Bench 将这些程序接到具体工作负载，
统一输入、优化参考、数值检查与原生计时器。选择产品内核或自己的实现，
即可比较完整算子，或测量它在 llama.cpp 中的作用。

论文结果沿着三个尺度解释这一过程：算子与应用的性能、可组合作者选择的作用，
以及实现这些选择的编译机制。**[完整结果图集](docs/paper/gallery.zh-CN.md)**
包含全部 12 幅结果图、23 个矢量面板、对应数据与计时说明。
[复现流程](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)
从产品安装一直连接到原生调用和应用测量。

## 论文实验结果

论文覆盖 **553 个算子—平台案例**，其中 **543 项具有匹配的原生参考**，
以及 **18 个模型阶段**。以下图片和表格来自[论文数据](docs/paper/README.zh-CN.md)，
倍率均为参考耗时除以 Triton–Weft 耗时。

### 算子性能

[![代表算子与全部 543 项匹配原生比较](docs/paper/operators-ecdf.svg)](docs/paper/operators-ecdf.svg)

| 平台 | 正式用例 | 匹配比较 | 几何平均加速 |
|---|---:|---:|---:|
| SG2044 · RVV128 | 133 | 132 | **1.60×** |
| K1 · RVV256／IME1 | 137 | 136 | **1.54×** |
| K3 A100 · RVV1024／IME2 | 136 | 135 | **1.83×** |
| Xeon Platinum 8470Q · AVX-512／VNNI／AMX | 147 | 140 | **1.32×** |

参考包含 541 项 GGML 和两项 oneDNN。
[算子 CSV](docs/paper/operators.csv)记录工作负载、数值检查、耗时、参考与 JIT 观察。

### 模型推理

[![九组模型配置的 prefill 与 decode 配对结果](docs/paper/models.svg)](docs/paper/models.svg)

| 平台 | 模型 | 参考 | Prefill | Decode |
|---|---|---|---:|---:|
| x86 | Llama 2 7B Q4_0 | llama.cpp | 1.705× | 1.217× |
| x86 | DeepSeek 8B Q4_K_M | llama.cpp | 1.186× | 1.194× |
| x86 | Qwen 2.5 14B Q4_K_M | llama.cpp | 1.154× | 1.179× |
| SG2044 | Llama 2 7B Q4_0 | llama.cpp | 1.260× | 1.011× |
| SG2044 | DeepSeek 8B Q4_K_M | llama.cpp | 1.167× | 1.016× |
| K1 | Llama 2 7B Q4_0 | llama.cpp | 1.432× | 1.026× |
| K1 | DeepSeek 8B Q4_K_M | llama.cpp | 1.188× | 1.168× |
| K3 A100 | DeepSeek 8B Q4_K_M | vendor IME2 | 1.157× | 1.361× |
| K3 A100 | DeepSeek 8B Q4_K_M | CPU-vector | 1.939× | 1.976× |

SG2044 模型使用 64 线程，K1、K3 和 x86 使用八线程。SG2044／K1／x86 为 128-token prefill、512-token context 后生成
32 tokens；K3 为 32-token prefill、32-token context 后生成 8 tokens。
[模型 CSV](docs/paper/applications.csv)保留全部 18 个阶段的耗时、参考策略和替换计数。

### 性能来自哪里？

| 证据 | 查看内容 |
|---|---|
| [向量／矩阵、FP32 库和线程扩展](docs/paper/gallery.zh-CN.md#向量与矩阵执行) | 硬件利用、完整调用成本与多核执行 |
| [五路线作者控制](docs/paper/gallery.zh-CN.md#五条路线实现相同控制任务) | C、Triton CPU、Exo 2、TensorIR、Weft 中的 Q4_K／Q6_K 准备与精度选择 |
| [算术与状态范围](docs/paper/gallery.zh-CN.md#算术组织与输出状态) | 同一编译器和物理绑定下的源程序组织 |
| [源码增量与复用](docs/paper/gallery.zh-CN.md#源码增量与复用) | 新格式、第二个算子、注意力和流式统计 |
| [编译机制](docs/paper/gallery.zh-CN.md#固定源语义的编译机制) | 固定源语义后的 Encoding、转发、片段保留与数值范围 |

[![两个算子之间新增与复用的源码](docs/paper/increment-measurements.svg)](docs/paper/increment-measurements.svg)

[数据指南](docs/paper/README.zh-CN.md)链接全部结果表与精确图件来源。
[Triton–Weft 中文手册](https://tianchen-ai.github.io/triton-weft/zh/)
说明这些实验背后的模型与编译器。

版本 **1.0.0**。先按照 [Triton–Weft 安装指南](https://tianchen-ai.github.io/triton-weft/zh/getting-started/installation/)安装产品，正常内核由 `triton.kernels.weft` 提供。

## 安装与复现

本工具只围绕两种工作：

1. **算子与参考库比较**：选算子、输入和实现。支持 Triton–Weft / Weft，以及独立 GGML、OpenBLAS、oneDNN 参考。
2. **模型阶段计时**：正常 llama.cpp 可以独立建立 baseline；指定 Weft 替换映射时，由工具准备 package 并比较。

输入生成、输出分配、数值参考和计时都由已有 workload 与公共 worker 提供。
更换同一算子的实现，不需要写 C++ benchmark。SSH、机器路径和历史结果在外围。
另有可选的 `verify`，用于选定的编译、数值和框架验证；它不产生性能倍率。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
export PATH="$PWD/.venv/bin:$PATH"
export WEFT_BENCH_CONFIG=/path/to/bench.local.json

weft-bench list
weft-bench list --kind inputs
weft-bench show rms_norm --inputs decode

weft-bench op q4_k_dot --target rvv128
weft-bench op rms_norm --weft /work/norm.py:my_norm --inputs decode --target rvv128
weft-bench op rms_norm --weft /work/artifact --inputs /work/input.json --save-outputs --target rvv128

weft-bench list --kind replacements
weft-bench e2e llama2-7b-q4_0 --target rvv128 \
  --replace ggml.vec_dot/q4_0=/work/dot.py:my_dot --phase both
weft-bench e2e llama2-7b-q4_0 --target rvv256-ime \
  --replace ggml.ime1/q4_centered_prefill=default --phase prefill
```

`op` 默认运行 GGML 和 Weft。两边使用同一份准备好的输入，各自持有输出，
在同一个原生进程中使用公共计时器。可以仅运行 `--impl ggml` 或 `--impl weft`。
外部配置的 `default_implementations` 可以将 x86 默认设为纯参考库；
`--impl openblas` 测量 F32 GEMM，`--impl onednn` 测量 strict F32、BF16/F32 或 S8/S32 matmul。
这条路径不要求安装 Weft 编译器。实际库派发、CPU/OS 状态、affinity 和 NUMA 随结果记录。
已有固定成绩可用 `--baseline /path/to/results.json` 复用，默认只运行 Weft。

输入独立选择：`default`、`decode`、`prefill`，或自定义 JSON/数组/文件。
`show` 给出输入输出名称、dtype、shape 和 layout。JSON 结果记录同一约定，
`--save-outputs` 额外保存两边的 Y，便于直接读取。

模型替换通过 `--replace 角色=kernel` 或 `--replacement-file mappings.json` 指定。
工具调用所选编译器的公共编译与 package API，生成仅含指定映射的 package。
Triton–Weft 直接使用导出的 native library 与 `triton.Package.create`；
旧 Weft adapter 使用原有编译/link 流程，配置决定选择哪条路径。
替换入口目前固定为 vec-dot、activation quantization 和 packed IME。
`--prefill`、`--decode`、`--depth`、`--threads` 调整预置场景。
准备好的集成版本负责真正调用替换；结果显示测量窗口内是否命中。

`check`、`checkpoint`、`explain` 和 `replay` 保留为外围开发便利功能。
普通算子比较不需要先建立 checkpoint。编译器/安装包验证通过 `verify --list`
发现配置好的条目，再用 `--case` 或 `--group` 选择；性能仍使用 `op` / `e2e`。

插件名称是 **Weft Bench**，同时支持两种编译体系。安装插件不会自动安装编译器，
也不会自动切换外部默认环境；启动器的 `--environment triton-weft` 选择本机配置的
对应环境。机器地址、凭据、模型和私有配置不随仓库分发。

详见：[安装与依赖](docs/install.md) · [快速开始](docs/quickstart.md) · [输入和实现接口](docs/extending.md) ·
[环境配置](docs/targets.md) · [职责边界](docs/migration.md) ·
[x86 参考库与固定 baseline](docs/x86-references.md) ·
[Codex 插件安装与使用](docs/codex-plugin.md) · [正确性验证](docs/verification.md)。

# Triton–Weft Bench

[English](README.md) · [论文实验复现](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/) · [论文 CSV](docs/paper/README.zh-CN.md)

**使用 Triton–Weft 与本 Bench 工程复现论文中的算子和 llama.cpp 实验。**
Catalog 登记作者程序、shape、参考与模型 package，
[双语复现流程](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)
覆盖目标机 JIT、完整套件、原生参考和 prefill／decode。

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

# 安装与选择执行环境

[English](install.md) · [工具书](README.zh-CN.md)

先按照 [Triton–Weft 安装指南](https://tianchen-ai.github.io/triton-weft/zh/getting-started/installation/)
安装产品，再安装 Bench 1.0。目标侧 Python 也需要能导入产品内核库。

```sh
git clone https://github.com/Tianchen-ai/triton-weft-bench.git
cd triton-weft-bench
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[verification]'
weft-bench list
```

Python 要求 3.10+。源码开发使用 `pip install -e '.[verification]'`。
`verification` extra 提供 NumPy；需要在真正执行验证的 Python 环境中安装。

## 配置依赖

| 工作 | 执行环境 |
|---|---|
| GGML/OpenBLAS/oneDNN | 原生 C/C++ 工具链、对应头文件与库 |
| Triton–Weft JIT | 已安装产品、Bench、目标原生工具链 |
| 已导出 AOT | artifact、匹配的原生 SDK 和 worker 工具链 |
| PyTorch 验证 | 兼容的 CPU Torch 与产品 |
| llama.cpp | 正常/集成构建、GGUF 模型、SDK 与显式替换映射 |

产品提供 `triton.kernels.weft`，Bench 包装不保存第二份正常内核实现。
机器配置放在忽略的 `.local/` 中：

```sh
mkdir -p .local
cp configs/example.json .local/bench.local.json
export WEFT_BENCH_CONFIG="$PWD/.local/bench.local.json"
weft-bench show q4_k_dot --target x86-1t
```

修改目标 Python、允许的 CPU、SDK 和参考库路径后，
按照[快速开始](quickstart.zh-CN.md)进行比较。
源码 wheel、目标 ABI、NumPy ABI 与系统库需对应同一执行环境。

# 程序、参考与实验目录

[English](catalogs.md) · [工具书](README.zh-CN.md)

`src/weft_bench/catalogs/` 登记目标、工作负载、参考库和模型配方。
每项包含数学任务、shape、dtype、调用范围、源码及 symbol、
constexpr、后端选项和参考 provider。

```sh
weft-bench list --kind kernels --target x86-1t
weft-bench list --kind suites --target k3-a100 --config bench.local.json
weft-bench show q4_k_dot --target x86-1t --config bench.local.json
weft-bench list --kind baselines
```

四个 catalog 分别包含 SG2044 133、K1 137、K3 A100 136、x86 147 项，
总计 553 项。`full` 显式列出全部成员，`show` 与单项 `op` 使用相同 shape。

## 源码归属

| 位置 | 内容 |
|---|---|
| `triton.integrations.ggml_formats` | 共享基础格式、字段与字节布局 |
| `triton.kernels.weft.ggml.layouts` | 作者选择的矩阵/交错布局 |
| `triton.kernels.weft.ggml.blocks` | 编码块算术与完整投影 |
| `triton.kernels.weft.programs` | 归一化、注意力、MoE、Mamba 与宿主组合 |
| Bench `adapters` 与 `libraries` | 输入、参考与调用约定 |

`package:kernels/...` 解析到产品已安装的内核库。
`weft_bench.kernels` 是兼容导入层。
Encoding 描述存储，host 准备或显式 kernel 产生该存储。

## 参考与模型

参考目录包括优化 GGML、IME/IME2、OpenBLAS、oneDNN 和正常 llama.cpp。
配置提供具体 checkout、构建、工具链与模型位置；
运行结果保存实际分派和构建信息。
[论文数据](paper/README.zh-CN.md)与当前源码目录分开保存，包含十四份最终表。
模型 package 配方固定角色绑定，`prepare-model` 生成部署包，
随后在模型 profile 中显式选择它。

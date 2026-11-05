# TileLang–Weft 源码 JIT 与原生部署

[English](https://tianchen-ai.github.io/triton-weft-bench/tilelang-weft-integration/)

目标环境安装 TileLang–Weft、匹配的 `weft-core` 和本 Bench wheel。
源码 JIT 选择 `weft_bench.adapters.triton_jit:TileLangJitAdapter`，复用已有 fixture、
输入、数值检查、原生 timer、SSH 执行和结果存储。

候选继续使用已有 source 配置：

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

通过 `show affine --target TARGET` 查看 workload，再使用当前本地 target 配置执行
`op affine --impl tilelang --check-only --target TARGET --output OUTPUT`。
测量时去掉 `--check-only`。worker 分开记录首次编译、Python/grid 热调用、磁盘缓存
恢复和导出原生入口；实际源码 grid 与测量合同核对，调优时记录选定配置的 grid。

eager 函数从 workload port 获取 Tensor metadata；显式 `signature` 可提供
shape/dtype/strides。packed raw storage 保留作者的 Encoding 声明，字节 Tensor
geometry 与逻辑编码 View 分开。buffer 维度表达式使用 TileLang 已有 binder 求值。
多输出、readwrite 与 workspace 沿用已有 workload 和 `workspaces` 配置。

原生部署使用 `weft_bench.adapters.triton_native:TileLangArtifactAdapter`。
`source` 通过 TileLang 自身 source/AOT API 编译，`artifact` 加载已导出的库；
结果标记为 native deployment，与源码 JIT 分开。

llama.cpp 配置选择 `producer: "tilelang"`，在 `tilelang` 配置中使用已有的
`python`、`compile_target`、`toolchain` 和逐角色 `bindings`。
`prepare-model` 在推理前编译并调用共同角色校验器打包；`validate-model`、`e2e`
复用已有模型 runner 和实际命中检查。真正 GGML 行 ABI 使用 `deployment.vec_dot`
程序；矩阵 shape 的 dot 具有不同入口合同。

普通作者程序属于 TileLang–Weft。独立的
[TileLang–Weft Bench](https://github.com/Tianchen-ai/tilelang-weft-bench)
维护源码绑定、四个正式套件和稳定 `output/` 目录中的结果。
[双语结果手册](https://tianchen-ai.github.io/tilelang-weft-bench/results.zh-CN.html)
公开 553 项算子与平台观测、543 项匹配外部参考、原生性能图及原始样本。
数值检查范围、源码参数、缓存恢复和 Python 热调用分别记录。

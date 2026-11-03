# x86 参考库与模型基线

[English](https://tianchen-ai.github.io/triton-weft-bench/x86-references/) · [工具书](README.zh-CN.md)

按计算选择正常优化的原生库，候选和参考采用相同输入、线程与计时范围。

| 参考 | 任务 | 调用边界 |
|---|---|---|
| GGML | 编码点积、量化/反量化、投影、norm、RoPE、softmax、FP32 GEMM | 原生函数或准备后的 CPU graph |
| OpenBLAS | FP32 GEMM | LP64 CBLAS SGEMM |
| oneDNN | FP32、BF16/F32、S8/S32 matmul | execute + wait，包含内部 packing |
| 正常 llama.cpp | GGUF 推理 | 分开的 prefill/decode 阶段 |

oneDNN 使用显式 stride 和 strict FP32 数学；
原语创建及 scratchpad 分配在计时外，内部 packing 在调用内。
OpenBLAS/oneDNN 配置提供头、库和版本。
所有参考使用匹配编译器与 OpenMP。

## 权重、激活与数值检查

完整投影的持久权重布局在重复调用前准备，激活量化属于每次调用。
工作区按 artifact ABI 分配，并在数值检查前填充，kernel 必须初始化所消费内容。
混合 F16 权重与 F32 激活采用独立声明的输入合同。
模型的正常数值参考固定 GGUF、设置和 logits 观察位置。

## 线程与执行证据

```sh
weft-bench environment --target x86-1t --config /work/bench.local.json
weft-bench op gemm_f32 --impl openblas --target x86-1t
weft-bench op gemm_f32 --impl onednn --target x86-1t
weft-bench op q4_k_dot --impl ggml --target x86-1t --output output/baselines/q4
```

配置固定 affinity、NUMA 与线程；结果记录实际分派、构建及环境。
频率快照是诊断，不用于修正耗时。
AMX 执行依据真实产物和已运行路径，调用之间恢复各自 tile 配置。
固定 baseline 检查 shape、输入、缓存和调用边界后复用；
详细库字段与布局合同见[英文版本](x86-references.md)。

# Triton–Weft 集成

[English](triton-weft-integration.md) · [工具书](README.zh-CN.md)

Bench 使用产品公共接口：`@triton.jit(model="weft")`、
`triton.compile`、`CompiledKernel.export`、`triton.load` 与
`triton.Package.create`。编译、缓存、调优和代码生成属于产品。

## 原生部署与交叉编译

`weft_bench.adapters.triton_native:TritonAdapter` 接受源码或导出 artifact。
源码指定 Python、source/symbol、signature、constexprs、后端选项和工具链。
AOT 提供带 native library、manifest、header 与依赖的目录。
目标配置提供 ISA/VLEN 和执行侧 SDK；host 可以使用显式 cross toolchain/sysroot。

View ABI 固定逻辑 shape、存储、效果与 stride。
pointer ABI 使用任务实际缓冲区和显式尺寸参数。
Bench 的 C++ 编译器构建 worker，候选库直接来自产品导出。

| 范围 | 包含 |
|---|---|
| native single | 完整 C ABI 调用 |
| native grid | grid 调用、调度与同步 |
| Python JIT | 热 `kernel[grid]` 与 Python 绑定 |
| 模型阶段 | 完整 prefill 或 decode 窗口 |

## 目标侧 JIT

`TritonJitAdapter` 使用目标已安装的 Python、产品与 Bench，
正常源码解析到 `triton.kernels.weft`，自定义源码按相对包依赖传输。
输入、工作区、别名和 oracle 复用已有 workload。
编译、首次调用、进程内缓存与新进程磁盘缓存分别记录。

`--cache cold` 选择数据缓存模式，独立于编译缓存。
输入恢复与 eviction 在调用外；完整投影所需量化在调用内。
grid 包含全部任务，不能把 grid 耗时除以实例数充当 leaf 耗时。

## llama.cpp 与诊断

package 固定角色到内核的映射，集成后端直接加载导出库。
模型测量记录配置和实际命中计数。
IR、源码、编译日志与 ABI 保存在结果目录，
`explain/replay` 使用这些材料定位同一工作负载。

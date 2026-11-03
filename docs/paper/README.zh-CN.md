# 最终论文数据

[English](README.md) · [复现流程](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)

十四份 CSV 保留论文最终测量。图件使用最终认可的矢量 SVG，
[来源清单](provenance.json)记录论文版本、文件哈希、行数与线程数。

| 表格 | 行数 | 内容 |
|---|---:|---|
| [operators.csv](operators.csv) | 553 | 全部算子—平台案例与匹配耗时 |
| [operator-families.csv](operator-families.csv) | 5 | 类别数量与几何平均 |
| [applications.csv](applications.csv) | 18 | 完整推理阶段与参考策略 |
| [application-x86-windows.csv](application-x86-windows.csv) | 120 | x86 分进程阶段测量 |
| [application-roles.csv](application-roles.csv) | 18 | 计算和准备阶段的作用 |
| [fp32-libraries.csv](fp32-libraries.csv) | 12 | FP32 GEMM 对比三个库 |
| [grid.csv](grid.csv) | 48 | 1/2/4/8 线程完整调用 |
| [ggml-compute.csv](ggml-compute.csv) | 32 | GGML 工作线程内计算 |
| [author-controls.csv](author-controls.csv) | 240 | 五路线的共享与精度控制 |
| [author-organization.csv](author-organization.csv) | 48 | 算术与输出状态范围 |
| [five-route-bindings.csv](five-route-bindings.csv) | 30 | 选定任务绑定 |
| [format-addition.csv](format-addition.csv) | 5 | Q4 构造和新增 Q6 的源码增量 |
| [state-algorithms.csv](state-algorithms.csv) | 36 | 注意力与 Welford 组织 |
| [compiler-mechanisms.csv](compiler-mechanisms.csv) | 214 | 配对物理编译机制 |

## 阅读数据

`operators.csv` 的 `weft_ms`、`reference_ms` 是匹配 native 调用耗时，
`speedup` 为参考除以 Weft；`shape`、`scope` 与 `correctness`
说明形状、调用边界与检查。553 项中有 543 项匹配参考。

`applications.csv` 包含八组模型—平台部署的完整阶段，
另行记录 K3 CPU 向量参考，共18行。SG2044 为64线程，其余为八线程。
`aggregation` 指明独立进程的聚合方式，x86 原始阶段窗口在单独表内。

控制实验的 `s0/s1` 表示局部/共享准备，`g0/g1` 表示直接/分组累加。
编译器机制固定作者程序，比较物理机制开启与关闭；
`disabled_over_enabled` 及样本列保留对应耗时与倍率。

## 图件与复现

[![代表算子](operators-heatmap.svg)](operators-heatmap.svg)

[![全部匹配分布](operators-ecdf.svg)](operators-ecdf.svg)

[![完整推理](models.svg)](models.svg)

安装 [Triton–Weft](https://github.com/Tianchen-ai/triton-weft)
与本 Bench，配置参考库及目标环境，使用 `show` 查看工作负载，
再运行 `op`、`prepare-model` 与 `e2e`。
[双语完整指南](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)
提供命令。产品供应正常内核，Bench 负责输入、比较与计时。

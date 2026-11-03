# 论文结果与数据

[English](https://tianchen-ai.github.io/triton-weft-bench/paper/) · [完整结果图集](gallery.zh-CN.md) · [复现流程](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)

[完整图集](gallery.zh-CN.md)呈现论文全部结果图（图 9–20），依次解释算子／应用性能、
作者控制与复用、编译机制。下表保留最终测量和源码计数，
[来源清单](provenance.json)记录论文版本、哈希、行数、线程数与每个矢量面板的来源。

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
| [increment-costs.csv](increment-costs.csv) | 3 | 图 17：各格式新增与复用 SLOC |
| [increment-definitions.csv](increment-definitions.csv) | 18 | 图 17：具体源码定义及 SLOC |

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

**[完整结果图集](gallery.zh-CN.md)**包含 12 幅结果图的 23 个 SVG 面板，
每组图都附解释、计时范围与数据链接。两份 `increment-*.csv` 保留图 17 使用的
源码定义快照，标识符对应移入产品库前测量的定义。`first` 为反量化源码，
`second` 为新增 dot 源码，两者之和是两个算子合计新增量。
`format-addition.csv` 用 token 度量另一组五路线 Q4_K／Q6_K 任务。

安装 [Triton–Weft](https://github.com/Tianchen-ai/triton-weft)
与本 Bench，配置参考库及目标环境，使用 `show` 查看工作负载，
再运行 `op`、`prepare-model` 与 `e2e`。
[双语完整指南](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)
提供命令。产品供应正常内核，Bench 负责输入、比较与计时。

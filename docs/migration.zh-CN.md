# 产品与 Bench 的职责

[English](migration.md) · [工具书](README.zh-CN.md)

Weft 1.0 的正常实现由产品管理，Bench 通过公共编译、导出和运行接口消费它。

| 资产 | 所属 |
|---|---|
| 编程模型、前端、pass、目标代码生成 | Triton–Weft |
| JIT、调优、缓存、artifact/package 和 SDK | Triton–Weft |
| 正常内核、格式、helper、布局和宿主组合 | `triton.kernels.weft` |
| 工作负载输入、数值 oracle、参考调用 | Bench |
| 原生计时、目标部署和模型比较 | Bench |
| 冻结论文表与实验记录 | Bench 数据与独立实验材料 |

兼容路径 `weft_bench.kernels` 转发到产品，避免两套正常内核。
模型与框架正确性功能属于产品，比较驱动和研究干预属于 Bench。
旧结果保持原始形状、缓存、线程和调用边界；与新协议兼容时才复用固定参考。
`checkpoint` 汇总用户选定记录，`explain/replay` 用于解释与复现。

原生 worker 只组合实现调用、输入和计时；SSH、编译位置与历史比较由外围管理。
产品运行时不依赖 Bench。完整接入见[集成章节](triton-weft-integration.zh-CN.md)。

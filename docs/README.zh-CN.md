# Triton–Weft Bench 1.0

[English](https://tianchen-ai.github.io/triton-weft-bench/)

**从 Weft 程序，走到条件匹配的 CPU 实验。**

Weft 让作者组合表示、算术、共享和状态。Bench 把这些程序接到匹配的输入、
优化参考、数值检查与原生计时。论文先从单个内核看到完整 llama.cpp 阶段，
再通过作者控制、复用及编译器对源关系的利用解释性能来源。

[![全部 543 项匹配算子比较](paper/operators-ecdf.svg)](paper/operators-ecdf.svg)

SG2044、K1、K3、Xeon 8470Q 相对优化参考的几何平均加速依次为
**1.60×、1.54×、1.83×、1.32×**。
**[查看全部论文结果图](paper/gallery.zh-CN.md)**：硬件执行、FP32 库、线程扩展、
完整推理、源码控制、格式增量及编译机制。
[数据指南](paper/README.zh-CN.md)链接对应测量和源码计数。

## 运行自己的比较

安装 [Triton–Weft](https://tianchen-ai.github.io/triton-weft/zh/getting-started/installation/)
和 Bench，配置 CPU 与参考库，再选择工作负载和内核。下方指南介绍完整路径，
[Triton–Weft 中文手册](https://tianchen-ai.github.io/triton-weft/zh/)介绍编程模型与安装后的内核库。

| 任务 | 入口 |
|---|---|
| 安装产品与 Bench | [安装](install.zh-CN.md) |
| 运行匹配比较 | [快速开始](quickstart.zh-CN.md) |
| 独立选择内核、数据和库 | [输入与实现](extending.zh-CN.md) |
| 配置本地或 SSH 目标 | [环境](targets.zh-CN.md) |
| 理解正常内核归属 | [产品与 Bench](migration.zh-CN.md) |
| 查看登记的源码与参考 | [目录](catalogs.zh-CN.md) |
| 使用 JIT、AOT、grid 和模型 | [产品集成](triton-weft-integration.zh-CN.md) |
| 建立优化的原生参考 | [x86 参考库](x86-references.zh-CN.md) |
| 验证语义与部署 | [正确性验证](verification.zh-CN.md) |
| 从 Codex 调用 CLI | [插件](codex-plugin.zh-CN.md) |
| 查看全部论文结果图 | [完整图集](paper/gallery.zh-CN.md) |
| 查看测量与源码计数 | [论文表格](paper/README.zh-CN.md) |

产品实现编程模型、编译器、运行时与正常内核库。
Bench 提供共享输入、参考调用、数值检查和测量。

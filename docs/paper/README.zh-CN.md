# 论文算子与模型数据

[English](README.md) · [复现流程](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)

这些表对应 Weft 论文 V10 的整体性能图，
保留冻结 `data/operators.csv` 和 `data/models.csv` 中的测量值，省去机器本地路径。

- [operators.csv](operators.csv)：553 行算子—平台记录，其中 543 项具有匹配原生参考。
- [models.csv](models.csv)：九组模型配对配置的 18 个阶段。

## 阅读各列

`case`、`shape`、`target` 和 `scope` 标识工作负载。
`median_ms`、`baseline_median_ms` 是匹配的原生／模型耗时，
`speedup` 为参考除以候选，`correctness` 记录实际检查范围。
预热、采样、cache 和 protocol 保留测量设置。

JIT 各列分别记录首次编译调用、Python 热调用和磁盘缓存复用。
模型记录包含正常后端策略与实际替换计数。
匹配算子参考包括 541 项 GGML 和两项 oneDNN；
K3 模型分别保存 CPU–RVV、厂商 IME2 策略。

## 复现

安装 Triton–Weft 与 Bench，配置目标和参考，通过
`show CASE --target TARGET` 查看，再使用 `op`、`prepare-model` 和 `e2e`。
[完整指南](https://tianchen-ai.github.io/triton-weft/zh/experiments/reproduce/)提供命令。

V10 的产品／Bench 源码快照为 `369c22929`／`e17240a`。
汇总结合安装包 JIT 覆盖和受影响用例修复，各原始记录保留自身源码与设置。
当前 catalog 提供作者程序入口、shape、参数与参考 provider，
供显式选择安装包和机器后运行。

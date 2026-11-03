# 配置驱动的正确性验证

[English](verification.md) · [工具书](README.zh-CN.md)

`verify` 枚举并执行配置中的产品验证计划，复用算子输入、数值 oracle 与原生适配器。
该入口检查编译、输出和运行时功能，性能由 `op/e2e` 单独测量。

```sh
weft-bench verify --config /work/bench.local.json --target cpu --list
weft-bench verify --config /work/bench.local.json --target cpu --group operators --output /work/checks/cpu
weft-bench verify --config /work/bench.local.json --target cpu --output /work/checks/cpu --resume
weft-bench verify --config /work/bench.local.json --target cpu --case 'operators/q4_*' --output /work/checks/cpu --rerun failed
```

`--list` 只读配置。catalog 提供已选源码和 shape，
JIT 路线调用目标侧安装包，检查输出、进程内/磁盘缓存与导出产物。
原生路线可用 `--prepare-only --workers 2` 在 host 生成产物，
通过 `verification.prepared_entries` 指定后续设备执行。

## 验证内容

配置中的 group 与稳定 ID 区分算子、宿主 Program、运行时、框架和示例。
运行时覆盖 grid、并行、scan、原子读改写与辅助函数；
示例观察器检查 Encoding 字节、尾部和完整结果。
状态任务在调用外 reset，整数精确比较，浮点采用声明的独立参考与容差。
目标 Python 安装 `.[verification]`，框架项另需兼容的 Torch。

## 结果与恢复

`verification.json` 保存当前条目，`failures.json` 保存需要处理的结果，
各 case 目录保存请求、输出和诊断。
`--stop-after-current` 在当前条目结束后停止，`--resume` 继续待办，
`--rerun failed|all` 选择复测范围。

恢复检查源码、编译器、安装路径与环境标识；环境改变时使用新标签或显式复测。
报告使用本次 `selection_counts`，区分编译失败、构建失败、
运行失败、数值失败和平台不适用。
原生条目可交给 `explain/replay`；完整配置字段和观察合同见[英文版本](verification.md)。

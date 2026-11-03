# 选择、比较与查看结果

[English](https://tianchen-ai.github.io/triton-weft-bench/quickstart/) · [工具书](README.zh-CN.md)

完成[安装](install.zh-CN.md)，选择外部配置：

```sh
export WEFT_BENCH_CONFIG=/path/to/bench.local.json
weft-bench list --kind targets
weft-bench show q4_k_dot --target x86-1t
weft-bench list --kind suites --target x86-1t
```

`show` 给出输入输出、shape、格式、作者程序和参考。
target 是本地配置中的名字。正式 catalog 包括 `sg2044`、`k1`、
`k3-a100` 与 `x86-1t`。

## 算子比较

```sh
weft-bench op q4_k_dot --target x86-1t
weft-bench op rms_norm --target x86-1t --inputs decode
weft-bench op rms_norm --target x86-1t --weft /work/norm.py:my_norm
weft-bench op --suite full --target x86-1t
```

`--weft` 可以指定源码及 symbol，也可指向导出目录。
`--shape M=2,N=64`、`--seed 123` 或 `--inputs input.json` 独立选择输入。
同一 worker 为两边准备相同数据、分配输出、运行数值参考，再计时原生调用。

## 固定参考

```sh
weft-bench op q4_k_dot --target x86-1t --impl ggml --output output/baselines/q4
weft-bench op q4_k_dot --target x86-1t --baseline output/baselines/q4
```

参考复用检查形状、输入、缓存、调用范围和执行设置。
完整投影包含激活准备；点积微内核消费已打包操作数。

## 模型与输出

```sh
weft-bench list --kind models --target k1
weft-bench prepare-model --target k1 --model llama2-7b-q4_0 --output output/model-package
weft-bench e2e llama2-7b-q4_0 --target k1 --phase both
```

模型配置显式选择 GGUF、正常/集成二进制和准备后的 package。
`--replace ROLE=SOURCE:SYMBOL` 选择已有角色，结果记录测量区间内的调用。
`results.json`、CSV、日志和可选 `--save-outputs` 保存输出及其类型形状。
`explain` 查看记录，`replay` 使用保存的输入与配置复现。
[验证](verification.zh-CN.md)只检查正确性，性能仍由 `op/e2e` 测量。

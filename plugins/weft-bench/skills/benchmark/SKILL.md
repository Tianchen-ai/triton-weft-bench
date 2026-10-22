---
name: benchmark
description: 使用 Weft Bench 比较 Triton–Weft 的 x86/RVV kernel 与 GGML 等参考库，或测量 llama.cpp 的 prefill/decode；支持配置验证、固定 baseline 复用与失败复现，也保留旧 Weft 接入。
---

# Triton–Weft Bench

直接使用已安装的 `weft-bench`，默认候选为正式 Triton–Weft 安装。算子使用现有 workload、输入与固定 reference；
模型使用正常 llama.cpp 与显式选择的 Weft package。插件负责操作指引和本机环境发现，
测量、编译、package、SSH 与结果比较由现有工具完成。

## 找到入口

以下 `BENCH_SKILL` 指向本 `SKILL.md` 所在的绝对目录。不要推定插件缓存版本或当前工作目录。

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" list
```

启动器将原参数交给 CLI，不改变当前工作目录。它读取本机
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json` 中的 `command` 和 `config`。
没有该文件时使用 PATH 中的 `weft-bench` 与常规 `bench.local.json` 配置位置。
显式 `--config` 优先于 `WEFT_BENCH_CONFIG`，后者优先于本地默认值。
`--plugin-info` 只显示入口、配置位置和可用 target 名称，不连接设备。

本机可在 `environments` 中配置独立的 Bench 安装和配置。启动器前缀
`--environment NAME` 选择其中一组；未指定时保持原默认环境。显式环境的配置优先于继承的
`WEFT_BENCH_CONFIG`，CLI `--config` 仍可覆盖。`--plugin-info` 会列出已配置环境名称。
正式环境通过 target 选择 sg2044、k1、k3-a100 或 x86-1t，不再依赖旧 Weft 默认路线。
读取 [Triton native 接入](references/triton.md)，使用 `--plugin-info` 核对实际安装与配置。
`list --kind targets|kernels|baselines|models|suites` 展示正式登记项；
`show CASE --target TARGET` 给出 DSL、物理参数、计时范围和保留产物证据。
数学上属于矩阵乘法不表示执行矩阵指令；区分作者 engine、产物扩展与实际分支观察。
使用 x86 候选或参考库时读取 [x86 工作流](references/x86.md)。同一编译链支持 pointer/scalar、
dense/packed View、动态及大型 local Value，含 Level/state/stage；保留作者原 shape/tile，
按实际 artifact 的 ISA 与运行权限执行。Config、缓存与 AOT 语义见
[source 参数](references/triton.md#source-parameters-and-triton-config)。AMX 和模型集成仍需各自证据。

配置缺失时，先使用任务已给出的配置；需要查找已保存的机器资源或凭据时使用资源插件。
依赖安装和配置方式见 [环境与操作](references/workflows.md)。

## 按任务选择流程

| 用户要完成的事 | 入口 |
|---|---|
| 独立参考库 / 正常模型 baseline | `op CASE --impl NAME`（ggml/openblas/onednn）/ `e2e MODEL --impl normal` |
| CPU/OS、affinity 与 NUMA 观察 | `environment --target TARGET`，不产生性能或 AMX 执行结论 |
| 单次 native 数值检查 | `op CASE --impl NAME --check-only --target TARGET` |
| 当前 kernel 对 GGML | `op CASE --weft SOURCE:SYMBOL --target TARGET` |
| 已有编译产物对 GGML | `op CASE --weft ARTIFACT_DIR --target TARGET`；Triton 产物选择已配置 native adapter |
| Triton 本机 JIT / CPU Tensor | `op CASE --impl NAME --target TARGET`，选择 `TritonJitAdapter` 配置 |
| 改输入或 shape | `show CASE`，然后 `--inputs PRESET_OR_JSON` / `--shape ...` |
| 运行配置中的用例组 | `list --kind suites --config CONFIG --target TARGET`，再用 `op --suite NAME`；见 [环境与操作](references/workflows.md) |
| 使用固定 ref | 同一场景加 `--baseline REF_RUN`；默认只测候选 |
| 改前改后比较 | `compare BEFORE AFTER`；配置了旧版时也可用 `check --case CASE` |
| 构建已选模型配方 | `prepare-model --model MODEL --target TARGET --output DIR`；配方已有明确角色和 source |
| 指定 llama.cpp 替换 | `e2e MODEL --replace ROLE=KERNEL --phase prefill|decode|both`；`--model-profile` 选择配置的正常后端策略 |
| 只检查模型数值或定位 dot 差异 | `validate-model MODEL --target TARGET --output DIR --validation-reference REF_DIR`；`--check-dots` 保存首个超差输入 |
| 解释或复现失败 | `explain RUN --case CASE` / `replay RUN --case CASE` |
| 验证编译器改动、安装包或运行时 | `verify --target TARGET --list` 查看任务配置，再按 `--case` / `--group` 选择；见 [验证与恢复](references/verification.md) |

使用 [环境与操作](references/workflows.md) 中与当前任务有关的例子。
用 `list --kind inputs`、`list --kind replacements` 和各命令 `--help` 查询当前安装版支持项。
换 kernel 使用参数，复用已有输入和 ref；只有计算语义或 IO 边界确实不同才需要增加 workload。

## 性能开发中需要保留的语义

- 用户只要求调查或解释时读取已有结果；要求测量或优化时运行相关 case。
  默认选择改动涉及的 workload，用户要求套件时再运行套件。
- 固定 ref 已匹配当前场景时直接复用。核对 shape、数据、cache 模式、线程、设备和计算范围；
  工具报告不匹配时呈现差异，补建对应场景的 ref，保留原结果。
- dot leaf、多行 dot、完整 projection、IME packed leaf 是不同边界。
  把普通 dot 的改善与实际模型路径关联，不能用 projection 成绩充当 dot 成绩。
- Triton 的 native 单实例、完整 grid、Python `kernel[grid]` 分别计时；冷 JIT 与 warm
  Python 调用也分别记录。不能把 Python 入口时间和 native GGML 时间算成算子加速。
- native `entry=single` 使用单线程 target；环境线程数不会为 kernel 创建并行任务。
  并行候选使用已配置的 grid/framework 入口及其实际线程记录。F16 projection 还需区分
  转换 activation 与保留 F32 activation 的合同，见 [x86 工作流](references/x86.md)。
- E2E 的 `--phase` 选择测量阶段；同一个 `--replace` 映射用于所有所选阶段。
  同一入口需要两种阶段专用 kernel 时分两次调用。
- `--replace` 只作用于已有 Weft llama.cpp hook 的对应入口，使用已集成候选构建。
  保留正常 baseline 的 RVV、repack、OpenMP、IME 等配置。
  从测量区间计数确认命中；`not_observed` 不能解释成零命中或已命中。
- 不根据一次微基准自动选择或更换模型 package。实际替换由用户请求及显式参数确定。
- 默认数值检查、预热和计时由工具处理。无需为每次 bench 再写测试脚本或扩展全量矩阵。
  只有用户要求时采集 `--profile`；已有热点使用 `hotspots` 复用。
- `verify` 是配置驱动的正确性流程，不生成算子耗时或加速比。编译成功、数值通过、
  框架支持和性能改善分别呈现；不要因用户要求性能测量而自动运行完整验证清单。

输出保存在 Bench 的 `output/`：固定 baseline、current、reports 分开。默认 target/case 路径稳定，
新状态写完后替换旧状态，失败不能被上次成功掩盖。依赖与环境位于 `.local/`，不进入 Git。
当前安装与正式目录从 `--plugin-info` 和本地配置读取，公共说明不保存机器路径。
返回结果表或 CSV/JSON 链接、
对应范围和下一步定位入口；区分编译失败、数值失败、运行成功和性能改善。
输出与 replay 材料可能包含本机路径、SSH 别名及配置，只在任务需要时共享。

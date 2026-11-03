# 从 Codex 使用 Weft Bench

[English](codex-plugin.md) · [工具书](README.zh-CN.md)

插件为 agent 提供已有 CLI 的工作流与本机配置发现，不增加另一套测量引擎。
先[安装 Bench](install.zh-CN.md)，再在仓库中注册：

```sh
codex plugin marketplace add .
codex plugin add weft-bench@weft-tools
codex plugin list --marketplace weft-tools --json
```

插件内容位于 `plugins/weft-bench`。
CLI 与配置位置可放在
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json`：

```json
{
  "command": ["/path/to/venv/bin/weft-bench"],
  "config": "/path/to/bench.local.json"
}
```

显式 `--config` 优先，其次为 `WEFT_BENCH_CONFIG`。
可选 `environments` 定义另一组 command/config，
启动器使用 `--environment NAME` 选择；硬件通过 `--target` 选择。
凭据、机器地址和模型路径保留在本地。

## 检查与使用

```sh
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py --plugin-info
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py list --kind replacements
```

可要求 agent 比较 Q4_K 与 GGML、复用 baseline、测量模型阶段，
或检查选定的运行时功能。
修改插件后重新执行 `codex plugin add`，核对实际 installedPath。
用户任务决定读取结果、执行正确性还是进行性能测量。

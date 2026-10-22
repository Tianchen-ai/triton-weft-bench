# Use Weft Bench from Codex

The `weft-bench` plugin gives a Codex agent the existing Triton–Weft and Weft
operator/model workflows and configured numerical/runtime verification, plus a
launcher that finds this machine's CLI and configuration.
It is a local skill plugin: agents call the real CLI through their terminal.
It does not introduce another benchmark engine or require an MCP service.

## Install

Install the [published Bench wheel](install.md), then register the plugin from a
checkout with Codex CLI installed:

```sh
codex plugin marketplace add .
codex plugin add weft-bench@weft-tools
codex plugin list --marketplace weft-tools --json
```

The repository marketplace points only at `plugins/weft-bench`. Installing it
enables the plugin for the user, so agents in other repositories can discover it.
Start a new Codex chat after installation; if the desktop plugin list is stale,
restart the desktop app. This installs a local plugin, not a public directory entry.

For a virtual environment or a configuration outside the current shell, create
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json` outside the repository:

```json
{
  "command": ["/path/to/venv/bin/weft-bench"],
  "config": "/path/to/bench.local.json"
}
```

Otherwise the launcher finds `weft-bench` on PATH and uses
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/bench.local.json` when it exists.
Explicit `--config` and `WEFT_BENCH_CONFIG` override these defaults.
Optional `environments` entries select a separate CLI/configuration with the launcher
prefix `--environment NAME`. Each entry contains its own `command` and `config`.
An explicitly selected environment overrides inherited `WEFT_BENCH_CONFIG`; CLI
`--config` still takes precedence. Omitting the prefix keeps the existing default.
The formal installation uses one Triton–Weft CLI/configuration and selects hardware
with `--target`. Optional environments remain available for explicit comparisons;
they do not select a compiler merely from the plugin name.
Machine addresses, SSH credentials, model paths and saved results are not packaged.
See [environment setup](targets.md) for the bench target configuration.

## Ask the agent

Examples:

- “使用 Weft Bench 插件，把这个 Q4_K kernel 与 GGML 比较，复用已有 baseline。”
- “使用 Weft Bench 插件，替换 llama.cpp 的 Q4_0 vec-dot，分别测 prefill 和 decode。”
- “使用 Weft Bench 插件，解释这次结果和上次的差别，并复现失败项。”
- “在配置的 Triton–Weft 环境中，用 verify 检查这次改动涉及的算子和运行时入口。”
- “在配置的 x86 环境中建立 GGML/OpenBLAS/oneDNN baseline，记录实际 AMX primitive，并单独跑正常 llama.cpp。”

The bundled skill is `weft-bench:benchmark`. It is eligible for automatic selection
on relevant Weft performance tasks. It discovers supported workloads, input presets
and replacement slots from the installed CLI. User intent determines whether to
read existing results, run selected operators or measure model stages.

The launcher can be checked directly from this checkout without connecting a target:

```sh
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py --plugin-info
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py list --kind replacements
```

After installation Codex loads its cached plugin copy. For a local content update,
run `codex plugin add weft-bench@weft-tools --json` again. The local installer can
refresh the existing version; a version bump is not required for this development
workflow. Verify the files at the returned `installedPath`, then invoke its
`skills/benchmark/scripts/bench.py --plugin-info` and `verify --help` in the selected
environment. Checking only the source checkout does not verify installation.
Editable installation keeps the CLI tied to the bench checkout independently of
the plugin instructions. Reinstalling the plugin does not select a different
environment in the machine's `codex.local.json`.

The package layout follows the supported
[Codex plugin manifest and local marketplace format](https://developers.openai.com/plugins/build/plugins).

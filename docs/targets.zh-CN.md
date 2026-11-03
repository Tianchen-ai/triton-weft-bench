# 环境与目标配置

[English](targets.md) · [工具书](README.zh-CN.md)

以 `configs/example.json` 为起点，在外部配置中指定目标架构、允许的 CPU、
线程数、工具链、工作目录、参考库和实现工厂。选定 catalog 后复用其工作负载与绑定。

```json
{
  "targets": {
    "cpu": {
      "catalog": "x86-1t",
      "threads": 1,
      "deployment": {"cpus": [0], "workspace": "/work/bench-workers"},
      "implementations": {
        "weft": {
          "factory": "weft_bench.adapters.triton_jit:TritonJitAdapter",
          "python": "/opt/weft/bin/python",
          "toolchain": {}
        }
      }
    }
  }
}
```

补充 GGML/reference 路径后，通过 `--config` 或 `WEFT_BENCH_CONFIG` 选择。
CPU、SDK、模型等路径只保存在本地配置。SSH 目标还指定连接、部署目录和超时，
安装相同的目标侧产品与 Bench。

## JIT 与原生实现

`TritonJitAdapter` 在目标 Python 调用普通 JIT，记录编译、缓存和热调用。
`triton_native:TritonAdapter` 使用源码或 AOT artifact，
并通过 `triton_runtime.include/library` 选择执行侧 SDK。
源码绑定可包含 signature、constexprs、backend_options 与 toolchain；
带类型入口可以从声明推导 ABI。

`entry: "single"` 对应单实例；`entry: "grid"` 还需 grid 与线程数。
RISC-V 的实际 VLEN 在选定核心后探测；K3 A100 的注册需发生在能力查询前。
IME 与 IME2 使用各自指令合同和匹配的汇编器。

## 参考与模型

GGML 配置记录源码、库、compile database 与补丁，保持正常优化分派。
Clang 23 的 OpenMP 需提供匹配 ABI。
SpacemiT 参考保留实际 vendor 后端及 `GGML_SPACEMIT_BACKEND=1` 等配置，
CPU-RVV 作为另一条明确参考策略。

llama 配置指定 threads、GGUF、正常/集成二进制、SDK 和候选 package。
测量前检查实际构建命令、模型与替换绑定；计时和正确性使用相同选择。
详细字段与厂商汇编选项见本页[英文版本](targets.md)。

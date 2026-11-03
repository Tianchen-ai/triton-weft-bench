# 输入与实现接口

[English](https://tianchen-ai.github.io/triton-weft-bench/extending/) · [工具书](README.zh-CN.md)

为已有任务换内核，只需提供实现绑定；输入生成、数值参考、计时和传输均由 Bench 复用。

```python
from weft_bench import compare, Inputs, WeftKernel
from weft_bench.loading import load_config, select_target

target = select_target(load_config("/work/bench.local.json"), "cpu")
rows = compare(
    "rms_norm",
    candidate=WeftKernel(source="/work/norm.py", symbol="my_norm"),
    inputs=Inputs(shape={"M": 2, "N": 64}, seed=410),
    target=target,
)
```

ABI 指定参数顺序、shape、stride、Encoding 与读写效果。
参数名不同可提供 `arguments` 别名；
`WeftKernel(artifact="/work/artifact")` 测已有导出，
`baseline="onednn"` 等选择配置参考。

## 自定义输入

`Inputs` 接收 shape、seed、buffers 和 parameters。
JSON 中的相对文件路径以 JSON 所在目录为基准；
NumPy 数组、列表和原始文件使用工作负载声明的 dtype 与字节布局。
编码权重提供格式字节；BF16 数组使用 uint16 位存储。

`input/output/inout` 区分读写，状态任务在计时外 reset，
数值检查覆盖更新后的状态及输出。
完整投影的激活量化属于调用本身；packed leaf 的准备在其声明边界外。

## 模型与新适配器

`LlamaCpp(...).replace({...})` 与 `compare_model` 复用已有模型角色，
package 验证存储和 ABI，再由公共 SDK 加载。
正常后端选择、测量窗口与实际替换计数随记录保存。

外部原生适配器实现
`prepare_native(case, target, options, directory, executor, workload, name)`，
返回 `PreparedNative`：调用、include/link 标志和诊断。
共享 worker 继续负责输入、状态恢复、数值检查与计时。
OpenBLAS/oneDNN 适配器提供实际实现参考。

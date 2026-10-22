"""Transport-free descriptions and native worker composition.

An implementation contributes preparation and a native call. Workloads own their
inputs, outputs and numerical oracle. Neither contributes a benchmark program.
"""

from dataclasses import dataclass, field
import json
import re


@dataclass(frozen=True)
class NativeWorkload:
    header: str
    setup: str
    buffers: dict[str, str]
    output_type: str = "float"
    output_size: str = "data.output_size"
    check: str = "data.check(output)"
    includes: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    before_call: str = ""
    after_call: str = ""
    correctness: str = "within-tolerance"
    scalars: dict[str, str] = field(default_factory=dict)
    data_type: str = ""
    reset: str = ""
    collect: str = ""
    snapshots: dict[str, str] = field(default_factory=dict)
    batch_buffers: dict[str, str] = field(default_factory=dict)
    strides: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeCall:
    name: str
    prepare: str
    invoke: str
    collect: str = ""
    reset: str = ""
    includes: tuple[str, ...] = ()
    snapshots: dict[str, str] = field(default_factory=dict)
    whole_workload: bool = False


@dataclass
class PreparedNative:
    """An implementation's call and build inputs for the shared native worker."""

    call: NativeCall
    link_flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    include_flags: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)


def worker_source(
    workload: NativeWorkload,
    calls: list[NativeCall],
    input_source: str,
    *,
    save_outputs=False,
    record_environment=False,
):
    includes = {workload.header, "worker.h", *workload.includes}
    for call in calls:
        includes.update(call.includes)
    lines = [
        *("#include " + json.dumps(h) for h in sorted(includes)),
        "int main() { try {",
        *( ["wb::describe_environment();"] if record_environment else
           ["#ifdef WB_EXPECTED_VLEN_BITS", "wb::riscv_vector_json();", "#endif"] ),
        input_source,
        workload.setup,
    ]
    for call in calls:
        label = json.dumps(call.name)
        snapshots = {**workload.snapshots, **call.snapshots} if save_outputs else {}
        if any(name == "Y" or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in snapshots):
            raise ValueError("snapshot names must be file-safe labels distinct from the primary Y output")
        export = "".join(
            "wb::save_buffer(" + json.dumps(call.name + "-" + name + ".bin") + "," + value + ");"
            for name, value in snapshots.items()
        )
        lines += [
            "{",
            f"std::vector<{workload.output_type}> output({workload.output_size});",
            call.prepare,
            "auto invoke = [&] { "
            + ("" if call.whole_workload else workload.before_call)
            + call.invoke
            + ("" if call.whole_workload else workload.after_call)
            + " };",
            "auto collect = [&] { " + call.collect + workload.collect + export + " };",
            "auto reset = [&] { " + workload.reset + call.reset + " };",
            "auto check = [&] { " + workload.check + "; };",
            f"wb::execute({label}, invoke, collect, reset, check, output, "
            + ("true" if call.reset or workload.reset else "false")
            + ", "
            + ("true" if save_outputs else "false")
            + ", "
            + json.dumps(workload.correctness)
            + ");",
            "}",
        ]
    lines += [
        'return 0; } catch(const std::exception &e) { std::fprintf(stderr,"prepare: %s\\n",e.what()); return 2; } }'
    ]
    return "\n".join(lines) + "\n"


def fixture_source(workload, ports, input_source):
    """Expose the existing input storage and oracle to a language-runtime caller."""
    if not workload.data_type or workload.before_call or workload.after_call or workload.reset or workload.collect:
        raise NotImplementedError("runtime calls require a whole-workload fixture with a declared data type")
    includes = {workload.header, "worker.h", *workload.includes}
    getters = []
    for port in ports:
        getters.append("if(std::strcmp(name," + json.dumps(port.name) + ")==0)return const_cast<void*>(static_cast<const void*>("
                       + workload.buffers[port.name] + "));")
    return "\n".join("#include " + json.dumps(h) for h in sorted(includes)) + f'''
#include <cstring>
#include <stdexcept>
static wb::InputSet make_inputs() {{ {input_source} return inputs; }}
struct Fixture {{
  wb::InputSet inputs;
  {workload.data_type} data;
  std::vector<{workload.output_type}> output;
  Fixture():inputs(make_inputs()),data(inputs),output({workload.output_size}) {{}}
}};
extern "C" void *wb_fixture_create(char *error,size_t capacity) {{
  try {{ return new Fixture(); }}
  catch(const std::exception &e) {{std::snprintf(error,capacity,"%s",e.what());return nullptr;}}
}}
extern "C" void wb_fixture_release(void *handle) {{delete static_cast<Fixture*>(handle);}}
extern "C" void *wb_fixture_buffer(void *handle,const char *name) {{
  auto &data=static_cast<Fixture*>(handle)->data;
  auto &output=static_cast<Fixture*>(handle)->output;
  {''.join(getters)}
  return nullptr;
}}
extern "C" int wb_fixture_check(void *handle,const void *values,size_t bytes,char *error,size_t capacity) {{
  try {{
    auto &data=static_cast<Fixture*>(handle)->data;
    auto &output=static_cast<Fixture*>(handle)->output;
    if(bytes!=output.size()*sizeof({workload.output_type}))throw std::runtime_error("output byte length mismatch");
    if(values!=output.data())std::memcpy(output.data(),values,bytes);
    {workload.check};
    return 0;
  }} catch(const std::exception &e) {{std::snprintf(error,capacity,"%s",e.what());return 1;}}
}}
'''

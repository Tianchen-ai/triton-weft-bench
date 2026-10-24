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
    pointer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


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


def fixture_source(workload, call, metadata, input_source, *, save_outputs=False, eviction_bytes=0):
    """Borrow the native workload, including traversal, aliases and its oracle.

    The session callback keeps ordinary C++ automatic storage alive on the same
    execution thread. Each invocation calls the language runtime with addresses
    from the existing binder; this bridge never compiles or runs a kernel itself.
    """
    if type(eviction_bytes) is not int or eviction_bytes < 0:
        raise ValueError("fixture eviction_bytes must be a nonnegative integer")
    includes = {workload.header, "worker.h", "timer.h", *workload.includes, *call.includes}
    parameters, values = [], []
    for index, argument in enumerate(metadata["arguments"]):
        variable = "argument_" + str(index)
        parameters.append(argument["c_type"] + " " + variable)
        values.append("const_cast<void*>(static_cast<const void*>(" + variable + "))"
                      if argument["kind"] in ("view", "pointer") else "&" + variable)
    extras = len(metadata["shape_parameters"]) + sum(
        len(a["shape"]) for a in metadata["arguments"] if a.get("strided"))
    parameters += ["size_t extra_" + str(i) for i in range(extras)]
    invoke = (("" if call.whole_workload else workload.before_call) + call.invoke
              + ("" if call.whole_workload else workload.after_call))
    snapshots = {"Y": "output", **workload.snapshots, **call.snapshots} if save_outputs else {}
    export = "".join("wb::save_buffer(" + json.dumps(name + ".bin") + "," + value + ");"
                     for name, value in snapshots.items())
    return "\n".join("#include " + json.dumps(h) for h in sorted(includes)) + f'''
#include <functional>
#include <stdexcept>
using Launch = int (*)(void **);
using Session = int (*)(void *);
struct Fixture {{
  std::function<void(Launch)> invoke;
  std::function<void()> reset, check, save, evict;
}};
extern "C" int wb_fixture_invoke(void *handle,Launch launch,char *error,size_t capacity) {{
  try {{ static_cast<Fixture*>(handle)->invoke(launch); return 0; }}
  catch(const std::exception &e) {{std::snprintf(error,capacity,"%s",e.what());return 1;}}
}}
extern "C" int wb_fixture_action(void *handle,int action,char *error,size_t capacity) {{
  try {{
    auto &fixture=*static_cast<Fixture*>(handle);
    if(action==0)fixture.reset();
    else if(action==1)fixture.check();
    else if(action==2)fixture.save();
    else if(action==3)fixture.evict();
    else throw std::runtime_error("unknown fixture action");
    return 0;
  }} catch(const std::exception &e) {{std::snprintf(error,capacity,"%s",e.what());return 1;}}
}}
extern "C" int wb_fixture_session(Session session,char *error,size_t capacity) {{
  try {{
    {input_source}
    {workload.setup}
    std::vector<{workload.output_type}> output({workload.output_size});
    {call.prepare}
    Fixture fixture;
    std::vector<std::uint8_t> eviction({eviction_bytes}, 1);
    fixture.invoke=[&](Launch launch) {{
      auto jit_entry=[&]({', '.join(parameters)}) {{
        void *arguments[]={{ {', '.join(values)} }};
        if(launch(arguments))throw std::runtime_error("Triton launch failed");
      }};
      {invoke}
    }};
    fixture.reset=[&] {{ {workload.reset}{call.reset} }};
    fixture.check=[&] {{ {call.collect}{workload.collect}{workload.check}; }};
    fixture.save=[&] {{ {export} }};
    fixture.evict=[&] {{ wb::evict(eviction); }};
    return session(&fixture);
  }} catch(const std::exception &e) {{std::snprintf(error,capacity,"%s",e.what());return 1;}}
}}
'''

"""TileLang's own source/JIT/cache protocol for the existing fixture session."""

from dataclasses import asdict
import logging
import time


class CompilationEvents(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []
        self.started = None

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("TileLang JIT phase start: cache.compile"):
            self.started = time.perf_counter_ns()
        elif message.startswith("TileLang JIT phase done: cache.compile"):
            self.events.append({"cache_hit": False, "ir_initialization_us": 0,
                                "lowering_stages_us": [["tilelang.compile", (time.perf_counter_ns() - self.started) / 1000]],
                                "store_results_us": 0})


class TileLangFrontend:
    def __init__(self, factory, request):
        import tilelang
        from tilelang.backend.module import create_backend_context
        from tilelang.weft.target import descriptor, target_from_descriptor

        self.factory, self.request = factory, request
        self.jit = getattr(factory, "jit_impl", factory)
        self.context = create_backend_context("weft")
        if request["toolchain"]:
            self.context = create_backend_context(target_from_descriptor(
                descriptor(self.context.target), toolchain=request["toolchain"]))
        self.target = asdict(descriptor(self.context.target))
        self.package, self.version = tilelang.__file__, tilelang.__version__
        self.compiled = None
        self.argument_order = [arg["name"] for arg in request.get("invocation", []) if not arg.get("shape_parameter")]
        self.jit.target = self.context.target
        self.jit.pass_configs = {**(self.jit.pass_configs or {}),
                                     "tl.weft.backend_options": self.request["backend_options"]}
        self.observer = CompilationEvents()
        logging.getLogger("tilelang.jit.diagnostics").addHandler(self.observer)
        logging.getLogger("tilelang.jit.diagnostics").setLevel(logging.INFO)

    def describe(self):
        import tilelang
        from weft_core import compiler
        from tilelang.weft.frontend import import_module

        function = self.jit.get_tir(**self.description_arguments())
        self.argument_order = [function.buffer_map[p].name if p in function.buffer_map else p.name for p in function.params]
        module, imported = import_module(tilelang.tvm.IRModule({"main": function}))
        interface = compiler.interface(module)["interface"]
        interface["arguments"] = [arg for arg in interface["arguments"] if not arg["name"].startswith("__triton_")]
        buffers = {buffer.name: buffer for buffer in function.buffer_map.values()}
        implicit = {var.name for var in imported.implicit}
        for arg in interface["arguments"]:
            if arg["name"] in implicit:
                arg["shape_parameter"] = True
            storage = imported.storage_contracts.get(arg["name"], [])
            if storage:
                if len(storage) != 1:
                    raise ValueError("Bench requires one explicit logical storage contract per measured buffer")
                arg["logical_view"] = storage[0]
            if arg["name"] in buffers:
                buffer = buffers[arg["name"]]
                from tilelang.jit.adapter.weft_native import expression_functor
                from tilelang.weft.frontend.importer import dtype

                dimensions = {**self.request["shape"], **self.request["parameters"], **self.request["constexprs"]}
                arg["tensor_shape"] = [int(expression_functor(n)(dimensions)) for n in buffer.shape]
                arg["tensor_dtype"] = dtype(buffer.dtype).name
        return interface, imported

    def description_arguments(self):
        from tilelang.weft import language as T
        from tvm import tirx

        arguments = dict(self.request["constexprs"])
        ports = {port["name"]: port for port in self.request["ports"]}
        aliases = self.request["arguments"]
        names = {"f32": "float32", "f64": "float64", "f16": "float16", "bf16": "bfloat16",
                 "i1": "bool", **{p + str(b): n + str(b) for p, n in (("i", "int"), ("u", "uint")) for b in (8, 16, 32, 64)}}
        for name, annotation in self.jit.func.tensor_args.items():
            if name in arguments:
                continue
            role = aliases.get(name, name)
            if isinstance(annotation, tirx.Var) and str(annotation.dtype) != "handle":
                values = {**self.request["shape"], **self.request["parameters"]}
                if role not in values:
                    raise ValueError(f"runtime scalar {name} has no workload parameter {role}")
                arguments[name] = values[role]
            else:
                explicit = self.request["signature"].get(name)
                if explicit is not None:
                    if not isinstance(explicit, dict):
                        raise ValueError("TileLang eager signature entries describe shape, dtype and optional strides")
                    shape, dtype = explicit["shape"], explicit["dtype"]
                    strides = explicit.get("strides")
                else:
                    if role not in ports or ports[role]["dtype"] not in names:
                        raise ValueError(f"eager buffer {name} needs an explicit tensor signature")
                    port = ports[role]
                    shape, dtype, strides = port["shape"], names[port["dtype"]], port.get("strides")
                arguments[name] = T.StridedTensor(tuple(shape), tuple(strides), dtype) if strides else T.Tensor(tuple(shape), dtype)
        return arguments

    def buffer(self, storage, spec):
        import numpy as np
        import torch

        dtype = {"f32": np.float32, "f64": np.float64, "f16": np.float16,
                 "i1": np.bool_, "u1": np.bool_,
                 **{p + str(bits): getattr(np, name + str(bits)) for p, name in (("i", "int"), ("u", "uint"))
                    for bits in (8, 16, 32, 64)}}
        storage_dtype = spec.get("tensor_dtype", spec["dtype"])
        if storage_dtype == "bf16":
            value = torch.frombuffer(storage, dtype=torch.bfloat16)
        else:
            value = torch.from_numpy(np.frombuffer(storage, dtype=dtype[storage_dtype]))
        shape = spec["shape"]
        strides = spec.get("strides")
        return value.as_strided(shape, strides) if strides else value.reshape(shape)

    def call(self, bound, grid, options):
        eager = bool(self.jit.func.tensor_args)
        source_bound = {name: value for name, value in bound.items() if name in self.jit.signature.parameters}
        if hasattr(self.factory, "jit_impl"):
            from tilelang.autotuner import set_autotune_inputs
            with set_autotune_inputs(*(bound[name] for name in self.argument_order)):
                result = self.factory(**self.request["constexprs"], **source_bound) if eager else self.factory(**self.request["constexprs"])
        else:
            result = self.factory(**self.request["constexprs"], **source_bound) if eager else self.factory(**self.request["constexprs"])
        if eager:
            cache = self.factory._tuner_cache if hasattr(self.factory, "jit_impl") else self.jit._kernel_cache
            kernels = [item[0] if isinstance(item, tuple) else item for item in cache.values()]
            if len(kernels) != 1:
                raise ValueError("one native comparison must retain one TileLang specialization")
            kernel = kernels[0]
        else:
            kernel = result
        if kernel.execution_backend != "weft_native":
            raise ValueError("TileLang JIT did not select the shared Weft native compiler")
        dimensions = {**self.request["shape"], **self.request["parameters"]}
        call_bound = {name: bound[name] if name in bound else dimensions[self.request["arguments"].get(name, name)]
                      for name in kernel.adapter.names}
        if kernel is not self.compiled:
            _, symbols = kernel.adapter.bind_arguments(**call_bound)
            actual_grid = tuple(int(dim(symbols)) for dim in kernel.adapter.grid) or (1,)
            if not hasattr(self.factory, "jit_impl") and tuple(grid) != actual_grid:
                raise ValueError(f"TileLang source grid {actual_grid} differs from measurement grid {grid}")
            self.grid = actual_grid
            actual_threads = kernel.adapter.native.metadata.num_cpu_threads
            if actual_threads and actual_threads != self.request["num_cpu_threads"]:
                raise ValueError("TileLang source thread count differs from the measurement contract")
        if not eager:
            kernel(**call_bound)
        self.compiled = kernel
        return kernel

    def events(self):
        return self.observer.events

    def export(self, kernel, directory):
        return kernel.export_native(directory)

    def metadata(self, kernel):
        return kernel.adapter.native.metadata

    def cache_evidence(self):
        if not self.observer.events:
            if not getattr(self.compiled, "_tilelang_cache_path", None):
                raise RuntimeError("TileLang returned no disk-cache identity")
            self.observer.events.append({"cache_hit": True, "ir_initialization_us": 0,
                                         "lowering_stages_us": [], "store_results_us": 0})

    def aot(self, directory, calls, grid):
        from weft_core.artifact import NativeArtifact
        loaded = NativeArtifact.load(directory)
        for bound in calls:
            dimensions = {**self.request["shape"], **self.request["parameters"]}
            supplied, symbols = self.compiled.adapter.bind_arguments(**{
                name: bound[name] if name in bound else dimensions[self.request["arguments"].get(name, name)]
                for name in self.compiled.adapter.names})
            native = self.compiled.adapter.native_arguments(supplied, symbols)
            loaded.launch(self.grid, [native[name] for name in loaded.metadata.launch_args])

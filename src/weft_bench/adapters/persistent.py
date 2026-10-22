"""Bind explicit read-only artifact layouts before the operator interval."""

from dataclasses import replace
import json
import math


def persistent_inputs(case, workload, abi, names, aliases, providers=None):
    if len(names) != len(set(names)):
        raise ValueError("persistent_inputs contains duplicate arguments")
    arguments = {arg["name"]: arg for arg in abi["arguments"]}
    dimensions = {**abi.get("bindings", {}), **case.shape}
    buffers = dict(workload.buffers)
    setup, bases, details = [], {}, []
    providers = providers or {}
    if set(providers) - set(names):
        raise ValueError("persistent providers must name declared persistent inputs")
    for name in names:
        if name not in arguments:
            raise ValueError("unknown persistent input " + name)
        arg = arguments[name]
        role = aliases.get(name, name)
        if arg["kind"] != "view" or arg["access"] != "read" or role not in buffers:
            raise ValueError("persistent packing requires a read-only workload View")
        if case.family not in ("projection", "amx_tile") or role != "W":
            raise NotImplementedError("this workload has no declared persistent preparation interval for " + role)
        provider = providers.get(name)
        if provider is not None:
            layouts = {"q4_0": ("AMXQ4_0Rows[16]", 32, 18, "Q4_0"),
                       "q4_k": ("AMXQ4_KRows[16]", 256, 148, "Q4_K"),
                       "q6_k": ("AMXQ6_KRows[16]", 256, 210, "Q6_K")}
            weight = case.data.get("format")
            if weight not in layouts or provider != "ggml-amx-" + weight:
                raise NotImplementedError("unknown persistent framework storage provider")
            encoding, elements, storage, base = layouts[weight]
            shape = [int(n) if str(n).isdecimal() else dimensions[n] for n in arg["shape"]]
            if (shape != [dimensions["N"], dimensions["K"]] or shape[0] % 16 or shape[1] % elements or
                arg["encoding"] != encoding or arg["record_elements"] != elements or
                arg["storage_bytes"] != storage or arg["interleave_rows"] != 16):
                raise ValueError(f"artifact does not declare the GGML AMX {base} weight storage contract")
            prefix = "persistent_" + str(len(setup))
            setup.append(f"ProjectionReference {prefix}_owner(data.weight_type,data.M,data.N,data.K,"
                         "data.weights.data(),data.activation.data(),output);"
                         f"if({prefix}_owner.weight_buffer_name!=\"AMX\")"
                         f'throw std::runtime_error("normal GGML selected weight storage "+{prefix}_owner.weight_buffer_name+", expected AMX");'
                         f"auto {prefix}=static_cast<uint8_t*>({prefix}_owner.result->src[0]->data);")
            buffers[role] = prefix
            bases[name] = base
            details.append({"argument": name, "workload_port": role, "base_encoding": base,
                            "encoding": arg["encoding"], "shape": shape,
                            "provider": "ggml-model-default/AMX", "preparation": "persistent-before-timing"})
            continue
        storage = arg.get("storage_layout")
        if storage and arg.get("interleave_rows") and not arg.get("artifact_pack"):
            # Canonical layout analysis exposes the author's exact byte mapping
            # before JIT. Prepare those bytes without precompiling the consumer.
            shape = [int(n) if str(n).isdecimal() else dimensions[n] for n in arg["shape"]]
            rows, elements, record = arg["interleave_rows"], arg["record_elements"], arg["storage_bytes"]
            if len(shape) != 2 or shape[0] % rows or shape[1] % elements:
                raise ValueError("persistent JIT input requires complete records and cohorts")
            size = math.prod(shape) // elements * record
            alignment = math.lcm(64, int(arg["alignment"]))
            prefix = "persistent_" + str(len(setup))
            source = buffers[role]
            setup.append(f"std::vector<uint8_t> {prefix}_storage({size}+{alignment}-1);"
                         f"auto {prefix}=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>"
                         f"({prefix}_storage.data())+{alignment}-1)/{alignment}*{alignment});")
            chunks = storage["spans"]
            covered = 0
            for span in chunks:
                offset, count, chunk = span["offset"], span["bytes"], span["chunk"]
                if offset != covered or chunk <= 0 or count % chunk:
                    raise ValueError("Canonical interleave spans must cover the record exactly")
                covered += count
                columns = shape[1] // elements
                setup.append(f"for(size_t cohort=0;cohort<{shape[0] // rows};++cohort)"
                             f"for(size_t block=0;block<{columns};++block)"
                             f"for(size_t byte={offset};byte<{offset + count};byte+={chunk})"
                             f"for(size_t row=0;row<{rows};++row)"
                             f"std::memcpy({prefix}+(cohort*{columns}+block)*{record * rows}"
                             f"+{offset * rows}+(byte-{offset})*{rows}+row*{chunk},"
                             f"reinterpret_cast<const uint8_t*>({source})+((cohort*{rows}+row)*{columns}+block)*{record}+byte,{chunk});")
            if covered != record:
                raise ValueError("Canonical interleave does not cover the storage record")
            buffers[role] = prefix
            bases[name] = storage["base_encoding"]
            details.append({"argument": name, "encoding": arg["encoding"], "shape": shape,
                            "bytes": size, "provider": "author-declared-interleave",
                            "preparation": "persistent-before-JIT-call"})
            continue
        pack = arg.get("artifact_pack")
        if not pack or not arg.get("interleave_rows"):
            raise ValueError("persistent input lacks an exported artifact pack contract")
        for key in ("size_symbol", "pack_symbol"):
            if not pack[key].isascii() or not pack[key].isidentifier():
                raise ValueError("invalid artifact pack symbol")
        shape = [int(n) if str(n).isdecimal() else dimensions[n] for n in arg["shape"]]
        if len(shape) != 2 or shape[0] % arg["interleave_rows"] or shape[1] % arg["record_elements"]:
            raise ValueError("persistent input shape does not satisfy its complete-record/cohort layout")
        expected_bytes = math.prod(shape) // arg["record_elements"] * arg["storage_bytes"]
        allocation_alignment = math.lcm(64, int(arg["alignment"]))
        index = len(setup)
        prefix = "persistent_" + str(index)
        source = buffers[role]
        setup.append(
            f"auto {prefix}_size=native.symbol<size_t(*)(size_t,size_t)>({json.dumps(pack['size_symbol'])});"
            f"auto {prefix}_pack=native.symbol<void(*)(const void*,void*,size_t,size_t)>({json.dumps(pack['pack_symbol'])});"
            f"if({prefix}_size({shape[0]},{shape[1]})!={expected_bytes})"
            'throw std::runtime_error("artifact pack size disagrees with its typed layout");'
            f"std::vector<uint8_t> {prefix}_storage({expected_bytes}+{allocation_alignment}-1);"
            f"auto {prefix}=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>({prefix}_storage.data())"
            f"+{allocation_alignment}-1)/{allocation_alignment}*{allocation_alignment});"
            f"{prefix}_pack({source},{prefix},{shape[0]},{shape[1]});"
        )
        buffers[role] = prefix
        bases[name] = pack["base_encoding"]
        details.append({"argument": name, "workload_port": role, "base_encoding": pack["base_encoding"],
                        "encoding": arg["encoding"], "shape": shape, "bytes": expected_bytes,
                        "required_alignment": arg["alignment"], "allocation_alignment": allocation_alignment,
                        "pack_symbol": pack["pack_symbol"], "preparation": "persistent-before-timing",
                        "provider": "candidate-exported-native-builder"})
    return replace(workload, buffers=buffers), "".join(setup), bases, details

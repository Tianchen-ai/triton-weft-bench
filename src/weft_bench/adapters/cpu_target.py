"""CPU compiler descriptors, independent of the measurement machine contract."""


def target_architecture(settings):
    architecture = settings.get("architecture")
    if architecture is not None:
        if "arch" in settings and settings["arch"] != architecture:
            raise ValueError("compiler target arch and architecture disagree")
        return architecture
    if "arch" in settings:
        return settings["arch"]
    # Existing RVV configurations predate the explicit architecture field.
    if str(settings.get("march", "")).startswith("rv64"):
        return "riscv64"
    raise ValueError("CPU target requires an explicit arch or architecture")


def compiler_target(settings):
    values = dict(settings)
    values["arch"] = target_architecture(values)
    values.pop("architecture", None)
    values.setdefault("backend", "weft")
    fields = {"backend", "arch", "march", "abi", "vlen_bits"}
    optional = {"matrix_extension"}
    missing, unknown = fields - values.keys(), values.keys() - fields - optional
    if missing or unknown:
        raise ValueError(f"compiler target needs {sorted(missing)}; unknown fields {sorted(unknown)}; "
                         "set implementation.compile_target separately from the measurement target")
    if values["backend"] != "weft":
        raise ValueError("Triton CPU adapter requires the weft backend")
    if any(not isinstance(values[key], str) or not values[key] for key in fields - {"vlen_bits"}):
        raise ValueError("compiler target backend, arch, march and abi must be nonempty strings")
    if type(values["vlen_bits"]) is not int or values["vlen_bits"] < 0:
        raise ValueError("compiler target vlen_bits must be a nonnegative integer")
    values.setdefault("matrix_extension", "none")
    if not isinstance(values["matrix_extension"], str) or not values["matrix_extension"]:
        raise ValueError("compiler target matrix_extension must be a nonempty name")
    return values


def selected_target(options, execution_target):
    return compiler_target(options.get("compile_target", execution_target))


def check_jit_target(architecture, expected, detected):
    """JIT uses the product's native detection; configuration is an expectation."""
    actual = compiler_target(detected)
    if architecture is not None and actual["arch"] != architecture:
        raise ValueError(f"native JIT architecture {actual['arch']!r} differs from requested {architecture!r}")
    if expected is not None:
        wanted = compiler_target(expected)
        for key in ("arch", "abi", "vlen_bits"):
            if actual[key] != wanted[key]:
                raise ValueError(f"native JIT {key}: detected {actual[key]!r}, required {wanted[key]!r}")
        if wanted["matrix_extension"] != "none" and actual["matrix_extension"] != wanted["matrix_extension"]:
            raise ValueError("selected execution cores lack the requested matrix extension")
        from triton.backends.compiler import CPUTarget
        from triton.backends.weft.target import target_capabilities
        required = target_capabilities(CPUTarget(**wanted), matrix_extension=wanted["matrix_extension"])["isa_extensions"]
        available = target_capabilities(CPUTarget(**actual), matrix_extension=actual["matrix_extension"])["isa_extensions"]
        missing = ([name for name, version in required.items()
                    if name not in available or tuple(available[name]) < tuple(version)]
                   if actual["arch"] == "riscv64" else sorted(set(required) - set(available)))
        if missing:
            raise ValueError(f"selected execution cores lack required ISA versions: {missing}")
    return actual

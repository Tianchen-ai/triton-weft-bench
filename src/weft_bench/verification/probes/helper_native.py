"""Observe helper boundaries in actual compiler output, without rebuilding it."""
import re


def called_noinline_functions(source, language):
    if language == "llvm-ir":
        attributes = dict(re.findall(r"^attributes\s+(#\d+)\s*=\s*\{([^\n]*)\}", source, re.MULTILINE))
        definitions = re.finditer(
            r'^\s*define\b[^\n]*?@("[^"]+"|[-a-zA-Z$._0-9]+)\([^\n]*\)([^\n]*)\{',
            source, re.MULTILINE)
        called = []
        for match in definitions:
            symbol, suffix = match.groups()
            attrs = suffix + " " + " ".join(attributes.get(group, "") for group in re.findall(r"#\d+", suffix))
            if re.search(r"\bnoinline\b", attrs) and re.search(r"\bcall\b[^\n]*@" + re.escape(symbol) + r"\(", source):
                called.append(symbol)
        return called
    if language == "intrinsic-c":
        definitions = re.findall(r"static\s+__attribute__\(\(noinline\)\)\s+void\s+(\w+)\s*\(", source)
        return [name for name in definitions if re.search(r"^\s*" + re.escape(name) + r"\(", source, re.MULTILINE)]
    raise ValueError("unknown helper native source language " + language)


def capture(compiled, root, label, minimum_noinline=0):
    arch = compiled.metadata.target.arch
    if arch == "x86_64":
        physical, native, language = "x86ir", "llir", "llvm-ir"
    elif arch == "riscv64":
        physical, native, language = "riscvir", "c", "intrinsic-c"
    else:
        raise ValueError("helper observation does not support " + arch)
    paths = {}
    for stage in ("weftir", physical, native):
        source = compiled.asm[stage]
        assert source, "missing compiler output for " + stage
        path = root / (label + "." + stage)
        path.write_text(source)
        paths[stage] = str(path)
    assert compiled.asm["so"], "native artifact was not compiled"
    functions = called_noinline_functions(compiled.asm[native], language)
    assert len(functions) >= minimum_noinline, "native source lost required noinline functions or calls"
    return {"architecture": arch, "stages": paths, "native_language": language,
            "called_noinline_functions": functions, "native_build": compiled.metadata.native_build}

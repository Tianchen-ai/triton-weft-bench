"""Map an explicit source inventory to cases and conservative static helper edges."""
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path

from .catalog import plan


def coverage(target, *, output=None):
    settings = target.settings.get("verification", {})
    if not settings.get("source_inventory") and not settings.get("inventory_roots"):
        raise ValueError("verification requires source_inventory or explicit inventory_roots")
    records = json.loads(Path(settings["source_inventory"]).read_text()) if settings.get("source_inventory") else []
    if settings.get("inventory_roots"):
        known = {(str(Path(row["source"]).resolve()), row["symbol"]): row for row in records}
        records = []
        for root in settings["inventory_roots"]:
            for source in sorted(Path(root).rglob("*.py")):
                tree = ast.parse(source.read_text(), filename=str(source))
                for function in tree.body:
                    if not isinstance(function, ast.FunctionDef):
                        continue
                    decorators = [ast.unparse(d) for d in function.decorator_list]
                    if not any(d == "triton.jit" or d.startswith("triton.jit(") for d in decorators):
                        continue
                    key = (str(source.resolve()), function.name)
                    records.append({**known.get(key, {}), "source": key[0], "symbol": key[1],
                                    "line": function.lineno, "decorators": decorators})
    roots = [Path(p) for p in settings.get("coverage_roots", [])]
    files = {Path(row["source"]).resolve() for row in records}
    functions = {}
    syntax = {}
    for file in files:
        tree = ast.parse(file.read_text(), filename=str(file))
        syntax[file] = tree
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[(file, node.name)] = node

    def module(path):
        candidates = (path.with_suffix(".py"), path / "__init__.py")
        return next((p.resolve() for p in candidates if p.is_file()), None)

    edges = defaultdict(set)
    for file, tree in syntax.items():
        imports, modules = {}, {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                search = [file.parent.parents[node.level-2] if node.level > 1 else file.parent] if node.level else roots
                base = next((base / (node.module or "").replace(".", "/") for base in search
                             if module(base / (node.module or "").replace(".", "/"))), None)
                if base is None:
                    continue
                origin = module(base)
                for alias in node.names:
                    name = alias.asname or alias.name
                    child = module(base / alias.name)
                    if child:
                        modules[name] = child
                    else:
                        imports[name] = (origin, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    path = next((module(root / alias.name.replace(".", "/")) for root in roots
                                 if module(root / alias.name.replace(".", "/"))), None)
                    if path:
                        modules[alias.asname or alias.name] = path
        for (owner, name), function in functions.items():
            if owner != file:
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                candidate = None
                if isinstance(node.func, ast.Name):
                    local = (file, node.func.id)
                    candidate = local if local in functions else imports.get(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    try:
                        spelling = ast.unparse(node.func.value)
                        if spelling in modules:
                            candidate = (modules[spelling], node.func.attr)
                    except (TypeError, ValueError):
                        pass
                if candidate in functions:
                    edges[(file, name)].add(candidate)
    entries = plan(target)
    direct = defaultdict(list)
    for entry in entries:
        source = entry.get("source_origin") or entry.get("compilation", {}).get("source") or entry.get("source")
        symbol = entry.get("symbol") or entry.get("options", {}).get("symbol")
        if source and symbol:
            direct[(Path(source).resolve(), symbol)].append(entry["id"])
        for covered in entry.get("covers", []):
            direct[(Path(covered["source"]).resolve(), covered["symbol"])].append(entry["id"])
    reachable = defaultdict(set)
    for root, cases in direct.items():
        seen, pending = set(), list(edges[root])
        while pending:
            child = pending.pop()
            if child in seen:
                continue
            seen.add(child)
            reachable[child].update(cases)
            pending.extend(edges[child])
    ledger = {}
    if output and (Path(output) / "verification.json").exists():
        ledger = json.loads((Path(output) / "verification.json").read_text())["entries"]
    rows = []
    for record in records:
        key = (Path(record["source"]).resolve(), record["symbol"])
        cases = direct.get(key, [])
        callers = sorted(reachable.get(key, []))
        classification = "direct-case" if cases else "static-helper-reachability" if callers else "unmapped"
        rows.append({**record, "classification": classification, "cases": cases, "possible_callers": callers,
                     "case_statuses": {case: ledger.get(case, {}).get("status", "not_run") for case in cases or callers},
                     "line": functions[key].lineno if key in functions else record.get("line")})
    result = {"target": target.name, "counts": dict(Counter(r["classification"] for r in rows)),
              "note": "Static helper reachability is not proof that a measured invocation executed a helper.",
              "sources": rows}
    if output:
        path = Path(output) / "coverage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result

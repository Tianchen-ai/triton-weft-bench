"""Small NumPy scenarios for the public Weft Program catalog.

The catalog implements the candidate; this module owns independent input and
output semantics. No compilation or device configuration belongs here.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    arguments: dict
    outputs: tuple
    oracle: object
    properties: tuple = ()
    calls: int = 1
    encodings: dict = field(default_factory=dict)
    exact_outputs: tuple = ()


def silu(x):
    return x * np.exp(-np.logaddexp(0.0, -x))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def prepare(name, bindings=None, seed=410):
    """One legal, fully checked scenario; required source dimensions are honored."""
    bindings = bindings or {}
    if name.startswith("dense/"):
        if __package__:
            from .dense_oracles import prepare as dense_prepare
        else:
            from dense_oracles import prepare as dense_prepare
        return dense_prepare(name.split("/", 1)[1], bindings, seed, Scenario)
    if name.startswith("encoded/"):
        if __package__:
            from .encoded_oracles import prepare as encoded_prepare
        else:
            from encoded_oracles import prepare as encoded_prepare
        return encoded_prepare(name.split("/", 1)[1], bindings, seed, Scenario)
    if name.startswith("examples/"):
        if __package__:
            from .example_oracles import prepare as example_prepare
        else:
            from example_oracles import prepare as example_prepare
        return example_prepare(name.split("/", 1)[1], bindings, seed, Scenario)
    rng = np.random.default_rng(seed)

    def f(*shape):
        return rng.uniform(-0.5, 0.5, shape).astype(np.float32)

    def output(*shape, dtype=np.float32):
        return np.full(shape, np.nan if dtype == np.float32 else -1, dtype=dtype)

    if name == "semantics_composition":
        if __package__:
            from ..verification.probes.model_semantics import composition_reference
        else:
            from model_semantics import composition_reference
        m, n, steps = (bindings[key] for key in ("M", "N", "STEPS"))
        rows, block = bindings["ROWS"], bindings["BLOCK"]
        x = rng.integers(-5000, 5001, (m, n), dtype=np.int32)
        initial = rng.integers(-32768, 32768, m, dtype=np.int32)
        def reference(values):
            changed, result, _ = composition_reference(values["X"], values["Init"], steps, rows, block)
            return {"X": changed, "Y": result}
        return Scenario({"X": x, "Init": initial, "Y": output(m, 3, dtype=np.int32), "STEPS": steps},
            ("X", "Y"), reference, ("level-initialization", "staged-snapshot", "alias-write",
            "coupled-state", "handoff", "zero-tail", "exact-integer"))

    if name == "lexical_helper_extent":
        parent, count = bindings["M"], bindings["COUNT"]
        return Scenario({"X": np.zeros(parent, np.int32), "Y": output(2, dtype=np.int32), "COUNT": count},
            ("Y",), lambda v: {"Y": np.asarray([(count + 3) // 4, (count // 2 + 3) // 4], np.int32)},
            ("lexical-helper-scope", "dynamic-extent", "exact-integer"))

    if name == "composed_windows":
        count = bindings["COUNT"]
        def reference(values):
            result = values["Y"].copy()
            result[:count] = values["X"][:count] + 1
            return {"Y": result}
        return Scenario({"Padding": np.zeros(4, np.int32), "X": np.arange(31, dtype=np.int32),
            "Y": np.full(31, -777, np.int32), "COUNT": count}, ("Y",), reference,
            ("composed-views", "helper", "preserved-tail", "exact-integer"))

    if name == "static_births":
        source = np.arange(7, dtype=np.float32)
        return Scenario({"X": source, "Y": output(7)}, ("X", "Y"),
            lambda v: {"X": v["X"] + 1, "Y": v["X"].copy() if bindings["SHARED"] else v["X"] * 2},
            ("constexpr-birth", "snapshot-before-alias-write"), exact_outputs=("X", "Y"))

    if name == "static_window_copy":
        return Scenario({"X": np.arange(16, dtype=np.int32), "Y": output(16, dtype=np.int32)},
            ("Y",), lambda v: {"Y": v["X"] + 1}, ("exact-static-window", "helper"))

    if name == "encoded_group_reshape":
        groups=bindings.get("GROUPS",2)
        record=np.dtype([("d",np.float16),("q",np.int8,(32,))])
        w,x=np.zeros(1,record),np.zeros(1,record)
        w["d"],x["d"]=1,1
        w["q"]=rng.integers(-128,128,(1,32),dtype=np.int16).astype(np.int8)
        x["q"]=np.arange(-16,16,dtype=np.int8)
        def reference(v):
            left=v["W"].view(record)["q"].astype(np.int64).reshape(groups,-1)
            right=v["X"].view(record)["q"].astype(np.int64).reshape(groups,-1)
            if bindings.get("BIASED",False):left=(left+128)%256
            return {"Y":(left*right).sum(axis=1).astype(np.int32)}
        return Scenario({"W":w.view(np.uint8),"X":x.view(np.uint8),"Y":output(groups,dtype=np.int32)},
                        ("Y",),reference,("encoded-reshape","distinct-group-offsets","exact-integer"),
                        encodings={"W":("ggml_formats.Q8_0",[32]),"X":("ggml_formats.Q8_0",[32])})

    if name == "grouped_chain_boundary":
        m, k = bindings.get("M", 3), bindings.get("K", 35)
        x = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
        w = rng.integers(0, 256, (m, k), dtype=np.uint16).astype(np.uint8)
        if k >= 4:
            x[:, :2], x[:, 2:4], w[:, :4] = 127, -128, 255
        def reference(v):
            products = v["X"].astype(np.int64) * v["W"].astype(np.int64)
            total = np.zeros(m, np.int64)
            for start in range(0, k, 2):
                partial = products[:, start:start+2].sum(axis=1)
                if not bindings.get("FOLD", False):
                    partial = (partial + 32768) % 65536 - 32768
                total += partial
            return {"Y": total.astype(np.int32)}
        return Scenario({"X": x, "W": w, "Y": output(m, dtype=np.int32)},
                        ("Y",), reference, ("chained-groups", "empty-and-partial-groups", "exact-integer"))

    if name == "grouped_range_boundary":
        m, k = bindings.get("M", 3), bindings.get("K", 35)
        x = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
        w = rng.integers(0, 256, (m, k), dtype=np.uint16).astype(np.uint8)
        if k >= 8:
            x[:, :4], x[:, 4:8], w[:, :8] = 127, -128, 255
        def reference(v):
            weights = v["W"].astype(np.int64)
            if bindings.get("SAFE", False): weights &= 15
            products = v["X"].astype(np.int64) * weights
            total = np.zeros(m, np.int64)
            for start in range(0, k, 4):
                partial = products[:, start:start+4].sum(axis=1)
                total += (partial + 32768) % 65536 - 32768
            return {"Y": total.astype(np.int32)}
        return Scenario({"X": x, "W": w, "Y": output(m, dtype=np.int32)},
                        ("Y",), reference, ("narrow-group-wrap", "zero-and-tail", "exact-integer"))

    if "norm" in name:
        m, n = 3, bindings.get("N", 32)
        args = {"X": f(m, n), "Weight": f(n), "Y": output(m, n), "epsilon": 1e-5}
        if name.startswith("layer"):
            args["Bias"] = f(n)
        if name == "residual_rms_norm":
            args["Residual"] = f(m, n)

        def reference(v):
            x = v["X"].astype(np.float64)
            if "Residual" in v:
                x = x + v["Residual"]
            if "Bias" in v:
                x = x - x.mean(axis=-1, keepdims=True)
            y = x / np.sqrt((x*x).mean(axis=-1, keepdims=True) + v["epsilon"])
            return {"Y": y * v["Weight"] + v.get("Bias", 0)}
        return Scenario(args, ("Y",), reference, ("all-rows-and-output-elements",))

    unary = {"silu": silu, "softplus": lambda x: np.logaddexp(0.0, x),
             "logarithm": np.log, "exponential": np.exp, "copy": lambda x: x,
             "softmax": softmax, "softmax_row": softmax}
    if name in unary:
        n = bindings.get("N", 35)
        args = {"X": f(3, n), "Y": output(3, n)}
        if name == "logarithm":
            args["X"] = np.exp(args["X"] * 4).astype(np.float32)
        return Scenario(args, ("Y",), lambda v: {"Y": unary[name](v["X"].astype(np.float64))})
    if name == "swiglu":
        args = {"Gate": f(3, 35), "Up": f(3, 35), "Y": output(3, 35)}
        return Scenario(args, ("Y",), lambda v: {"Y": silu(v["Gate"].astype(np.float64)) * v["Up"]})
    if name == "add":
        args = {"X": f(3, 35), "Residual": f(3, 35), "Y": output(3, 35)}
        return Scenario(args, ("Y",), lambda v: {"Y": v["X"].astype(np.float64) + v["Residual"]})
    if name == "where":
        args = {"X": f(3, 35), "Other": f(3, 35), "Y": output(3, 35),
                "Condition": (np.arange(105).reshape(3, 35) % 3 == 0)}
        return Scenario(args, ("Y",), lambda v: {"Y": np.where(v["Condition"], v["X"], v["Other"])},
                        ("both-boolean-branches",))
    if name == "fill":
        return Scenario({"Y": output(3, 35), "value": 1.25}, ("Y",),
                        lambda v: {"Y": np.full_like(v["Y"], v["value"])})

    if name.startswith(("linear", "matmul", "batched_matmul")):
        m, n, k = 5, 19, 35
        prefix = (2,) if name == "batched_matmul" else ()
        transposed = name.startswith("linear")
        args = {"X": f(*prefix, m, k), "Weight": f(*prefix, *((n, k) if transposed else (k, n))),
                "Y": output(*prefix, m, n)}
        if "bias" in name:
            args["Bias"] = f(n)

        def reference(v):
            w = v["Weight"].astype(np.float64)
            y = v["X"].astype(np.float64) @ (w.swapaxes(-1, -2) if transposed else w)
            y = y + v.get("Bias", 0)
            return {"Y": silu(y) if "silu" in name else y}
        return Scenario(args, ("Y",), reference, ("complete-output", "partial-tiles"))

    if name.startswith("rope_"):
        t, h, p = 3, 2, 17
        shape = (t, h, 2, p) if name == "rope_split" else (t, h, p, 2)
        angle = f(t, p) * 3
        args = {"X": f(*shape), "Y": output(*shape),
                "Cos": np.cos(angle).astype(np.float32), "Sin": np.sin(angle).astype(np.float32)}

        def reference(v):
            x = v["X"].astype(np.float64)
            a, b = (x[:, :, 0], x[:, :, 1]) if name == "rope_split" else (x[..., 0], x[..., 1])
            c, s = v["Cos"][:, None, :], v["Sin"][:, None, :]
            return {"Y": np.stack((a*c-b*s, b*c+a*s), axis=2 if name == "rope_split" else -1)}
        return Scenario(args, ("Y",), reference, ("precomputed-coefficients", "partial-block"))
    if name == "kv_update":
        args = {"K": f(2, 3, 2, 35), "V": f(2, 3, 2, 35),
                "KeyCache": f(2, 7, 2, 35), "ValueCache": f(2, 7, 2, 35), "start": 2}

        def reference(v):
            result = {}
            for cache, new in (("KeyCache", "K"), ("ValueCache", "V")):
                result[cache] = v[cache].copy()
                result[cache][:, v["start"]:v["start"]+v[new].shape[1]] = v[new]
            return result
        return Scenario(args, ("KeyCache", "ValueCache"), reference, ("nonzero-start", "untouched-prefix-and-tail"))
    if name == "causal_attention":
        d = bindings.get("D", 16)
        args = {"Q": f(2, 3, 4, d), "K": f(2, 7, 2, d), "V": f(2, 7, 2, d),
                "Y": output(2, 3, 4, d), "start": 2, "scale": float(d**-0.5)}

        def reference(v):
            q, k, value = (v[x].astype(np.float64) for x in ("Q", "K", "V"))
            y = np.empty_like(q)
            for token in range(q.shape[1]):
                end = v["start"] + token + 1
                for head in range(q.shape[2]):
                    group = head // (q.shape[2] // k.shape[2])
                    scores = np.einsum("bd,bsd->bs", q[:, token, head], k[:, :end, group]) * v["scale"]
                    y[:, token, head] = np.einsum("bs,bsd->bd", softmax(scores), value[:, :end, group])
            return {"Y": y}
        return Scenario(args, ("Y",), reference, ("causal-prefix", "grouped-query-heads", "nonzero-start"))

    indices = np.array([[0, 2], [2, 0], [4, 2], [0, 2]], dtype=np.int32)
    counts = np.bincount(indices.ravel(), minlength=6)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    positions = np.empty_like(indices)
    cursor = offsets[:-1].copy()
    for token in range(4):
        for slot in range(2):
            expert = indices[token, slot]
            positions[token, slot] = cursor[expert]
            cursor[expert] += 1
    if name == "router":
        logits = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 1, -1, 2, 0],
                           [-3, 0, 2, 2, -1, 2], [0.5, -1, 2, -3, 1, 0]], dtype=np.float32)
        args = {"Logits": logits, "Indices": output(4, 2, dtype=np.int32), "Weights": output(4, 2)}

        def reference(v):
            selected = np.argsort(-v["Logits"], axis=-1, kind="stable")[:, :2].astype(np.int32)
            scores = np.take_along_axis(v["Logits"].astype(np.float64), selected, axis=-1)
            return {"Indices": selected, "Weights": softmax(scores)}
        return Scenario(args, ("Indices", "Weights"), reference, ("stable-topk-ties", "selected-score-softmax"))
    if name == "group_tokens":
        args = {"X": f(4, 35), "Indices": indices, "Offsets": output(7, dtype=np.int32),
                "Cursor": output(6, dtype=np.int32), "Positions": output(4, 2, dtype=np.int32), "Grouped": output(8, 35)}

        def reference(v):
            grouped = np.empty_like(v["Grouped"])
            for token in range(4):
                for slot in range(2):
                    grouped[positions[token, slot]] = v["X"][token]
            return {"Grouped": grouped, "Offsets": offsets, "Cursor": offsets[1:], "Positions": positions}
        return Scenario(args, ("Grouped", "Offsets", "Cursor", "Positions"), reference,
                        ("stable-group-order", "exclusive-offsets", "inverse-positions", "empty-experts"))
    if name == "expert_linear":
        args = {"X": f(8, 17), "Weight": f(6, 19, 17), "Offsets": offsets, "Y": output(8, 19)}

        def reference(v):
            y = np.empty_like(v["Y"], dtype=np.float64)
            for expert in range(v["Weight"].shape[0]):
                start, stop = v["Offsets"][expert:expert+2]
                y[start:stop] = v["X"][start:stop].astype(np.float64) @ v["Weight"][expert].astype(np.float64).T
            return {"Y": y}
        return Scenario(args, ("Y",), reference, ("empty-single-and-multiple-token-experts", "complete-output"))
    if name == "combine":
        args = {"Grouped": f(8, 35), "Positions": positions, "Weights": softmax(f(4, 2)).astype(np.float32), "Y": output(4, 35)}
        return Scenario(args, ("Y",), lambda v: {
            "Y": (v["Grouped"][v["Positions"]].astype(np.float64) * v["Weights"][..., None]).sum(axis=1)},
            ("inverse-token-order", "weighted-merge"))

    if name == "causal_conv1d":
        args = {"X": f(2, 3, 17), "Weight": f(17, 4), "Bias": f(17),
                "History": f(2, 4, 17), "Y": output(2, 3, 17)}

        def reference(v):
            history = v["History"].astype(np.float64).copy()
            y = np.empty_like(v["Y"], dtype=np.float64)
            for token in range(v["X"].shape[1]):
                history[:, :-1] = history[:, 1:].copy()
                history[:, -1] = v["X"][:, token]
                y[:, token] = silu((history * v["Weight"].T[None]).sum(axis=1) + v["Bias"])
            return {"Y": y, "History": history}
        return Scenario(args, ("Y", "History"), reference, ("nonzero-initial-history", "state-continuation"), calls=2)
    if name == "selective_scan":
        b, t, d, n = 2, 3, 17, bindings.get("N", 32)
        args = {"X": f(b, t, d), "Delta": f(b, t, d), "Bcoef": f(b, t, n), "Ccoef": f(b, t, n),
                "Alog": f(d, n), "Skip": f(d), "Gate": f(b, t, d), "Bias": f(d),
                "State": f(b, d, n), "Y": output(b, t, d)}

        def reference(v):
            a = {key: value.astype(np.float64) for key, value in v.items()}
            state, y = a["State"].copy(), np.empty_like(a["Y"])
            for token in range(t):
                dt = np.logaddexp(0.0, a["Delta"][:, token] + a["Bias"])
                x = a["X"][:, token]
                state = state * np.exp(-np.exp(a["Alog"])[None] * dt[..., None])
                state += a["Bcoef"][:, token, None, :] * (dt * x)[..., None]
                y[:, token] = ((state * a["Ccoef"][:, token, None, :]).sum(axis=-1) + a["Skip"] * x) * silu(a["Gate"][:, token])
            return {"Y": y, "State": state}
        return Scenario(args, ("Y", "State"), reference, ("complete-state-vector", "nonzero-initial-state", "state-continuation"), calls=2)
    raise NotImplementedError("No numerical scenario for Program " + name)

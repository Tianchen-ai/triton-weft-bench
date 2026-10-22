"""Input/oracle contracts for the public typed dense examples."""
import numpy as np


def prepare(name, bindings, seed, Scenario):
    rng = np.random.default_rng(seed)
    m, n, k = bindings.get("M", 3), bindings.get("N", 35), 7

    def f(*shape, dtype=np.float32):
        return rng.uniform(-0.5, 0.5, shape).astype(dtype)

    def out(*shape, dtype=np.float32):
        dtype = np.dtype(dtype)
        return np.full(shape, np.nan if dtype.kind == "f" else 0, dtype=dtype)

    if name.startswith("compare_"):
        dtype = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "u32": np.uint32}[name.rsplit("_", 1)[1]]
        values = [0, 1, 2, 3, 2**31, 2**32-1] if dtype == np.uint32 else [-9, -1, 0, 1, 3, 9]
        x = np.resize(np.array(values, dtype=dtype), 35)
        right = np.roll(x, 2)
        right[::4] = x[::4]
        args = {"X": x, "Y": out(6, x.size, dtype=dtype)}
        if "scalar" in name:
            args["pivot"] = int(3) if np.dtype(dtype).kind in "iu" else 1.0
        else:
            args["R"] = right

        def reference(v):
            left, right = (v["pivot"], v["X"]) if "pivot" in v else (v["X"], v["R"])
            predicates = (left == right, left != right, left < right, left <= right, left > right, left >= right)
            return {"Y": np.stack([np.where(p, left, right) for p in predicates]).astype(dtype)}
        return Scenario(args, ("Y",), reference, ("all-six-relations", "ties", "tail"), exact_outputs=("Y",))

    math_ops = {
        "exp_nonpositive": lambda x: np.exp(-np.abs(x)),
        "exp_clamped": lambda x: np.exp(np.clip(x, -87, 87)),
        "log_clamped": lambda x: np.log(np.clip(x, 2.0**-100, 2.0**100)),
        "log_exp_normalizer": lambda x: np.log1p(np.exp(-np.abs(x))),
        "selected_exp": lambda x: np.exp(np.clip(x, -5, 5)),
        "masked_exp": lambda x: np.exp(np.where((x >= -5) & (x <= 5), x, 0)),
        "complemented_exp": lambda x: np.exp(np.where((x >= -5) & (x <= 5), x, 0)),
    }
    if name in math_ops:
        values = [-100, -87, -6, -5, -1, 0, 1, 5, 6, 87, 100]
        if name == "log_clamped":
            values = [-1, 0, 2.0**-110, 2.0**-100, 0.5, 1, 2.0**100, 2.0**110]
        x = np.resize(np.array(values, dtype=np.float32), (m, n))
        return Scenario({"X": x, "Y": out(m, n)}, ("Y",),
                        lambda v: {"Y": math_ops[name](v["X"].astype(np.float64))}, ("range-boundaries", "tail"))

    choices = {
        "signed_choices": (np.int32, -7, False), "unsigned_choices": (np.uint32, 7, True),
        "double_choices": (np.float64, -7, True), "signed_byte_choices": (np.int8, -7, False),
        "unsigned_half_choices": (np.uint16, 7, True), "signed_wide_choices": (np.int64, -7, False),
        "unsigned_wide_choices": (np.uint64, 7, True),
    }
    if name in choices:
        dtype, replacement, greater = choices[name]
        kind = np.dtype(dtype).kind
        values = np.arange(11, dtype=np.int64)
        if kind != "u":
            values -= 5
        x = np.resize(values.astype(dtype), (m, n))
        if dtype == np.uint64:
            x[:, ::3] += np.uint64(2**63)
        elif dtype == np.int64:
            x[:, ::3] += np.int64(2**40)
        elif dtype == np.uint32:
            x[:, ::3] += np.uint32(2**31)
        r = np.roll(x, 2, axis=1)
        args = {"X": x, "R": r, "Y": out(m, n, dtype=dtype), "Z": out(m, n, dtype=dtype), "threshold": 2.0 if kind == "f" else 2}

        def reference(v):
            selected = (v["X"] >= v["threshold"]) | ((v["R"] > v["X"]) if greater else (v["R"] < v["X"]))
            return {"Y": np.where(selected, v["X"], v["R"]), "Z": np.where(selected, dtype(replacement), v["R"])}
        return Scenario(args, ("Y", "Z"), reference, ("both-predicate-branches", "typed-comparison"), exact_outputs=("Y", "Z"))
    if name in ("masked_affine", "threshold_indices", "logical_gate", "observed_predicate", "loaded_gate", "loaded_choices", "observed_gate"):
        args = {"X": np.resize(np.array([-2, -1, 0, 0.25, 1, 2], np.float32), (m, n))}
        if name == "threshold_indices":
            args.update(Ids=np.arange(m*n, dtype=np.int32).reshape(m, n), Scores=out(m, n),
                        SelectedIds=out(m, n, dtype=np.int32), threshold=0.25)
            return Scenario(args, ("Scores", "SelectedIds"), lambda v: {
                "Scores": np.where(v["X"] >= v["threshold"], v["X"], 0),
                "SelectedIds": np.where(v["X"] >= v["threshold"], v["Ids"], -1)})
        args["Y"] = out(m, n)
        if name == "observed_predicate":
            args["Predicate"] = out(m, n, dtype=np.bool_)
            return Scenario(args, ("Predicate", "Y"), lambda v: {"Predicate": v["X"] > 0, "Y": np.maximum(v["X"], 0)}, exact_outputs=("Y",))
        args["Z"] = out(m, n)
        if name == "masked_affine":
            args.update(Residual=f(m, n), lower=-1.0, upper=1.0, scale=0.75)
        elif name != "observed_gate":
            args["R"] = f(m, n)
        if name == "logical_gate":
            args["enabled"] = True
        if name.startswith("loaded") or name == "observed_gate":
            args["Mask"] = np.arange(m*n).reshape(m, n) % 3 == 0
        if name == "observed_gate":
            args["Selected"] = out(m, n, dtype=np.bool_)

        def reference(v):
            x = v["X"]
            if name == "masked_affine":
                mask = (x >= v["lower"]) & (x <= v["upper"])
                return {"Y": np.where(mask, x.astype(np.float64)*v["scale"]+v["Residual"], v["Residual"]), "Z": np.where(mask, x, 0)}
            if name == "logical_gate":
                mask = ((x > v["R"]) ^ (v["R"] > 0)) & v["enabled"]
                return {"Y": np.where(mask, x, v["R"]), "Z": np.where(~mask, x, v["R"])}
            if name == "observed_gate":
                mask = v["Mask"] | (x < 0)
                return {"Selected": mask, "Y": np.where(mask, x, 0), "Z": np.where(mask, 1, x)}
            mask = v["Mask"] & (x >= 0) if name == "loaded_gate" else v["Mask"]
            return {"Y": np.where(mask, x, v["R"]), "Z": np.where(mask, 1 if name == "loaded_gate" else -1, x)}
        return Scenario(args, tuple(k for k in ("Y", "Z", "Selected") if k in args), reference)

    state_names = ("state_update", "state_scale", "recurrent_accumulate", "coupled_recurrence", "branch_recurrence", "initialized_recurrence", "reset_recurrence", "parallel_carries")
    if name in state_names:
        if name in ("state_update", "state_scale"):
            args = {"State": f(m, n)}
            if name == "state_scale":
                args["scale"] = 0.75
                return Scenario(args, ("State",), lambda v: {"State": v["State"].astype(np.float64)*v["scale"]}, calls=2)
            args.update(Delta=f(m, n), decay=0.75)
            return Scenario(args, ("State",), lambda v: {"State": v["State"].astype(np.float64)*v["decay"]+v["Delta"]}, calls=2)
        shape = (m, n) if name == "recurrent_accumulate" else (n,)
        args = {"Y": out(*shape), "steps": 5}
        if name != "reset_recurrence":
            args["X"] = f(*shape)
        if name == "recurrent_accumulate":
            args["decay"] = 0.5
        if name in ("coupled_recurrence", "parallel_carries"):
            args["Z"] = out(*shape)
        if name == "parallel_carries":
            args.update(W=out(*shape), Initial=out(*shape))

        def reference(v):
            x = v.get("X", np.zeros_like(v["Y"])).astype(np.float64)
            if name == "reset_recurrence":
                return {"Y": np.full_like(v["Y"], 3 if v["steps"] else -1)}
            first = x.copy() if name in ("initialized_recurrence", "parallel_carries") else np.zeros_like(x)
            second = 2*x if name == "parallel_carries" else np.ones_like(x)
            third = x + 3
            for index in range(v["steps"]):
                if name == "parallel_carries":
                    if index % 3 == 0:
                        first, second, third = second, third, first
                    elif index % 3 == 1:
                        first, third = second.copy(), third+x
                    else:
                        first, third, second = third+x, second, first
                elif name == "coupled_recurrence":
                    first, second = second+x, first*0.5+second*0.25 if index % 2 == 0 else first-second*0.125
                elif name == "initialized_recurrence":
                    first = first*0.5+1
                else:
                    first = first+x if index % 2 == 0 else first*v.get("decay", 0.5)
            result = {"Y": first}
            if "Z" in v:
                result["Z"] = second
            if "W" in v:
                result.update(W=third, Initial=x)
            return result
        return Scenario(args, tuple(k for k in ("Y", "Z", "W", "Initial") if k in args), reference, ("ordered-state", "branch-carries"))

    if name in ("statistics", "transformed_statistics", "dynamic_statistics"):
        args = {"X": f(n) if name == "statistics" else f(m, n),
                "Y": out(3) if name == "statistics" else out(m, 3)}
        if name != "statistics":
            args.update(shift=0.25, scale=1.25)

        def reference(v):
            x = v["X"].astype(np.float64)
            if name != "statistics":
                x = ((x-v["shift"])*v["scale"])**2
            return {"Y": np.stack((x.sum(axis=-1), x.max(axis=-1), x.min(axis=-1)), axis=-1)}
        return Scenario(args, ("Y",), reference)
    if name in ("product_sum", "dot_f16", "mixed_dot_f16", "mixed_dot_bf16", "dot_f64"):
        dtype = np.float64 if name == "dot_f64" else np.float32
        x, weight = f(n, dtype=dtype), f(n, dtype=dtype)
        encodings = {}
        if name == "dot_f16":
            x = x.astype(np.float16)
        if name.endswith("f16") and "bf16" not in name:
            weight = weight.astype(np.float16)
        if name == "mixed_dot_bf16":
            weight = (weight.view(np.uint32) >> 16).astype(np.uint16)
            encodings["Weight"] = ("bf16", [n])
        args = {"X": x, "Weight": weight, "Y": out(1, dtype=dtype)}

        def reference(v):
            w = (v["Weight"].astype(np.uint32) << 16).view(np.float32) if name == "mixed_dot_bf16" else v["Weight"]
            return {"Y": np.array([np.dot(v["X"].astype(np.float64), w.astype(np.float64))])}
        return Scenario(args, ("Y",), reference, encodings=encodings)

    if name == "low_bit_projection":
        m, n, k = 3, 5, 64
        x = rng.integers(-8, 8, (m, k), dtype=np.int8)
        q = rng.integers(0, 16, (n, k//32, 32), dtype=np.uint8)
        packed = q[..., :16] | (q[..., 16:] << 4)
        args = {"X": x, "Weight": packed, "Y": out(m, n, dtype=np.int32)}

        def reference(v):
            w = np.concatenate((v["Weight"] & 15, v["Weight"] >> 4), axis=-1).reshape(n, k)
            return {"Y": v["X"].astype(np.int64) @ w.astype(np.int64).T}
        return Scenario(args, ("Y",), reference, ("signed-i8-times-packed-u4", "two-storage-blocks"),
                        encodings={"X": ("SignedI8", [m, k]), "Weight": ("PackedU4", [n, k])})

    contractions = ("dense_projection", "dense_projection_bias", "renamed_projection", "row_major_matmul", "row_major_matmul_f16", "batched_inner_product", "pairwise_inner_product", "snapshot_matmul", "dynamic_matmul", "lhs_lane_matmul", "output_lane_f64", "matmul_cast", "matmul_observers", "batched_matmul", "batched_lhs_lane", "matmul_epilogue", "paired_projections")
    if name in contractions:
        n = bindings.get("N", 5)
        if name == "batched_inner_product":
            args = {"X": f(3, 2, k), "Weight": f(3, 2, k), "Y": out(3)}
            return Scenario(args, ("Y",), lambda v: {"Y": (v["X"].astype(np.float64)*v["Weight"]).sum(axis=(1, 2))})
        if name == "pairwise_inner_product":
            args = {"X": f(m, 2, k), "Weight": f(n, 2, k), "Y": out(m, n)}
            return Scenario(args, ("Y",), lambda v: {"Y": np.einsum("mpk,npk->mn", v["X"].astype(np.float64), v["Weight"].astype(np.float64))})
        dtype = np.float64 if name == "output_lane_f64" else np.float16 if name == "row_major_matmul_f16" else np.float32
        output_dtype = np.float64 if name in ("output_lane_f64", "matmul_cast") else np.float32
        prefix = (2,) if name in ("batched_matmul", "batched_lhs_lane") else ()
        lhs_transposed = name in ("lhs_lane_matmul", "batched_lhs_lane")
        weight_transposed = lhs_transposed or name in ("dense_projection", "dense_projection_bias", "renamed_projection")
        x = f(*prefix, *((k, m) if lhs_transposed else (m, k)), dtype=dtype)
        w = f(*prefix, *((n, k) if weight_transposed else (k, n)), dtype=dtype)
        args = {"X": x, "Weight": w, "Y": out(*prefix, m, n, dtype=output_dtype)}
        if name in ("dense_projection_bias", "matmul_epilogue"):
            args.update(Bias=f(n), scale=0.75)
        if name == "matmul_epilogue":
            args["Gate"] = f(m)
        if name == "matmul_observers":
            args["Norm"] = out(m)
        if name == "paired_projections":
            args = {"X": x, "Weight0": w, "Weight1": f(k, n), "Bias0": f(n), "Bias1": f(n), "Y0": out(m, n), "Y1": out(m, n)}

        def reference(v):
            if name == "paired_projections":
                return {"Y"+str(i): v["X"].astype(np.float64) @ v["Weight"+str(i)].astype(np.float64) + v["Bias"+str(i)] for i in (0, 1)}
            left, right = v["X"].astype(np.float64), v["Weight"].astype(np.float64)
            y = (left.swapaxes(-1, -2) if lhs_transposed else left) @ (right.swapaxes(-1, -2) if weight_transposed else right)
            if name == "dense_projection_bias":
                y = y*v["scale"]+v["Bias"]
            elif name == "matmul_epilogue":
                y = (y+v["Bias"]) * (y*v["Gate"][:, None]) * v["scale"]
            result = {"Y": y}
            if name == "snapshot_matmul":
                result["Weight"] = np.zeros_like(v["Weight"])
            if name == "matmul_observers":
                result["Norm"] = (y*y).sum(axis=-1)
            return result
        outputs = ("Y0", "Y1") if name == "paired_projections" else ("Y", "Weight") if name == "snapshot_matmul" else ("Y", "Norm") if name == "matmul_observers" else ("Y",)
        return Scenario(args, outputs, reference, ("all-output-elements", "typed-contraction"))
    raise NotImplementedError("No numerical scenario for dense source " + name)

"""Independent numerical scenarios for public, non-catalog examples."""
import numpy as np


def prepare(name, bindings, seed, Scenario):
    rng = np.random.default_rng(seed)
    if name == "helper_level":
        n = bindings.get("N", 128)
        args = {"X": rng.uniform(-.5, .5, n).astype(np.float32), "Y": np.full(n, np.nan, np.float32)}
        return Scenario(args, ("Y",), lambda v: {"Y": (v["X"].astype(np.float64)+3)*0.25},
                        ("helper-context-return", "caller-effect-after-return", "complete-output"))
    if name == "gemv_f32":
        args = {"W": rng.uniform(-.5, .5, (5, 35)).astype(np.float32),
                "X": rng.uniform(-.5, .5, 35).astype(np.float32), "Y": np.full(5, np.nan, np.float32)}
        return Scenario(args, ("Y",), lambda v: {"Y": v["W"].astype(np.float64) @ v["X"].astype(np.float64)})
    if name in ("ime_i8_contract", "i8_bias_clamp"):
        args = {"A": rng.integers(-8, 8, (4, 24), dtype=np.int8),
                "B": rng.integers(-8, 8, (24, 32), dtype=np.int8),
                "C": np.full((4, 32), -12345, np.int32)}
        if name == "i8_bias_clamp":
            args["Bias"] = rng.integers(-100, 100, 32, dtype=np.int32)

        def reference(v):
            result = v["A"].astype(np.int64) @ v["B"].astype(np.int64)
            return {"C": np.clip(result+v["Bias"], 0, 127) if "Bias" in v else result}
        block = bindings.get("KB", 8)
        boundary = "partial-K-block" if 24 % block else str(24//block) + "-K-blocks"
        return Scenario(args, ("C",), reference, ("typed-i8-contraction", boundary))
    if name == "online_flash_attention":
        args = {key: rng.uniform(-.5, .5, shape).astype(np.float16) for key, shape in
                (("Q", (5, 16)), ("K", (17, 16)), ("V", (17, 16)))}
        args["O"] = np.full((5, 16), np.nan, np.float16)

        def reference(v):
            scores = v["Q"].astype(np.float64) @ v["K"].astype(np.float64).T
            probabilities = np.exp(scores-scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            return {"O": (probabilities @ v["V"].astype(np.float64)).astype(np.float16)}
        return Scenario(args, ("O",), reference, ("unscaled-noncausal-attention", "partial-Q-and-K-tiles"))
    if name == "top_k_f32":
        x = np.array([1, 7, 3, 7, -1, 2, 0, 7, 4, 2, 6, 1, 0, -3, 5, 2, 4], np.float32)
        return Scenario({"X": x, "out": np.full(4, -99, np.int32)}, ("out",),
                        lambda v: {"out": np.argsort(-v["X"], kind="stable")[:4].astype(np.int32)}, ("stable-topk-ties",))
    if name == "affine_strided":
        x = rng.uniform(-.5, .5, (3, 70)).astype(np.float32)[:, ::2]
        args = {"X": x, "Y": np.full((3, 35), np.nan, np.float32), "scale": 1.25, "bias": -.5}
        return Scenario(args, ("Y",), lambda v: {"Y": v["X"].astype(np.float64)*v["scale"]+v["bias"]}, ("noncontiguous-input",))
    if name == "affine_backward":
        grad = np.broadcast_to(rng.uniform(-.5, .5, (1, 35)).astype(np.float32), (3, 35))
        return Scenario({"Grad": grad, "DX": np.full((3, 35), np.nan, np.float32), "scale": 1.25}, ("DX",),
                        lambda v: {"DX": v["Grad"].astype(np.float64)*v["scale"]}, ("readonly-zero-stride-gradient",))
    raise NotImplementedError("No example scenario for " + name)

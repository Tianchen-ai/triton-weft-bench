"""Explicit GGML and IME byte-layout fixtures; no compiler or timing logic."""
import struct

import numpy as np


def q4_k_rows(rows, blocks, rng):
    q = rng.integers(0, 16, (rows, blocks, 8, 32), dtype=np.uint8)
    scales = rng.integers(1, 8, (rows, blocks, 8), dtype=np.uint8)
    minima = rng.integers(0, 16, (rows, blocks, 8), dtype=np.uint8)
    raw = np.zeros((rows, blocks, 144), dtype=np.uint8)
    decoded = np.empty((rows, blocks, 8, 32), dtype=np.float64)
    for row in range(rows):
        for block in range(blocks):
            data = raw[row, block]
            data[:4] = np.frombuffer(struct.pack("<ee", 0.125, 0.0625), np.uint8)
            s, m = scales[row, block], minima[row, block]
            data[4:8] = (s[:4] & 63) | ((s[4:] >> 4) << 6)
            data[8:12] = (m[:4] & 63) | ((m[4:] >> 4) << 6)
            data[12:16] = (s[4:] & 15) | ((m[4:] & 15) << 4)
            for pair in range(4):
                data[16+pair*32:48+pair*32] = q[row, block, pair*2] | (q[row, block, pair*2+1] << 4)
            decoded[row, block] = 0.125*s[:, None]*q[row, block] - 0.0625*m[:, None]
    return raw, decoded.reshape(rows, blocks*256), q, scales, minima


def q8_k_rows(rows, blocks, rng):
    q = rng.integers(-127, 64, (rows, blocks, 256), dtype=np.int16).astype(np.int8)
    q[:, :, 0] = -128  # Unique signed absmax; exactly representable scale.
    raw = pack_q8_k(np.full((rows, blocks), 1/64, np.float32), q)
    return raw, (q.astype(np.float32)/64).reshape(rows, blocks*256)


def pack_q8_k(scales, q):
    rows, blocks, _ = q.shape
    raw = np.zeros((rows, blocks, 292), dtype=np.uint8)
    for row in range(rows):
        for block in range(blocks):
            raw[row, block, :4] = np.frombuffer(struct.pack("<f", scales[row, block]), np.uint8)
            raw[row, block, 4:260] = q[row, block].view(np.uint8)
            sums = q[row, block].astype(np.int16).reshape(16, 16).sum(axis=1).astype("<i2")
            raw[row, block, 260:] = sums.view(np.uint8)
    return raw


def quantize_q8_k(x):
    """Same signed-extremum/RNE contract as the existing native scalar oracle."""
    values = x.reshape(x.shape[0], -1, 256)
    hi, lo = values.max(axis=-1), values.min(axis=-1)
    extreme = np.where(np.abs(hi) > np.abs(lo), hi, lo)
    inverse = np.divide(np.float32(-127), extreme, out=np.zeros_like(extreme), where=extreme != 0)
    scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=inverse != 0)
    q = np.clip(np.rint(values*inverse[..., None]), -128, 127).astype(np.int8)
    return pack_q8_k(scale, q), (q.astype(np.float64)*scale[..., None]).reshape(x.shape)


def ime_weights(scales, zero_points, q, affine):
    rows, blocks, _ = q.shape
    stride, payload = (304, 48) if affine else (288, 32)
    result = np.zeros((rows//16, blocks, stride), dtype=np.uint8)
    for row in range(rows):
        for block in range(blocks):
            record = result[row//16, block]
            record[(row%16)*2:(row%16)*2+2] = np.frombuffer(struct.pack("<e", scales[row, block]), np.uint8)
            if affine:
                record[32+row%16] = zero_points[row, block]
            for k in range(32):
                index = payload+(k//16)*128+(row%16)*8+k%8
                record[index] |= int(q[row, block, k]) << (((k//8)%2)*4)
    return result


def ime_activation(x, row_tile):
    rows, k = x.shape
    blocks = k//32
    values = x.reshape(rows, blocks, 32)
    d = np.max(np.abs(values), axis=-1).astype(np.float32)*np.float32(1/127)
    inverse = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
    q = np.clip(np.rint(values*inverse[..., None]), -128, 127).astype(np.int8)
    result = np.zeros((rows//row_tile, blocks, row_tile*36), dtype=np.uint8)
    for row in range(rows):
        for block in range(blocks):
            record = result[row//row_tile, block]
            slot = row%row_tile
            record[slot*4:slot*4+4] = np.frombuffer(struct.pack("<f", d[row, block]), np.uint8)
            for k in range(32):
                record[4*row_tile+(k//8)*row_tile*8+slot*8+k%8] = q[row, block, k].view(np.uint8)
    decoded = (q.astype(np.float64)*d[..., None]).reshape(rows, blocks*32)
    return result, decoded


def prepare(name, bindings, seed, Scenario):
    rng = np.random.default_rng(seed)
    if name in ("q4_k_prepared_gemv", "q4_k_persistent"):
        n, blocks = 32, 2
        m = 1 if name == "q4_k_prepared_gemv" else 3
        plain, weights, _, _, _ = q4_k_rows(n, blocks, rng)
        # Q4K_I[16] uses the declared default one-byte field chunks.
        packed = plain.reshape(n//16, 16, blocks, 144).transpose(0, 2, 3, 1).copy()
        activation, x = q8_k_rows(m, blocks, rng)
        output_shape = (n,) if m == 1 else (m, n)
        args = {"W": packed, "Y": np.full(output_shape, np.nan, np.float32)}
        encodings = {"W": ("ggml.Q4K_I[16]", [n, blocks*256])}
        if m == 1:
            args["X"] = activation.reshape(-1)
            encodings["X"] = ("ggml.Q8_K", [blocks*256])
        else:
            args.update(X=x, Xq=np.full_like(activation, 0xa5))
            encodings["Xq"] = ("ggml.Q8_K", [m, blocks*256])
        expected_activation, expected_x = (activation, x) if m == 1 else quantize_q8_k(x)

        def reference(v):
            result = {"Y": (expected_x.astype(np.float64) @ weights.T).reshape(output_shape)}
            if m > 1:
                result["Xq"] = expected_activation
            return result
        return Scenario(args, ("Y",) if m == 1 else ("Y", "Xq"), reference,
                        ("multiple-quantized-blocks", "declared-byte-interleave"), encodings=encodings)

    if name in ("pack_q4_0", "pack_q4_1", "pack_q4_k"):
        n = 32
        if name == "pack_q4_k":
            w, _, q, s, minima = q4_k_rows(n, 2, rng)
            q = q.reshape(n, 16, 32)
            d = (0.125*s).astype(np.float16).astype(np.float32).reshape(n, 16)
            offset = (-0.0625*minima).astype(np.float16).astype(np.float32).reshape(n, 16)
            zeros = np.clip(np.rint(-offset/d), 0, 15).astype(np.uint8)
            affine, k = True, 512
        else:
            blocks = 3
            affine = name == "pack_q4_1"
            k = blocks*32
            q = rng.integers(0, 16, (n, blocks, 32), dtype=np.uint8)
            d = np.full((n, blocks), 0.125, np.float32)
            # Include a clipped zero point; conversion semantics are explicit.
            offsets = -0.125*np.resize(np.array([0, 1, 7, 16], np.float32), (n, blocks))
            zeros = np.clip(np.rint(-offsets/d), 0, 15).astype(np.uint8) if affine else np.full((n, blocks), 8, np.uint8)
            w = np.zeros((n, blocks, 20 if affine else 18), np.uint8)
            for row in range(n):
                for block in range(blocks):
                    prefix = struct.pack("<ee", d[row, block], offsets[row, block]) if affine else struct.pack("<e", d[row, block])
                    w[row, block, :len(prefix)] = np.frombuffer(prefix, np.uint8)
                    w[row, block, len(prefix):] = q[row, block, :16] | (q[row, block, 16:] << 4)
        wanted = ime_weights(d, zeros, q, affine)
        args = {"W": w, "P": np.full_like(wanted, 0xa5)}
        layout = "IMEQ4Affine_I[16]" if affine else "IMEQ4Centered_I[16]"
        fmt = {"pack_q4_0": "Q4_0", "pack_q4_1": "Q4_1", "pack_q4_k": "Q4_K"}[name]
        return Scenario(args, ("P",), lambda v: {"P": wanted}, ("exact-packed-fields", "multiple-row-groups"),
                        encodings={"W": ("ggml."+fmt, [n, k]), "P": (layout, [n, k])})

    if name == "quantize_ime1_prefill" or name.startswith("q4_"):
        m, n, k = (1 if "decode" in name else 4), 32, 64
        x = ((np.arange(m*k, dtype=np.int32).reshape(m, k)%17-8)/9).astype(np.float32)
        x[0, :32] = 0  # Zero activation is part of the ordinary quantization domain.
        packed_x, decoded_x = ime_activation(x, m)
        if name == "quantize_ime1_prefill":
            return Scenario({"X": x, "Xq": np.full_like(packed_x, 0xa5)}, ("Xq",), lambda v: {"Xq": packed_x},
                            ("zero-block", "nonzero-block", "exact-activation-records"),
                            encodings={"Xq": ("IMEQ8_I[4]", [m, k])})
        affine = "affine" in name
        q = rng.integers(0, 16, (n, k//32, 32), dtype=np.uint8)
        scales = np.full((n, k//32), 0.125, np.float32)
        zeros = rng.integers(0, 16, (n, k//32), dtype=np.uint8) if affine else np.full((n, k//32), 8, np.uint8)
        packed_w = ime_weights(scales, zeros, q, affine)
        decoded_w = ((q.astype(np.float64)-zeros[..., None])*scales[..., None]).reshape(n, k)
        args = {"W": packed_w, "Y": np.full((m, n), np.nan, np.float32), "Xq": packed_x.copy()}
        encodings = {"W": ("IMEQ4Affine_I[16]" if affine else "IMEQ4Centered_I[16]", [n, k]),
                     "Xq": (f"IMEQ8_I[{m}]", [m, k])}
        outputs = ("Y",)
        if "packed" not in name:
            args["X"] = x
            args["Xq"].fill(0xa5)
            outputs = ("Y", "Xq")

        def reference(v):
            result = {"Y": decoded_x @ decoded_w.T}
            if "X" in v:
                result["Xq"] = packed_x
            return result
        return Scenario(args, outputs, reference, ("packed-layout", "two-K-blocks", "two-output-tiles"), encodings=encodings)
    raise NotImplementedError("No encoded scenario for " + name)

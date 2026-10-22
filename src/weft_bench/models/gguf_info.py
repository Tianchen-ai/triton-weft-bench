"""Read GGUF metadata/tensor shapes without reading any tensor data."""

import collections
import json
import struct
import sys


def inspect(path):
    with open(path, "rb") as f:

        def number(fmt):
            size = struct.calcsize(fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError("truncated GGUF metadata")
            return struct.unpack("<" + fmt, data)[0]

        def string(keep=True):
            size = number("Q")
            if keep:
                return f.read(size).decode("utf-8")
            f.seek(size, 1)

        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }

        def value(kind, keep):
            if kind == 8:
                return string(keep)
            if kind == 9:
                sub = number("I")
                length = number("Q")
                if sub in formats and not keep:
                    f.seek(struct.calcsize(formats[sub]) * length, 1)
                    return None
                values = []
                for _ in range(length):
                    x = value(sub, keep)
                    if keep:
                        values.append(x)
                return values if keep else None
            if kind not in formats:
                raise ValueError(f"unsupported GGUF value type {kind}")
            return number(formats[kind])

        if f.read(4) != b"GGUF":
            raise ValueError("not GGUF")
        version = number("I")
        if version not in (2, 3):
            raise ValueError("supported GGUF versions are 2 and 3")
        tensors = number("Q")
        pairs = number("Q")
        metadata = {}
        for _ in range(pairs):
            key = string()
            kind = number("I")
            keep = key.startswith("general.") or any(
                s in key
                for s in (
                    "embedding_length",
                    "head_count",
                    "feed_forward_length",
                    "block_count",
                    "context_length",
                )
            )
            v = value(kind, keep)
            if keep:
                metadata[key] = v
        shapes = collections.Counter()
        examples = {}
        for _ in range(tensors):
            name = string()
            rank = number("I")
            dims = tuple(number("Q") for _ in range(rank))
            kind = number("I")
            number("Q")
            if rank == 2 and name.endswith(".weight"):
                shapes[(kind, dims)] += 1
                examples.setdefault((kind, dims), name)
        return {
            "model": str(path),
            "metadata": metadata,
            "tensor_shapes": [
                {
                    "ggml_type": kind,
                    "shape": list(dims),
                    "count": count,
                    "example": examples[(kind, dims)],
                }
                for (kind, dims), count in shapes.most_common()
            ],
        }


if __name__ == "__main__":
    print(json.dumps(inspect(sys.argv[1])))

"""llama.cpp's supported model workloads, independent of replacement kernels."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInputs:
    prefill: int = 128
    decode: int = 32
    depth: int = 512
    threads: int | None = None
    repetitions: int = 3

    @classmethod
    def preset(cls, name):
        if name == "default":
            return cls()
        if name == "smoke":
            return cls(prefill=16, decode=4, depth=32, repetitions=1)
        raise ValueError(f"unknown model input preset {name!r}")

    def phases(self):
        if min(self.prefill, self.decode, self.repetitions) < 1 or self.depth < 0:
            raise ValueError("invalid model phase lengths")
        if self.threads is not None and self.threads < 1:
            raise ValueError("model threads must be positive")
        return {
            "prefill": {"prompt": self.prefill, "generate": 0, "depth": 0},
            "decode": {"prompt": 0, "generate": self.decode, "depth": self.depth},
        }


@dataclass(frozen=True)
class LlamaCpp:
    binary: str
    package: str = ""
    replacements: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    source: str = ""
    build_directory: str = ""
    weft_include: str = ""

    def replace(self, mapping):
        from dataclasses import replace

        return replace(self, replacements=dict(mapping), package="")

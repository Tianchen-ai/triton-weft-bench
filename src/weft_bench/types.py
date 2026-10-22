from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    shape: dict[str, int]
    scope: str
    input_policy: str = "seed410"
    dtype: str = "f32"
    phase: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        # Older saved runs contained driver/build fields which now belong outside the workload.
        from dataclasses import fields

        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in names})


@dataclass(frozen=True)
class Implementation:
    """Compiler/source binding, independent of workload input and oracle."""

    source: str = ""
    symbol: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Measurement:
    warmup: int = 3
    samples: int = 5
    batch_ms: float = 10.0
    cache: str = "reuse"
    eviction_bytes: int = 64 * 1024 * 1024
    check_only: bool = False

    def __post_init__(self):
        if self.warmup < 0 or self.samples < 1 or self.batch_ms <= 0:
            raise ValueError("warmup >= 0, samples >= 1 and batch_ms > 0 required")
        if self.cache not in ("reuse", "cold") or self.eviction_bytes < 1:
            raise ValueError(
                "cache must be reuse/cold; eviction_bytes must be positive"
            )

    def env(self):
        return {
            "WB_WARMUP": str(self.warmup),
            "WB_SAMPLES": str(self.samples),
            "WB_BATCH_MS": str(self.batch_ms),
            "WB_CACHE": self.cache,
            "WB_EVICTION_BYTES": str(self.eviction_bytes),
            "WB_CHECK_ONLY": "1" if self.check_only else "0",
        }


@dataclass(frozen=True)
class Target:
    name: str
    settings: dict[str, Any]
    implementations: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class PreparedRun:
    argv: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    # Extra fields describing a different native driver's explicit timing scope.
    contract: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    case: str
    implementation: str
    target: str
    shape: dict[str, int]
    scope: str
    status: str
    median_ms: float | None = None
    samples_ms: list[float] = field(default_factory=list)
    correctness: str = "not_checked"
    speedup: float | None = None
    baseline: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

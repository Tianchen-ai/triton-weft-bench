"""Lightweight native timing; compiler dependencies are loaded only by adapters."""

from .api import benchmark, compare
from .inputs import Inputs
from .libraries import GGML, WeftKernel
from .models.scenarios import LlamaCpp, ModelInputs


def compare_model(*args, **kwargs):
    from .models.api import compare_model as run

    return run(*args, **kwargs)


def capture_model_reference(*args, **kwargs):
    from .models.validation import capture_model_reference as run

    return run(*args, **kwargs)


from .types import Case, Implementation, Measurement, PreparedRun, Result, Target

__all__ = [
    "benchmark",
    "compare",
    "Inputs",
    "GGML",
    "WeftKernel",
    "compare_model",
    "capture_model_reference",
    "LlamaCpp",
    "ModelInputs",
    "Case",
    "Implementation",
    "Measurement",
    "PreparedRun",
    "Result",
    "Target",
]

"""Model backend controls that change execution independently of build flags."""

BACKEND_ENVIRONMENT = (
    "GGML_SPACEMIT_BACKEND",
    "GGML_SPACEMIT_FUSION",
    "GGML_SPACEMIT_WORKERS",
    "SPACEMIT_PERFER_CORE_ID",
)


def backend_policy(environment):
    return {key: str(environment[key]) for key in BACKEND_ENVIRONMENT if key in environment}

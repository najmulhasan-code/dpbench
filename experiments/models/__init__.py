"""Model wrappers for experiment."""

from experiments.models.openai_model import create_openai_model
from experiments.models.anthropic_model import create_anthropic_model
from experiments.models.google_model import create_google_model
from experiments.models.xai_model import create_xai_model

MODEL_REGISTRY = {
    "openai": create_openai_model,
    "anthropic": create_anthropic_model,
    "google": create_google_model,
    "xai": create_xai_model,
}


def create_model(provider: str, model_id: str, **kwargs):
    """Create model function by provider name."""
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}")
    return MODEL_REGISTRY[provider](model_id, **kwargs)


__all__ = [
    "create_model",
    "create_openai_model",
    "create_anthropic_model",
    "create_google_model",
    "create_xai_model",
    "MODEL_REGISTRY",
]

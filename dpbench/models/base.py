"""Model interface definition."""

from typing import Callable, Protocol
import inspect


class ModelFunction(Protocol):
    """Protocol for model functions: (system_prompt, user_prompt) -> str."""

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        ...


def validate_model_function(model_fn: Callable) -> None:
    """Validate that model_fn has the correct signature."""
    if not callable(model_fn):
        raise TypeError(f"model_fn must be callable, got {type(model_fn).__name__}")
    sig = inspect.signature(model_fn)
    if len(sig.parameters) < 2:
        raise TypeError(
            f"model_fn must accept at least 2 arguments, got {len(sig.parameters)}"
        )

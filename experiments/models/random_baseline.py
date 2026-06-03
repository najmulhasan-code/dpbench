"""Random baseline agent. Returns random actions without calling any API."""

import random


def create_random_model(seed: int | None = None):
    """Create a model function that returns random actions.

    Used as a baseline to compare against LLM performance.
    No API calls, no cost, instant execution.
    """
    rng = random.Random(seed)
    actions = ["GRAB_LEFT", "GRAB_RIGHT", "RELEASE", "WAIT"]

    def model_fn(system_prompt: str, user_prompt: str) -> str:
        action = rng.choice(actions)
        return f"THINKING: Random action selected.\nACTION: {action}"

    return model_fn

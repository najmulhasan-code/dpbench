"""xAI model wrapper for Grok 4.1."""

import os
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

from dpbench.core.types import ModelResponse

load_dotenv()


def create_xai_model(
    model_id: str = "grok-4-1-fast-reasoning",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Callable[[str, str], ModelResponse]:
    """Returns (system_prompt, user_prompt) -> ModelResponse callable.

    Uses OpenAI-compatible API. Returns ModelResponse with token counts for cost tracking.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    def model_fn(system_prompt: str, user_prompt: str) -> ModelResponse:
        response = client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens if response.usage else None
        tokens_out = response.usage.completion_tokens if response.usage else None
        return ModelResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out)

    return model_fn

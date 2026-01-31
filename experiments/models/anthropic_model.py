"""Anthropic model wrapper for Claude Opus 4.5."""

import os
from typing import Callable

from dotenv import load_dotenv
import anthropic

from dpbench.core.types import ModelResponse

load_dotenv()


def create_anthropic_model(
    model_id: str = "claude-opus-4-5-20251101",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Callable[[str, str], ModelResponse]:
    """Returns (system_prompt, user_prompt) -> ModelResponse callable.

    Uses streaming to handle Claude Opus 4.5's longer processing times.
    Returns ModelResponse with token counts for cost tracking.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    client = anthropic.Anthropic(api_key=api_key)

    def model_fn(system_prompt: str, user_prompt: str) -> ModelResponse:
        # Use streaming for Opus models (required by SDK for long operations)
        with client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        ) as stream:
            # get_final_message() returns the full message with usage stats
            message = stream.get_final_message()
            text = message.content[0].text
            tokens_in = message.usage.input_tokens
            tokens_out = message.usage.output_tokens
            return ModelResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out)

    return model_fn

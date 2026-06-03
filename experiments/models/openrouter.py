"""OpenRouter model wrapper for DPBench experiments.

All models (closed-source and open-source) are accessed through
OpenRouter's API using the OpenAI SDK with OpenRouter's base URL,
as documented at openrouter.ai/docs/quickstart.
"""

import os
from openai import OpenAI
from dpbench.core.types import ModelResponse


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Set it in your .env file."
        )
    return key


def create_openrouter_model(
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """Build a callable that sends prompts to a model via OpenRouter.

    The returned function matches DPBench's model_fn protocol:
        (system_prompt: str, user_prompt: str) -> ModelResponse
    """
    api_key = _get_api_key()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    def call(system_prompt: str, user_prompt: str) -> ModelResponse:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = ""
        if completion.choices:
            text = completion.choices[0].message.content or ""

        tokens_in = None
        tokens_out = None
        if completion.usage:
            tokens_in = completion.usage.prompt_tokens
            tokens_out = completion.usage.completion_tokens

        return ModelResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    return call

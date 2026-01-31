"""Google model wrapper for Gemini 2.5 Pro."""

import os
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from dpbench.core.types import ModelResponse

load_dotenv()


def create_google_model(
    model_id: str = "gemini-2.5-pro",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Callable[[str, str], ModelResponse]:
    """Returns (system_prompt, user_prompt) -> ModelResponse callable.

    Returns ModelResponse with token counts for cost tracking.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    client = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(timeout=120000),  # 120 second timeout (in ms)
    )

    def model_fn(system_prompt: str, user_prompt: str) -> ModelResponse:
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # Extract text from response
        text = None
        try:
            if response.text:
                text = response.text
        except Exception:
            pass

        if text is None:
            try:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text = part.text
                                break
            except Exception:
                pass

        if text is None:
            raise ValueError(f"Could not extract text from Gemini response: {response}")

        # Extract token counts from usage_metadata
        tokens_in = None
        tokens_out = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            tokens_in = getattr(usage, 'prompt_token_count', None)
            tokens_out = getattr(usage, 'candidates_token_count', None)

        return ModelResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out)

    return model_fn

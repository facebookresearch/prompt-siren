# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Provider for Google Gemini via native API with thinking support."""
import os

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model

try:
    from pydantic_ai.models.google import GoogleModel
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        "Please install the `google-genai` package to use the Gemini provider, "
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


class GeminiProvider:
    """Provider for Google Gemini models via native API.

    This provider handles all models with the "gemini:" prefix and creates
    GoogleModel instances configured with the native Gemini API.

    Configuration is read from the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.

    The native API supports:
    - Thinking/reasoning traces via google_thinking_config
    - Thought signatures for multi-turn conversations
    - Full feature parity with Google AI Studio

    Usage:
        agent.config.model=gemini:gemini-2.5-flash
        agent.config.model=gemini:gemini-2.5-pro
        agent.config.model=gemini:gemini-2.0-flash-thinking-exp
    """

    def create_model(self, model_string: str) -> Model:
        """Create a GoogleModel configured for Gemini.

        Args:
            model_string: Full model string (e.g., "gemini:gemini-2.5-flash")

        Returns:
            GoogleModel instance configured to work with the native Gemini API.
        """
        # Ensure API key is set before creating the model
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise UserError(
                "Set the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable "
                "to use the native Gemini provider."
            )

        # Set GOOGLE_API_KEY if not already set (pydantic-ai uses this)
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = api_key

        model_name = model_string.split(":", 1)[1]
        return GoogleModel(model_name, provider="google-gla")


def create_gemini_provider() -> GeminiProvider:
    """Factory function for registry compatibility."""
    return GeminiProvider()

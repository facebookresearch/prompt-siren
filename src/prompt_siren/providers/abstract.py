# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Provider protocol for the Siren.

This module defines the protocol for model providers that can handle custom
model creation logic for specific provider prefixes (e.g., "bedrock:", "custom:").

Providers are registered with a name (e.g., "bedrock") and handle all models
with that prefix (e.g., "bedrock:us.anthropic.claude-sonnet-4").

Providers read their configuration from environment variables, following
the same pattern as pydantic-ai's built-in providers.
"""

from typing import Protocol

from pydantic_ai.models import Model


class AbstractProvider(Protocol):
    """Protocol for provider implementations.

    Providers handle the creation of Model instances for specific
    provider prefixes. For example, a BedrockProvider registered as "bedrock"
    handles all models starting with "bedrock:" and creates appropriate Model
    instances with the correct client configuration.

    Providers should read their configuration (API keys, regions, etc.) from
    environment variables, following the same pattern as pydantic-ai's built-in
    providers (e.g., OpenAI reads OPENAI_API_KEY, Bedrock reads AWS_REGION).
    """

    def create_model(self, model_string: str) -> Model:
        """Create a model instance for the given model string.

        Args:
            model_string: Full model string (e.g., "bedrock:us.anthropic.claude-sonnet-4")

        Returns:
            Model instance ready to use with pydantic-ai

        Raises:
            ValueError: If the model string is not valid for this provider
        """
        ...

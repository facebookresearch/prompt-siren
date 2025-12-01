# Copyright (c) Meta Platforms, Inc. and affiliates.
"""AWS Bedrock provider implementation."""

from anthropic import AsyncAnthropicBedrock
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


class BedrockProvider:
    """Provider for AWS Bedrock models.

    This provider handles all models with the "bedrock:" prefix and creates
    AnthropicModel instances configured with the AWS Bedrock client.

    Configuration is automatically read from AWS environment variables or
    configuration files, following AWS SDK conventions.
    """

    def __init__(self):
        """Initialize the Bedrock provider.

        Creates the AsyncAnthropicBedrock client, which automatically reads
        configuration from AWS environment variables and configuration files.
        """
        # AsyncAnthropicBedrock automatically reads from AWS env vars and config
        self._client = AsyncAnthropicBedrock()

    def create_model(self, model_string: str) -> Model:
        """Create an AnthropicModel configured for Bedrock.

        Args:
            model_string: Full model string (e.g., "bedrock:us.anthropic.claude-sonnet-4")

        Returns:
            AnthropicModel instance configured with Bedrock client
        """
        # Create an AnthropicProvider with our Bedrock client
        anthropic_provider = AnthropicProvider(anthropic_client=self._client)

        # Create and return the AnthropicModel
        return AnthropicModel(model_name=model_string, provider=anthropic_provider)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry for provider types and their factory functions.

This module provides a centralized registry for providers, allowing for
dynamic model creation with type safety. Providers handle all models with
a specific prefix (e.g., "bedrock:*") rather than individual models.

Providers are looked up by parsing the prefix from the model string
(e.g., "bedrock:model-name" -> look up "bedrock" provider).
"""

from collections.abc import Callable
from typing import TypeAlias

from pydantic_ai import models
from pydantic_ai.models import Model

from ..registry_base import BaseRegistry
from .abstract import AbstractProvider

# Type alias for provider factory functions (no config needed)
ProviderFactory: TypeAlias = Callable[[], AbstractProvider]

# Create a global provider registry instance using BaseRegistry
provider_registry = BaseRegistry[AbstractProvider, None]("provider", "prompt_siren.providers")


def register_provider(provider_name: str, factory: ProviderFactory) -> None:
    """Register a provider with its factory function.

    Args:
        provider_name: Name of the provider (e.g., "bedrock", "custom").
                       This name is used as the prefix in model strings (e.g., "bedrock:model-name")
        factory: Function that takes no arguments and returns an AbstractProvider instance

    Raises:
        ValueError: If the provider name is already registered

    Example:
        >>> def create_my_provider() -> MyProvider:
        ...     return MyProvider()
        >>> register_provider("my_provider", create_my_provider)
    """
    provider_registry.register(provider_name, None, factory)


def get_registered_providers() -> list[str]:
    """Get a list of all registered providers.

    Returns:
        List of provider names
    """
    return provider_registry.get_registered_components()


def infer_model(model_string: str) -> Model:
    """Infer and create a model, checking registered providers first.

    This function first checks if the model string has a prefix that matches
    any registered custom provider. If so, it uses that provider to create
    the model. Otherwise, it falls back to pydantic_ai's default model inference.

    Args:
        model_string: Model identifier (e.g., "bedrock:claude-sonnet-4", "openai:gpt-5")

    Returns:
        Model instance ready to use with pydantic-ai

    Examples:
        >>> # Use pydantic_ai's default inference for built-in providers
        >>> model = infer_model("openai:gpt-5")

        >>> # Use custom provider (if "bedrock" is registered)
        >>> model = infer_model("bedrock:us.anthropic.claude-sonnet-4")

    Note:
        Providers read their configuration from environment variables,
        following the same pattern as pydantic-ai's built-in providers.
    """
    # Check if model string has a prefix that matches a registered provider
    if ":" in model_string:
        prefix = model_string.split(":", 1)[0]
        if prefix in provider_registry.get_registered_components():
            # Create provider instance and use it to create the model
            provider = provider_registry.create_component(prefix, None, None)
            return provider.create_model(model_string)

    # Fall back to pydantic_ai's default inference
    return models.infer_model(model_string)

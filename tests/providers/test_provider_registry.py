# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the provider registry."""

import pytest
from prompt_siren.providers import (
    infer_model,
    provider_registry,
    register_provider,
)
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


class MockProvider:
    """Mock provider for testing."""

    def create_model(self, model_string: str) -> Model:
        """Create a test model."""
        return TestModel()


def create_mock_provider() -> MockProvider:
    """Factory for mock provider."""
    return MockProvider()


def test_register_provider():
    """Test registering a provider."""
    # Register a mock provider
    register_provider("test_mock", create_mock_provider)

    # Verify it's registered
    assert "test_mock" in provider_registry.get_registered_components()


def test_infer_model_with_custom_provider():
    """Test that infer_model uses custom providers."""
    # Register mock provider
    register_provider("test_custom", create_mock_provider)

    # Use the custom provider
    model = infer_model("test_custom:some-model")

    # Should return a TestModel from our mock provider
    assert isinstance(model, TestModel)


def test_infer_model_falls_back_to_pydantic_ai():
    """Test that infer_model falls back to pydantic_ai for unknown prefixes."""
    # Use a built-in pydantic_ai provider (test model)
    model = infer_model("test")

    # Should return a model from pydantic_ai
    assert isinstance(model, Model)


def test_infer_model_without_prefix():
    """Test that models without prefixes are handled by pydantic_ai."""
    # Models without ':' should go to pydantic_ai
    model = infer_model("test")

    assert isinstance(model, Model)


def test_provider_registry_no_config():
    """Test that providers don't use config classes."""
    register_provider("test_no_config", create_mock_provider)

    # Get config class - should be None for providers
    config_class = provider_registry.get_config_class("test_no_config")

    assert config_class is None


def test_create_component_without_config():
    """Test that components without config are created correctly."""
    register_provider("test_component", create_mock_provider)

    # Create component without passing config
    provider = provider_registry.create_component("test_component", None, None)

    assert isinstance(provider, MockProvider)


def test_create_component_rejects_config_when_not_needed():
    """Test that passing config to a no-config component raises an error."""

    class DummyConfig(BaseModel):
        value: int = 1

    register_provider("test_reject_config", create_mock_provider)

    # Should raise ValueError when config is provided but not needed
    with pytest.raises(ValueError, match="doesn't accept config"):
        provider_registry.create_component("test_reject_config", DummyConfig(), None)


def test_duplicate_registration_raises_error():
    """Test that registering the same provider twice raises an error."""
    register_provider("test_duplicate", create_mock_provider)

    # Trying to register again should raise
    with pytest.raises(ValueError, match="already registered"):
        register_provider("test_duplicate", create_mock_provider)

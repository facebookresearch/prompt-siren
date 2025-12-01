# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the sandbox manager plugin system."""

import pytest
from prompt_siren.registry_base import UnknownComponentError
from prompt_siren.sandbox_managers import (
    create_sandbox_manager,
    get_registered_sandbox_managers,
    get_sandbox_config_class,
)
from prompt_siren.sandbox_managers.docker import DockerSandboxConfig
from prompt_siren.sandbox_managers.registry import sandbox_registry

from ..conftest import (
    create_mock_sandbox,
    MockSandboxConfig,
    MockSandboxManager,
)


class TestSandboxManagerRegistry:
    """Tests for the sandbox manager plugin system."""

    def setup_method(self):
        """Set up test environment by clearing registry and registering mock sandbox."""
        # Clear the registry for clean tests
        sandbox_registry._registry.clear()
        sandbox_registry._entry_points_loaded = False

        # Manually register mock sandbox for testing
        sandbox_registry.register("mock", MockSandboxConfig, create_mock_sandbox)

    def test_register_and_get_config_class(self):
        """Test registering a sandbox manager and retrieving its config class."""
        # Check that it's listed in registered sandbox managers
        assert "mock" in get_registered_sandbox_managers()

        # Get the config class
        config_class = get_sandbox_config_class("mock")

        # Check that it's the correct class
        assert config_class == MockSandboxConfig

    def test_entry_point_discovery(self):
        """Test that entry points are discovered automatically."""
        # Clear manual registrations
        sandbox_registry._registry.clear()
        sandbox_registry._entry_points_loaded = False

        # Check that built-in sandbox managers are discovered via entry points
        registered_sandboxes = get_registered_sandbox_managers()
        assert "local-docker" in registered_sandboxes

        # Test that we can get the config classes
        docker_config_class = get_sandbox_config_class("local-docker")
        assert docker_config_class is DockerSandboxConfig

    def test_create_sandbox_manager(self):
        """Test creating a sandbox manager from config."""
        # Create a config
        config = MockSandboxConfig(name="Custom Mock Sandbox", custom_parameter="test-value")

        # Create the sandbox manager
        sandbox = create_sandbox_manager("mock", config)

        # Check that it's the correct type
        assert isinstance(sandbox, MockSandboxManager)

        # Check that it has the correct values
        assert sandbox.name == "Custom Mock Sandbox"
        assert sandbox.custom_parameter == "test-value"

        # Check that config property returns the original config
        assert sandbox.config == config
        assert sandbox.config.name == "Custom Mock Sandbox"
        assert sandbox.config.custom_parameter == "test-value"

    def test_missing_sandbox_type(self):
        """Test error when requesting an unregistered sandbox manager type."""
        with pytest.raises(UnknownComponentError):
            get_sandbox_config_class("non-existent-sandbox")

        config = MockSandboxConfig()
        with pytest.raises(UnknownComponentError):
            create_sandbox_manager("non-existent-sandbox", config)

    def test_duplicate_registration(self):
        """Test error when attempting to register the same sandbox manager type twice."""
        # Attempt to register the same type again should raise ValueError
        with pytest.raises(
            ValueError,
            match="Sandbox_Manager type 'mock' is already registered",
        ):
            sandbox_registry.register("mock", MockSandboxConfig, create_mock_sandbox)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Simple tests for the agent plugin system."""

import pytest
from prompt_siren.agents import (
    create_agent,
    get_agent_config_class,
    get_registered_agents,
)
from prompt_siren.agents.plain import PlainAgentConfig
from prompt_siren.agents.registry import agent_registry
from prompt_siren.registry_base import UnknownComponentError

from ..conftest import create_mock_agent, MockAgent, MockAgentConfig


class TestAgentRegistry:
    """Tests for the agent plugin system."""

    def setup_method(self):
        """Set up test environment by clearing registry and registering mock agent."""
        # Clear the registry for clean tests
        agent_registry._registry.clear()
        agent_registry._entry_points_loaded = False

        # Manually register mock agent for testing
        agent_registry.register("mock", MockAgentConfig, create_mock_agent)

    def test_register_and_get_config_class(self):
        """Test registering an agent and retrieving its config class."""
        # Check that it's listed in registered agents
        assert "mock" in get_registered_agents()

        # Get the config class
        config_class = get_agent_config_class("mock")

        # Check that it's the correct class
        assert config_class == MockAgentConfig

    def test_entry_point_discovery(self):
        """Test that entry points are discovered automatically."""
        # Clear manual registrations
        agent_registry._registry.clear()
        agent_registry._entry_points_loaded = False

        # Check that built-in agents are discovered via entry points
        registered_agents = get_registered_agents()
        assert "plain" in registered_agents

        # Test that we can get the config class
        config_class = get_agent_config_class("plain")
        assert config_class is PlainAgentConfig

    def test_create_agent(self):
        """Test creating an agent from config."""
        # Create a config
        config = MockAgentConfig(name="Custom Mock Agent", custom_parameter="test-value")

        # Create the agent
        agent = create_agent("mock", config)

        # Check that it's the correct type
        assert isinstance(agent, MockAgent)

        # Check that it has the correct values
        assert agent.name == "Custom Mock Agent"
        assert agent.custom_parameter == "test-value"

        # Check that config property returns the original config
        assert agent.config == config
        assert agent.config.name == "Custom Mock Agent"
        assert agent.config.custom_parameter == "test-value"

    def test_missing_agent_type(self):
        """Test error when requesting an unregistered agent type."""
        with pytest.raises(UnknownComponentError):
            get_agent_config_class("non-existent-agent")

        config = MockAgentConfig()
        with pytest.raises(UnknownComponentError):
            create_agent("non-existent-agent", config)

    def test_duplicate_registration(self):
        """Test error when attempting to register the same agent type twice."""
        # Attempt to register the same type again should raise ValueError
        with pytest.raises(ValueError, match="Agent type 'mock' is already registered"):
            agent_registry.register("mock", MockAgentConfig, create_mock_agent)

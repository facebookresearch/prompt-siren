# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Agent registry for composable configuration of agents.

This module provides a registry for agent types and their configurations,
enabling composability through protocols and factory functions rather than
inheritance. This approach allows for different agent types with specialized
configuration types while maintaining type safety.
"""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from ..registry_base import BaseRegistry
from .abstract import AbstractAgent

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for agent factory functions
AgentFactory: TypeAlias = Callable[[ConfigT, None], AbstractAgent]


# Create a global agent registry instance
agent_registry = BaseRegistry[AbstractAgent, None]("agent", "prompt_siren.agents")


# Convenience functions for agent-specific naming
def register_agent(
    agent_type: str,
    config_class: type[ConfigT],
    factory: AgentFactory[ConfigT],
) -> None:
    """Register a new agent type with its config class and factory function."""
    agent_registry.register(agent_type, config_class, factory)


def get_agent_config_class(agent_type: str) -> type[BaseModel]:
    """Get the config class for a given agent type."""
    config_class = agent_registry.get_config_class(agent_type)
    assert config_class is not None, f"Agent type '{agent_type}' must have a config class"
    return config_class


def create_agent(agent_type: str, config: BaseModel) -> AbstractAgent:
    """Create an agent instance for a given agent type and config."""
    return agent_registry.create_component(agent_type, config)


def get_registered_agents() -> list[str]:
    """Get a list of all registered agent types."""
    return agent_registry.get_registered_components()

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Agent implementations and configuration for Siren."""

from .abstract import AbstractAgent
from .plain import PlainAgent, PlainAgentConfig
from .registry import (
    create_agent,
    get_agent_config_class,
    get_registered_agents,
    register_agent,
)

__all__ = [
    # Core abstractions
    "AbstractAgent",
    # Concrete implementations
    "PlainAgent",
    "PlainAgentConfig",
    "create_agent",
    "get_agent_config_class",
    "get_registered_agents",
    # Registry functions
    "register_agent",
]

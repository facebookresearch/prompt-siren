"""Docker client registry for managing Docker client plugins.

This module provides a registry system for Docker client implementations,
allowing different backends to be registered and discovered via entry points.
"""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from ....registry_base import BaseRegistry
from .abstract import AbstractDockerClient

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for Docker client factory functions that accept optional kwargs
DockerClientFactory: TypeAlias = Callable[[ConfigT], AbstractDockerClient]

# Create a global Docker client registry instance using BaseRegistry
docker_client_registry = BaseRegistry[AbstractDockerClient, None](
    "docker_client", "prompt_siren.docker_clients"
)


def register_docker_client(
    client_name: str, config_class: type[ConfigT], factory: DockerClientFactory
) -> None:
    """Register a Docker client with its factory function."""
    docker_client_registry.register(client_name, config_class=config_class, factory=factory)


def get_docker_client_config_class(client_name: str) -> type[BaseModel]:
    """Get the configuration class for a Docker client type"""
    config_class = docker_client_registry.get_config_class(client_name)
    if config_class is None:
        raise RuntimeError(f"Docker client type '{client_name}' must have a config class")
    return config_class


def create_docker_client(client_name: str, config: BaseModel) -> AbstractDockerClient:
    """Create a Docker client instance from a configuration."""
    return docker_client_registry.create_component(client_name, config)


def get_registered_docker_clients() -> list[str]:
    """Get list of all registered Docker client names.

    Returns:
        List of registered client names
    """
    return docker_client_registry.get_registered_components()

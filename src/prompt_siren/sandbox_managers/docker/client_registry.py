"""Docker client registry for managing Docker client plugins.

This module provides a registry system for Docker client implementations,
allowing different backends (local Docker, DES, etc.) to be registered and
discovered via entry points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from ...registry_base import BaseRegistry
from .abstract_client import AbstractDockerClient

# Type alias for Docker client factory functions (no config needed)
DockerClientFactory: TypeAlias = Callable[[], AbstractDockerClient]

# Create a global Docker client registry instance using BaseRegistry
docker_client_registry = BaseRegistry[AbstractDockerClient, None](
    "docker_client", "prompt_siren.docker_clients"
)


def register_docker_client(client_name: str, factory: DockerClientFactory) -> None:
    """Register a Docker client with its factory function.

    Args:
        client_name: Name of the Docker client (e.g., "local", "des").
                    This name is used to identify which client to use.
        factory: Function that takes no arguments and returns an AbstractDockerClient instance

    Raises:
        ValueError: If the client name is already registered

    Example:
        >>> def create_my_client() -> AbstractDockerClient:
        ...     return MyDockerClient()
        >>> register_docker_client("my-client", create_my_client)
    """
    docker_client_registry.register(client_name, factory, config_class=None)


def get_docker_client(client_name: str) -> AbstractDockerClient:
    """Get a Docker client instance by name.

    Args:
        client_name: Name of the registered Docker client

    Returns:
        AbstractDockerClient instance

    Raises:
        KeyError: If the client name is not registered
    """
    return docker_client_registry.create_component(client_name, config=None, context=None)


def get_registered_docker_clients() -> list[str]:
    """Get list of all registered Docker client names.

    Returns:
        List of registered client names
    """
    return docker_client_registry.list_names()

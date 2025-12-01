# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass

from pydantic import BaseModel

from .image_spec import ImageSpec


class ContainerSpec(BaseModel):
    """Specification for any container."""

    image_spec: ImageSpec
    hostname: str | None = None
    ports: list[str] | None = None
    environment: dict[str, str] | None = None
    command: str | list[str] | None = None


@dataclass(frozen=True)
class ContainerSetup:
    """Setup for a single container."""

    name: str  # Container name (e.g., "agent", "attack_server", "db")
    spec: ContainerSpec
    dockerfile_extra: str | None = None


@dataclass(frozen=True)
class NetworkConfig:
    """Network configuration for multi-container tasks."""

    name: str
    internal: bool = False


@dataclass(frozen=True)
class TaskSetup:
    """Setup for tasks with one agent container and optional service containers.

    The agent container is always required and has a dedicated field.
    Service containers are optional and stored in a dict.
    This design makes the agent/service asymmetry explicit and docker-compose friendly.
    """

    task_id: str
    agent_container: ContainerSetup  # Required: container where agent tools execute
    service_containers: dict[str, ContainerSetup]  # Optional: named service containers
    network_config: NetworkConfig | None = None  # Auto-create if service_containers non-empty


SandboxTaskSetup = TaskSetup

# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass
from typing import TypeAlias

ContainerID: TypeAlias = str


@dataclass(frozen=True)
class SandboxState:
    """State for a sandbox with one agent container and optional service containers.

    The agent container is always required and has a dedicated field.
    Service containers are optional and stored in a dict.
    This design makes the agent/service asymmetry explicit and docker-compose friendly.
    """

    agent_container_id: ContainerID  # Required: where agent tools execute
    service_containers: dict[str, ContainerID]  # Optional: named service containers
    execution_id: str  # Internal: links state to TaskSandboxContext for resource tracking
    network_id: str | None = None  # None for single-container, set for multi-container

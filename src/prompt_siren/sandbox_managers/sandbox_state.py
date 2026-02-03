# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass, field
from typing import TypeAlias

ContainerID: TypeAlias = str

# Port bindings map container_port -> host_port
PortBindings: TypeAlias = dict[int, int]


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
    agent_port_bindings: PortBindings = field(default_factory=dict)
    """Actual port bindings for the agent container (container_port -> host_port).

    When dynamic port allocation is used (host_port=0 in ContainerSpec.ports),
    this contains the actual allocated host ports after container creation.
    """

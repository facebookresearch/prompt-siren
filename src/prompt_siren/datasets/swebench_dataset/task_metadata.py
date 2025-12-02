# Copyright (c) Meta Platforms, Inc. and affiliates.
from pydantic import BaseModel, Field

from ...sandbox_managers.sandbox_task_setup import ContainerSpec


class SWEBenchBenignTaskMetadata(BaseModel):
    """Metadata for benign tasks - specifies the main task container and optional service containers."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)


class SWEBenchMaliciousTaskMetadata(BaseModel):
    """Metadata for malicious tasks - specifies attack infrastructure with service containers."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    benign_dockerfile_extra: str | None = None

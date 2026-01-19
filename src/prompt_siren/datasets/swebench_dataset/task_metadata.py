# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from pydantic import BaseModel, Field

from ...sandbox_managers.image_spec import PullImageSpec
from ...sandbox_managers.sandbox_task_setup import ContainerSpec
from .image_tags import apply_registry_prefix, get_pair_image_tag


class SWEBenchBenignTaskMetadata(BaseModel):
    """Metadata for benign tasks - specifies the main task container and optional service containers."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    registry: str | None = None
    """Optional registry prefix for image tags."""


class SWEBenchMaliciousTaskMetadata(BaseModel):
    """Metadata for malicious tasks - specifies attack infrastructure with service containers.

    When benign_dockerfile_extra is set, the pair image should be pre-built using
    the build_images script. The get_pair_image_tag method returns the predictable
    tag for the pre-built image.
    """

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    benign_dockerfile_extra: str | None = None
    registry: str | None = None
    """Optional registry prefix for image tags."""

    def get_pair_image_tag(
        self,
        benign_task_id: str,
        malicious_task_id: str,
    ) -> str | None:
        """Get the pre-built pair image tag for this malicious task combined with a benign task.

        Args:
            benign_task_id: The ID of the benign task this is being paired with
            malicious_task_id: The ID of the malicious task

        Returns:
            The pre-built pair image tag, or None if no modification is needed
        """
        if self.benign_dockerfile_extra is None:
            return None
        return get_pair_image_tag(benign_task_id, malicious_task_id, registry=self.registry)

    def with_registry(self, registry: str | None) -> SWEBenchMaliciousTaskMetadata:
        """Create a copy of this metadata with the registry set.

        Also updates service container image specs with the registry prefix.

        Args:
            registry: Optional registry prefix for image tags

        Returns:
            A new metadata instance with the registry set
        """
        if registry is None:
            return self

        # Update service container image specs with registry prefix
        updated_service_containers: dict[str, ContainerSpec] = {}
        for name, spec in self.service_containers.items():
            if isinstance(spec.image_spec, PullImageSpec):
                new_tag = apply_registry_prefix(spec.image_spec.tag, registry)
                updated_service_containers[name] = ContainerSpec(
                    image_spec=PullImageSpec(tag=new_tag),
                    hostname=spec.hostname,
                    command=spec.command,
                )
            else:
                updated_service_containers[name] = spec

        # Update agent container spec if it uses PullImageSpec
        if isinstance(self.agent_container_spec.image_spec, PullImageSpec):
            new_agent_tag = apply_registry_prefix(
                self.agent_container_spec.image_spec.tag, registry
            )
            updated_agent_spec = ContainerSpec(
                image_spec=PullImageSpec(tag=new_agent_tag),
                hostname=self.agent_container_spec.hostname,
                command=self.agent_container_spec.command,
            )
        else:
            updated_agent_spec = self.agent_container_spec

        return SWEBenchMaliciousTaskMetadata(
            agent_container_spec=updated_agent_spec,
            service_containers=updated_service_containers,
            benign_dockerfile_extra=self.benign_dockerfile_extra,
            registry=registry,
        )

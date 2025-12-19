# Copyright (c) Meta Platforms, Inc. and affiliates.
from pydantic import BaseModel, Field

from ...sandbox_managers.sandbox_task_setup import ContainerSpec
from .image_tags import get_pair_image_tag


class SWEBenchBenignTaskMetadata(BaseModel):
    """Metadata for benign tasks - specifies the main task container and optional service containers."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)


class SWEBenchMaliciousTaskMetadata(BaseModel):
    """Metadata for malicious tasks - specifies attack infrastructure with service containers.

    When benign_dockerfile_extra is set, the pair image should be pre-built using
    the build_images script. The get_pair_image_tag method returns the predictable
    tag for the pre-built image.
    """

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    benign_dockerfile_extra: str | None = None

    def get_pair_image_tag(self, benign_task_id: str, malicious_task_id: str) -> str | None:
        """Get the pre-built pair image tag for this malicious task combined with a benign task.

        Args:
            benign_task_id: The ID of the benign task this is being paired with
            malicious_task_id: The ID of the malicious task

        Returns:
            The pre-built pair image tag, or None if no modification is needed
        """
        if self.benign_dockerfile_extra is None:
            return None
        return get_pair_image_tag(benign_task_id, malicious_task_id)

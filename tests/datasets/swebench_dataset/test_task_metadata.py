# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for SWEBench task metadata classes."""

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.datasets.swebench_dataset.task_metadata import (
    SWEBenchMaliciousTaskMetadata,
)
from prompt_siren.sandbox_managers.image_spec import BuildImageSpec, PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import ContainerSpec


class TestSWEBenchMaliciousTaskMetadataWithRegistry:
    """Tests for SWEBenchMaliciousTaskMetadata.with_registry method."""

    def test_returns_self_when_registry_is_none(self) -> None:
        """Test that with_registry returns self when registry is None."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
        )

        result = metadata.with_registry(None)

        assert result is metadata

    def test_updates_agent_container_pull_image_spec(self) -> None:
        """Test that with_registry updates agent container PullImageSpec with registry prefix."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
        )

        result = metadata.with_registry("my-registry.com/repo")

        assert isinstance(result.agent_container_spec.image_spec, PullImageSpec)
        assert result.agent_container_spec.image_spec.tag == "my-registry.com/repo/agent:latest"

    def test_preserves_agent_container_build_image_spec(self) -> None:
        """Test that with_registry preserves agent container BuildImageSpec (no prefix applied)."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=BuildImageSpec(context_path="/path", tag="agent:latest"),
            ),
            service_containers={},
        )

        result = metadata.with_registry("my-registry.com/repo")

        # BuildImageSpec should not be modified
        assert isinstance(result.agent_container_spec.image_spec, BuildImageSpec)
        assert result.agent_container_spec.image_spec.tag == "agent:latest"

    def test_updates_service_container_pull_image_specs(self) -> None:
        """Test that with_registry updates service container PullImageSpecs with registry prefix."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={
                "attacker": ContainerSpec(
                    image_spec=PullImageSpec(tag="attacker-server:v1"),
                    hostname="attacker.local",
                ),
                "db": ContainerSpec(
                    image_spec=PullImageSpec(tag="postgres:15"),
                ),
            },
        )

        result = metadata.with_registry("my-registry.com/repo")

        assert isinstance(result.service_containers["attacker"].image_spec, PullImageSpec)
        assert (
            result.service_containers["attacker"].image_spec.tag
            == "my-registry.com/repo/attacker-server:v1"
        )
        assert result.service_containers["attacker"].hostname == "attacker.local"

        assert isinstance(result.service_containers["db"].image_spec, PullImageSpec)
        assert result.service_containers["db"].image_spec.tag == "my-registry.com/repo/postgres:15"

    def test_preserves_service_container_build_image_specs(self) -> None:
        """Test that with_registry preserves service container BuildImageSpecs."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={
                "custom": ContainerSpec(
                    image_spec=BuildImageSpec(context_path="/build", tag="custom:latest"),
                ),
            },
        )

        result = metadata.with_registry("my-registry.com/repo")

        # BuildImageSpec should not be modified
        assert isinstance(result.service_containers["custom"].image_spec, BuildImageSpec)
        assert result.service_containers["custom"].image_spec.tag == "custom:latest"

    def test_sets_registry_field(self) -> None:
        """Test that with_registry sets the registry field on the new metadata."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
            registry=None,
        )

        result = metadata.with_registry("my-registry.com/repo")

        assert result.registry == "my-registry.com/repo"

    def test_preserves_benign_dockerfile_extra(self) -> None:
        """Test that with_registry preserves the benign_dockerfile_extra field."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
            benign_dockerfile_extra="RUN echo 'modified'",
        )

        result = metadata.with_registry("my-registry.com/repo")

        assert result.benign_dockerfile_extra == "RUN echo 'modified'"

    def test_returns_new_instance(self) -> None:
        """Test that with_registry returns a new instance, not mutating the original."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
        )

        result = metadata.with_registry("my-registry.com/repo")

        assert result is not metadata
        # Original should be unchanged
        assert metadata.registry is None
        assert isinstance(metadata.agent_container_spec.image_spec, PullImageSpec)
        assert metadata.agent_container_spec.image_spec.tag == "agent:latest"


class TestSWEBenchMaliciousTaskMetadataGetPairImageTag:
    """Tests for SWEBenchMaliciousTaskMetadata.get_pair_image_tag method."""

    def test_returns_none_when_no_dockerfile_extra(self) -> None:
        """Test that get_pair_image_tag returns None when benign_dockerfile_extra is None."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
            benign_dockerfile_extra=None,
        )

        result = metadata.get_pair_image_tag("benign-task-1", "malicious-task-1")

        assert result is None

    def test_returns_tag_when_dockerfile_extra_set(self) -> None:
        """Test that get_pair_image_tag returns a tag when benign_dockerfile_extra is set."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
            benign_dockerfile_extra="RUN echo 'modified'",
        )

        result = metadata.get_pair_image_tag("benign-task-1", "malicious-task-1")

        assert result is not None
        assert "benign-task-1" in result.replace("__", "-")
        assert "malicious-task-1" in result.replace("__", "-")

    def test_includes_registry_prefix_when_set(self) -> None:
        """Test that get_pair_image_tag includes registry prefix when registry is set."""
        metadata = SWEBenchMaliciousTaskMetadata(
            agent_container_spec=ContainerSpec(
                image_spec=PullImageSpec(tag="agent:latest"),
            ),
            service_containers={},
            benign_dockerfile_extra="RUN echo 'modified'",
            registry="my-registry.com/repo",
        )

        result = metadata.get_pair_image_tag("benign-task-1", "malicious-task-1")

        assert result is not None
        assert result.startswith("my-registry.com/repo/")

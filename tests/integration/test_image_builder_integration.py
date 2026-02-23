# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for ImageBuilder with real Docker."""

from pathlib import Path
from uuid import uuid4

import pytest
from prompt_siren.build_images import ImageBuilder
from prompt_siren.sandbox_managers.docker.manager import DockerSandboxManager
from prompt_siren.sandbox_managers.image_spec import (
    BuildImageSpec,
    DerivedImageSpec,
    PullImageSpec,
)
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    TaskSetup,
)

pytestmark = pytest.mark.anyio

FIXTURES_DIR = Path(__file__).parent / "fixtures"


async def _assert_file_in_container(
    docker_client_type: str,
    create_manager_config,
    image_tag: str,
    path: str,
    expected: str,
) -> None:
    config = create_manager_config(docker_client_type, network_enabled=False)
    manager = DockerSandboxManager(config)
    container_spec = ContainerSpec(image_spec=PullImageSpec(tag=image_tag))
    agent_container = ContainerSetup(name="agent", spec=container_spec)
    task_setup = TaskSetup(
        task_id=f"image-builder-{uuid4().hex[:8]}",
        agent_container=agent_container,
        service_containers={},
        network_config=None,
    )

    async with manager.setup_batch([task_setup]):
        async with manager.setup_task(task_setup) as sandbox_state:
            result = await manager.exec(
                sandbox_state.agent_container_id,
                ["cat", path],
            )
            assert result.exit_code == 0
            assert result.stdout is not None
            assert expected in result.stdout


@pytest.mark.docker_integration
class TestImageBuilderIntegration:
    async def test_build_modified_image_creates_expected_file(
        self,
        docker_client,
        docker_client_type: str,
        create_manager_config,
    ) -> None:
        """Verify build_modified_image applies dockerfile_extra to the base image."""
        base_tag = f"prompt-siren-builder-base:{uuid4().hex[:8]}"
        derived_tag = f"prompt-siren-builder-derived:{uuid4().hex[:8]}"

        builder = ImageBuilder(
            docker_client=docker_client,
            rebuild_existing=False,
        )

        await builder.build_from_context(
            context_path=str(FIXTURES_DIR),
            tag=base_tag,
        )

        dockerfile_extra = "RUN echo 'Derived build successful' > /derived-marker.txt"
        await builder.build_modified_image(
            base_tag=base_tag,
            dockerfile_extra=dockerfile_extra,
            output_tag=derived_tag,
        )

        await _assert_file_in_container(
            docker_client_type,
            create_manager_config,
            derived_tag,
            "/derived-marker.txt",
            "Derived build successful",
        )

        await _assert_file_in_container(
            docker_client_type,
            create_manager_config,
            derived_tag,
            "/test-marker.txt",
            "Test build successful",
        )

    async def test_build_all_specs_builds_derived_images(
        self,
        docker_client,
        docker_client_type: str,
        create_manager_config,
    ) -> None:
        """Verify build_all_specs builds derived images after base images."""
        base_tag = f"prompt-siren-builder-all-base:{uuid4().hex[:8]}"
        derived_tag = f"prompt-siren-builder-all-derived:{uuid4().hex[:8]}"

        specs = [
            BuildImageSpec(
                context_path=str(FIXTURES_DIR),
                tag=base_tag,
            ),
            DerivedImageSpec(
                base_image_tag=base_tag,
                dockerfile_extra="RUN echo 'Derived via build_all_specs' > /derived-all.txt",
                tag=derived_tag,
            ),
        ]

        builder = ImageBuilder(
            docker_client=docker_client,
            rebuild_existing=False,
        )

        build_errors = await builder.build_all_specs(specs)
        assert build_errors == []

        await _assert_file_in_container(
            docker_client_type,
            create_manager_config,
            derived_tag,
            "/derived-all.txt",
            "Derived via build_all_specs",
        )

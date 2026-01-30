# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pytest fixtures for Docker integration tests.

External tests use local Docker client only.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import pytest
from prompt_siren.build_images import ImageBuilder
from prompt_siren.sandbox_managers.docker.manager import (
    create_docker_client_from_config,
    DockerSandboxConfig,
)
from prompt_siren.sandbox_managers.docker.plugins import AbstractDockerClient
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError
from prompt_siren.sandbox_managers.image_spec import (
    BuildStage,
    MultiStageBuildImageSpec,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class ManagerConfigFactory(Protocol):
    """Protocol for create_manager_config fixture callable.

    This properly types the callable with optional parameters, unlike
    Callable[[str, bool, str | list[str] | None], DockerSandboxConfig]
    which requires all arguments to be positional.
    """

    def __call__(
        self,
        docker_client_type: str,
        network_enabled: bool = False,
        test_images: str | list[str] | None = None,
    ) -> DockerSandboxConfig: ...


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Override anyio_backend to be module-scoped for integration tests."""
    return "asyncio"


@pytest.fixture(scope="module")
def docker_client_type() -> str:
    """Docker client type for external tests.

    External tests always use local Docker client.
    """
    return "local"


@pytest.fixture(scope="module")
def create_manager_config() -> ManagerConfigFactory:
    """Factory fixture for creating DockerSandboxConfig.

    External version only supports local Docker.

    Returns:
        A callable that creates DockerSandboxConfig instances
    """

    def _create_config(
        docker_client_type: str,
        network_enabled: bool = False,
        test_images: str | list[str] | None = None,
    ) -> DockerSandboxConfig:
        """Create DockerSandboxConfig for external tests (local only).

        Args:
            docker_client_type: Client type ('local' for external)
            network_enabled: Whether to enable network access
            test_images: Test image(s) - unused in external tests

        Returns:
            DockerSandboxConfig configured for local Docker

        Raises:
            ValueError: If docker_client_type is not 'local' (in external)
        """
        if docker_client_type != "local":
            raise ValueError(
                f"External tests only support 'local' client type. Got: {docker_client_type}"
            )

        return DockerSandboxConfig(network_enabled=network_enabled, docker_client="local")

    return _create_config


@pytest.fixture(scope="module")
async def docker_client() -> AsyncIterator[AbstractDockerClient]:
    """Provide a shared Docker client for all integration tests in the module.

    Module-scoped to avoid creating a new connection for each test.
    External tests always use local Docker client plugin via registry.
    """
    client = create_docker_client_from_config("local", {})
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="module")
async def test_image(docker_client: AbstractDockerClient):
    """Pull the test image once for all tests in the module.

    Uses debian:bookworm-slim for small size with bash support.
    Module-scoped to avoid repeated image checks.
    """
    image = "debian:bookworm-slim"

    # Check if image exists locally before attempting pull (avoids rate limits)
    try:
        await docker_client.inspect_image(image)
    except DockerClientError:
        # Image doesn't exist, pull it
        await docker_client.pull_image(image)
    return image


async def _ensure_image_available(docker_client: AbstractDockerClient, image: str) -> None:
    """Ensure an image is available locally, pulling if necessary."""
    try:
        await docker_client.inspect_image(image)
    except DockerClientError:
        # Image doesn't exist, pull it
        await docker_client.pull_image(image)


@pytest.fixture(scope="module")
async def multistage_test_images(docker_client: AbstractDockerClient):
    """Pull the images needed for multi-stage build tests.

    Returns a list of images needed for multi-stage builds.
    Module-scoped to avoid repeated image checks.
    """
    images = [
        "debian:bookworm-slim",
        "alpine:3.19",
    ]

    for image in images:
        await _ensure_image_available(docker_client, image)
    return images


@pytest.fixture(scope="module")
async def build_test_images(docker_client: AbstractDockerClient) -> None:
    """Pre-build all test images needed for TestImageBuilding tests.

    This fixture builds images from the fixtures directory before
    the tests run. The ImageCache no longer builds images at runtime,
    so they must be pre-built.
    """
    builder = ImageBuilder(
        docker_client=docker_client,
        rebuild_existing=False,
    )

    # Build the basic test image
    await builder.build_from_context(
        context_path=str(FIXTURES_DIR),
        tag="prompt-siren-test-build:latest",
    )

    # Build the image with build args
    await builder.build_from_context(
        context_path=str(FIXTURES_DIR),
        tag="prompt-siren-test-build-args:latest",
        dockerfile_path="Dockerfile.dev",
        build_args={"TEST_ARG": "custom_value"},
    )

    # Build the mixed test image
    await builder.build_from_context(
        context_path=str(FIXTURES_DIR),
        tag="prompt-siren-test-mixed:latest",
    )

    # Build the network test image
    await builder.build_from_context(
        context_path=str(FIXTURES_DIR),
        tag="prompt-siren-network-test:latest",
        dockerfile_path="Dockerfile.network",
    )


@pytest.fixture(scope="module")
async def build_multistage_test_images(
    docker_client: AbstractDockerClient,
    multistage_test_images: list[str],
) -> None:
    """Pre-build all multi-stage test images needed for TestMultiStageBuild tests.

    This fixture builds the multi-stage images before the tests run.
    """
    builder = ImageBuilder(
        docker_client=docker_client,
        rebuild_existing=False,
    )

    # Build multi-stage test images
    base_tag = "test-multistage-base:latest"
    env_tag = "test-multistage-env:latest"
    instance_tag = "test-multistage-instance:latest"

    stages = [
        BuildStage(
            tag=base_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "base"),
            cache_key=base_tag,
        ),
        BuildStage(
            tag=env_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "env"),
            parent_tag=base_tag,
            cache_key=env_tag,
        ),
        BuildStage(
            tag=instance_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "instance"),
            parent_tag=env_tag,
        ),
    ]

    spec = MultiStageBuildImageSpec(stages=stages)
    await builder.build_all_specs([spec])

    # Build caching test images
    cache_base_tag = "test-multistage-cache-base:latest"
    cache_env_tag = "test-multistage-cache-env:latest"
    cache_instance_tag = "test-multistage-cache-instance:latest"

    cache_stages = [
        BuildStage(
            tag=cache_base_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "base"),
            cache_key=cache_base_tag,
        ),
        BuildStage(
            tag=cache_env_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "env"),
            parent_tag=cache_base_tag,
            cache_key=cache_env_tag,
        ),
        BuildStage(
            tag=cache_instance_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "instance"),
            parent_tag=cache_env_tag,
        ),
    ]

    cache_spec = MultiStageBuildImageSpec(stages=cache_stages)
    await builder.build_all_specs([cache_spec])

    # Build shared stages test images
    shared_base_tag = "test-shared-base:latest"
    shared_env_tag = "test-shared-env:latest"
    shared_instance1_tag = "test-shared-instance1:latest"
    shared_instance2_tag = "test-shared-instance2:latest"

    # Build shared base and env stages
    shared_base_stages = [
        BuildStage(
            tag=shared_base_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "base"),
            cache_key=shared_base_tag,
        ),
        BuildStage(
            tag=shared_env_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "env"),
            parent_tag=shared_base_tag,
            cache_key=shared_env_tag,
        ),
        BuildStage(
            tag=shared_instance1_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "instance"),
            parent_tag=shared_env_tag,
        ),
    ]
    shared_spec1 = MultiStageBuildImageSpec(stages=shared_base_stages)
    await builder.build_all_specs([shared_spec1])

    # Build instance2 (reuses base/env from cache)
    shared_instance2_stages = [
        BuildStage(
            tag=shared_base_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "base"),
            cache_key=shared_base_tag,
        ),
        BuildStage(
            tag=shared_env_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "env"),
            parent_tag=shared_base_tag,
            cache_key=shared_env_tag,
        ),
        BuildStage(
            tag=shared_instance2_tag,
            context_path=str(FIXTURES_DIR / "multistage" / "instance"),
            parent_tag=shared_env_tag,
        ),
    ]
    shared_spec2 = MultiStageBuildImageSpec(stages=shared_instance2_stages)
    await builder.build_all_specs([shared_spec2])

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pytest fixtures for Docker integration tests.

External tests use local Docker client only.
"""

from collections.abc import AsyncIterator
from typing import Protocol

import pytest
from prompt_siren.sandbox_managers.docker.manager import (
    create_docker_client_from_config,
    DockerSandboxConfig,
)
from prompt_siren.sandbox_managers.docker.plugins import AbstractDockerClient
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError


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

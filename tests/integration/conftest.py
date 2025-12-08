# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pytest fixtures for Docker integration tests.

External tests use local Docker client only.
For DES support, see internal/tests/integration/conftest.py
"""

from collections.abc import AsyncIterator, Callable

import pytest
from prompt_siren.sandbox_managers.docker.abstract_client import AbstractDockerClient
from prompt_siren.sandbox_managers.docker.errors import DockerClientError
from prompt_siren.sandbox_managers.docker.manager import DockerSandboxConfig
from prompt_siren.sandbox_managers.docker.plugins.local_client import LocalDockerClient


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Override anyio_backend to be module-scoped for integration tests."""
    return "asyncio"


@pytest.fixture(scope="module")
def docker_client_type() -> str:
    """Docker client type for external tests.

    External tests always use local Docker client.
    For DES support, run tests from internal directory.
    """
    return "local"


@pytest.fixture(scope="module")
def skip_if_des_unavailable(docker_client_type: str):
    """Skip test if DES client is requested but unavailable.

    This is a no-op for external tests (always local).
    Kept for compatibility with shared test code.
    """


@pytest.fixture(scope="module")
def create_manager_config() -> Callable[[str, bool, str | list[str] | None], DockerSandboxConfig]:
    """Factory fixture for creating DockerSandboxConfig.

    External version only supports local Docker.
    Internal version (see internal/tests/integration/conftest.py) adds DES support.

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
            test_images: Test image(s) - used by internal DES tests

        Returns:
            DockerSandboxConfig configured for local Docker

        Raises:
            ValueError: If docker_client_type is not 'local' (in external)
        """
        if docker_client_type != "local":
            raise ValueError(
                f"External tests only support 'local' client type. "
                f"For DES support, run tests from internal directory. "
                f"Got: {docker_client_type}"
            )

        return DockerSandboxConfig(network_enabled=network_enabled, docker_client="local")

    return _create_config


@pytest.fixture(scope="module")
async def docker_client() -> AsyncIterator[AbstractDockerClient]:
    """Provide a shared Docker client for all integration tests in the module.

    Module-scoped to avoid creating a new connection for each test.
    External tests always use local Docker client plugin.
    """
    client = LocalDockerClient()
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
    image = "vmvm-registry.fbinfra.net/debian:bookworm-slim"

    # Check if image exists locally before attempting pull (avoids rate limits)
    try:
        await docker_client.inspect_image(image)
    except DockerClientError:
        # Image doesn't exist, pull it
        await docker_client.pull_image(image)
    return image


@pytest.fixture(scope="module")
async def multistage_test_images(docker_client: AbstractDockerClient):
    """Pull the images needed for multi-stage build tests.

    Returns a list of images needed for multi-stage builds.
    Module-scoped to avoid repeated image checks.
    """
    images = [
        "vmvm-registry.fbinfra.net/debian:bookworm-slim",
        "vmvm-registry.fbinfra.net/alpine:3.19",
    ]

    for image in images:
        # Check if image exists locally before attempting pull (avoids rate limits)
        try:
            await docker_client.inspect_image(image)
        except DockerClientError:
            # Image doesn't exist, pull it
            await docker_client.pull_image(image)
    return images

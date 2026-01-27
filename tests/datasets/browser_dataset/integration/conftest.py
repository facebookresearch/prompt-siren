# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pytest fixtures for browser dataset integration tests.

These fixtures require Docker to be running.
"""

from collections.abc import AsyncIterator

import pytest
from prompt_siren.sandbox_managers.docker.manager import (
    create_docker_client_from_config,
    DockerSandboxConfig,
    DockerSandboxManager,
)
from prompt_siren.sandbox_managers.docker.plugins import AbstractDockerClient
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    TaskSetup,
)

# Browser container image (Headless Chrome with CDP support)
# chromedp/headless-shell is Debian-based and designed for CDP usage
BROWSER_IMAGE = "chromedp/headless-shell:latest"
CDP_PORT = 9222


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Override anyio_backend to be module-scoped for integration tests."""
    return "asyncio"


@pytest.fixture(scope="module")
async def docker_client() -> AsyncIterator[AbstractDockerClient]:
    """Provide a shared Docker client for all integration tests in the module."""
    client = create_docker_client_from_config("local", {})
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="module")
async def browser_image(docker_client: AbstractDockerClient) -> str:
    """Pull the browser image once for all tests in the module."""
    try:
        await docker_client.inspect_image(BROWSER_IMAGE)
    except DockerClientError:
        # Image doesn't exist, pull it
        await docker_client.pull_image(BROWSER_IMAGE)
    return BROWSER_IMAGE


@pytest.fixture(scope="module")
def docker_sandbox_config() -> DockerSandboxConfig:
    """Create DockerSandboxConfig for browser tests."""
    return DockerSandboxConfig(network_enabled=True, docker_client="local")


@pytest.fixture(scope="module")
def docker_sandbox_manager(docker_sandbox_config: DockerSandboxConfig) -> DockerSandboxManager:
    """Create DockerSandboxManager for browser tests."""
    return DockerSandboxManager(docker_sandbox_config)


@pytest.fixture(scope="module")
def browser_task_setup(browser_image: str) -> TaskSetup:
    """Create a TaskSetup for browser container tests.

    Note: chromedp/headless-shell has a built-in ENTRYPOINT that runs
    headless Chrome with CDP on port 9222. We don't specify a command.
    """
    browser_spec = ContainerSpec(
        image_spec=PullImageSpec(tag=browser_image),
        hostname="browser",
        ports={CDP_PORT: CDP_PORT},
    )

    return TaskSetup(
        task_id="browser-integration-test",
        agent_container=ContainerSetup(name="browser", spec=browser_spec),
        service_containers={},
        network_config=NetworkConfig(name="browser-test-network", internal=False),
    )

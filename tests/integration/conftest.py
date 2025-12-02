# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pytest fixtures for Docker integration tests."""

from collections.abc import AsyncIterator

import pytest
from aiodocker import Docker


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Override anyio_backend to be module-scoped for integration tests."""
    return "asyncio"


@pytest.fixture(scope="module")
async def docker_client() -> AsyncIterator[Docker]:
    """Provide a shared Docker client for all integration tests in the module.

    Module-scoped to avoid creating a new connection for each test.
    """
    docker = Docker()
    try:
        yield docker
    finally:
        await docker.close()


@pytest.fixture(scope="module")
async def test_image(docker_client: Docker):
    """Pull the test image once for all tests in the module.

    Uses debian:bookworm-slim for small size with bash support.
    Module-scoped to avoid repeated image checks.
    """
    image = "debian:bookworm-slim"
    # Check if image exists locally before attempting pull (avoids rate limits)
    try:
        await docker_client.images.inspect(image)
    except Exception:
        # Image doesn't exist, pull it
        await docker_client.images.pull(from_image=image)
    return image

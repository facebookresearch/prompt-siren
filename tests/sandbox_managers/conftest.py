# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared fixtures and utilities for sandbox manager tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    TaskSetup,
)


@pytest.fixture
def mock_docker():
    """Create a mock Docker client."""
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.images = MagicMock()
    return docker


def setup_task_for_testing(task_id: str, image_tag: str) -> TaskSetup:
    """Helper to create task setup for tests.

    Returns the task setup object to be passed directly to setup_task().
    """
    container_spec = ContainerSpec(image_spec=PullImageSpec(tag=image_tag))
    agent_container = ContainerSetup(name="agent", spec=container_spec)
    return TaskSetup(
        task_id=task_id,
        agent_container=agent_container,
        service_containers={},
        network_config=None,
    )


def create_mock_container(container_id: str, include_config: bool = True) -> MagicMock:
    """Create a mock Docker container with given ID.

    Args:
        container_id: The container ID to use
        include_config: Whether to include full config structure in show() response
    """
    container = MagicMock()

    # Build show() response
    show_response: dict[str, Any] = {"Id": container_id}
    if include_config:
        # Include structure expected by clone_container helper
        show_response["Config"] = {
            "Env": ["PATH=/usr/bin"],
            "Cmd": ["/bin/sh"],
        }
        show_response["HostConfig"] = {}

    container.show = AsyncMock(return_value=show_response)
    container.stop = AsyncMock()
    container.delete = AsyncMock()
    container.start = AsyncMock()
    container.commit = AsyncMock()
    container.docker = MagicMock()  # For clone operations
    return container

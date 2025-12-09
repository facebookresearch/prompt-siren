# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for DockerSandboxManager using mocks.

These tests focus on the manager's orchestration logic rather than delegation.
Delegation to ImageCache, TaskSandboxContext, and exec_in_container is covered
by integration tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prompt_siren.sandbox_managers.docker.manager import (
    DockerSandboxConfig,
    DockerSandboxManager,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_state import SandboxState
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    TaskSetup,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def manager() -> DockerSandboxManager:
    """Create a DockerSandboxManager instance."""
    config = DockerSandboxConfig(network_enabled=False)
    return DockerSandboxManager(config)


@pytest.fixture
def task_setup() -> TaskSetup:
    """Create a basic task setup."""
    container_spec = ContainerSpec(image_spec=PullImageSpec(tag="python:3.10-slim"))
    agent_container = ContainerSetup(name="agent", spec=container_spec)
    return TaskSetup(
        task_id="test-task",
        agent_container=agent_container,
        service_containers={},
        network_config=None,
    )


class TestDockerSandboxManagerCleanup:
    """Tests for cleanup orchestration."""

    @patch("prompt_siren.sandbox_managers.docker.manager.create_docker_client_from_config")
    async def test_setup_batch_cleans_up_contexts(
        self, mock_get_client, manager: DockerSandboxManager, task_setup: TaskSetup
    ):
        """Test that setup_batch cleans up all task contexts on exit."""
        mock_docker = AsyncMock()
        mock_docker.close = AsyncMock()
        mock_docker.inspect_image = AsyncMock()

        # Create mock container (AbstractContainer interface)
        mock_container = AsyncMock()
        mock_container.show = AsyncMock(
            return_value={"Id": "test-container-id", "State": {"Running": True}}
        )
        mock_container.start = AsyncMock()
        mock_container.stop = AsyncMock()
        mock_container.delete = AsyncMock()

        mock_docker.create_container = AsyncMock(return_value=mock_container)
        mock_get_client.return_value = mock_docker

        async with manager.setup_batch([task_setup]):
            # Create a task (which creates a context)
            async with manager.setup_task(task_setup):
                pass

        # Verify container was cleaned up
        mock_container.stop.assert_called_once()
        mock_container.delete.assert_called_once()


class TestDockerSandboxManagerGuardConditions:
    """Tests for guard conditions (RuntimeError when called outside batch)."""

    async def test_setup_task_outside_batch_raises_error(
        self, manager: DockerSandboxManager, task_setup: TaskSetup
    ):
        """Test that calling setup_task outside setup_batch raises RuntimeError."""
        with pytest.raises(RuntimeError, match="setup_task called outside of setup_batch context"):
            async with manager.setup_task(task_setup):
                pass

    async def test_clone_outside_batch_raises_error(self, manager: DockerSandboxManager):
        """Test that calling clone_sandbox_state outside setup_batch raises RuntimeError."""
        mock_state = SandboxState(
            agent_container_id="test-id",
            service_containers={},
            execution_id="test-exec-id",
            network_id=None,
        )

        with pytest.raises(
            RuntimeError, match="clone_sandbox_state called outside of setup_batch context"
        ):
            await manager.clone_sandbox_state(mock_state)

    async def test_exec_outside_batch_raises_error(self, manager: DockerSandboxManager):
        """Test that calling exec outside setup_batch raises RuntimeError."""
        with pytest.raises(RuntimeError, match="exec called outside of setup_batch context"):
            await manager.exec("test-container-id", ["echo", "test"])


class TestDockerSandboxManagerContextLookup:
    """Tests for context lookup logic."""

    @patch("prompt_siren.sandbox_managers.docker.manager.create_docker_client_from_config")
    async def test_clone_invalid_execution_id_raises_error(
        self, mock_get_client, manager: DockerSandboxManager, task_setup: TaskSetup
    ):
        """Test that cloning with invalid execution_id raises ValueError."""
        mock_docker = AsyncMock()
        mock_docker.close = AsyncMock()
        mock_docker.inspect_image = AsyncMock()
        mock_get_client.return_value = mock_docker

        mock_state = SandboxState(
            agent_container_id="test-id",
            service_containers={},
            execution_id="invalid-exec-id",
            network_id=None,
        )

        async with manager.setup_batch([task_setup]):
            with pytest.raises(ValueError, match="Cannot clone sandbox state"):
                await manager.clone_sandbox_state(mock_state)


class TestDockerSandboxManagerConcurrency:
    """Tests for concurrent task execution."""

    @patch("prompt_siren.sandbox_managers.docker.manager.create_docker_client_from_config")
    async def test_parallel_tasks_with_same_task_id(
        self, mock_get_client, manager: DockerSandboxManager, task_setup: TaskSetup
    ):
        """Test that parallel tasks with same task_id get unique execution_ids."""
        mock_docker = AsyncMock()
        mock_docker.close = AsyncMock()
        mock_docker.inspect_image = AsyncMock()

        # Track execution IDs
        execution_ids = []

        # Create multiple mock containers (AbstractContainer interface)
        containers = []
        for i in range(3):
            mock_container = AsyncMock()
            mock_container.show = AsyncMock(
                return_value={"Id": f"container-{i}", "State": {"Running": True}}
            )
            mock_container.start = AsyncMock()
            mock_container.stop = AsyncMock()
            mock_container.delete = AsyncMock()
            containers.append(mock_container)

        mock_docker.create_container = AsyncMock(side_effect=containers)
        mock_get_client.return_value = mock_docker

        async def run_task():
            async with manager.setup_task(task_setup) as sandbox_state:
                execution_ids.append(sandbox_state.execution_id)
                return sandbox_state.agent_container_id

        async with manager.setup_batch([task_setup]):
            # Run 3 tasks concurrently with same task_id

            results = await asyncio.gather(run_task(), run_task(), run_task())

            # Verify all have unique container IDs and execution IDs
            assert len(results) == 3
            assert len(set(results)) == 3  # Unique container IDs
            assert len(set(execution_ids)) == 3  # Unique execution IDs


class TestDockerSandboxManagerCloning:
    """Tests for container cloning functionality."""

    @patch("prompt_siren.sandbox_managers.docker.manager.create_docker_client_from_config")
    async def test_clone_preserves_custom_command(
        self, mock_get_client, manager: DockerSandboxManager, task_setup: TaskSetup
    ):
        """Test that cloning a container preserves custom command."""
        # Setup mock Docker client (AbstractDockerClient interface)
        mock_docker = AsyncMock()
        mock_docker.close = AsyncMock()
        mock_docker.inspect_image = AsyncMock()
        mock_docker.delete_image = AsyncMock()
        mock_get_client.return_value = mock_docker

        # Custom command to test
        custom_command = ["/bin/bash", "-c", "while true; do echo 'test'; sleep 1; done"]

        # Mock source container (AbstractContainer interface)
        source_container = AsyncMock()
        source_container.show = AsyncMock(
            return_value={
                "Id": "source-container-id",
                "Config": {
                    "Cmd": custom_command,
                    "Env": ["PATH=/usr/bin", "TEST=value"],
                    "Hostname": "test-host",
                },
                "HostConfig": {"NetworkMode": "none"},
                "State": {"Running": True},
            }
        )
        source_container.start = AsyncMock()
        source_container.stop = AsyncMock()
        source_container.delete = AsyncMock()
        source_container.commit = AsyncMock()

        # Mock cloned container (AbstractContainer interface)
        cloned_container = AsyncMock()
        cloned_container.show = AsyncMock(
            return_value={
                "Id": "cloned-container-id",
                "Config": {
                    "Cmd": custom_command,
                    "Env": ["PATH=/usr/bin", "TEST=value"],
                    "Hostname": "test-host",
                },
                "HostConfig": {"NetworkMode": "none"},
                "State": {"Running": True},
            }
        )
        cloned_container.start = AsyncMock()
        cloned_container.stop = AsyncMock()
        cloned_container.delete = AsyncMock()

        # Setup container creation sequence
        mock_docker.create_container = AsyncMock(side_effect=[source_container, cloned_container])
        mock_docker.get_container = AsyncMock(return_value=source_container)

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as source_state:
                # Clone the container
                cloned_state = await manager.clone_sandbox_state(source_state)

                # Verify clone has different container ID but same execution_id
                assert cloned_state.agent_container_id != source_state.agent_container_id
                assert cloned_state.execution_id == source_state.execution_id

                # Verify commit was called to create temp image
                source_container.commit.assert_called_once()
                commit_call = source_container.commit.call_args
                assert "temp-clone-" in commit_call.kwargs["repository"]

                # Verify new container was created with custom command preserved
                create_calls = mock_docker.create_container.call_args_list
                assert len(create_calls) == 2  # Source + clone

                clone_create_call = create_calls[1]  # Second call is the clone
                clone_config = clone_create_call.kwargs["config"]

                # Verify command is preserved
                assert clone_config["Cmd"] == custom_command

                # Verify environment and hostname are preserved
                assert clone_config["Env"] == ["PATH=/usr/bin", "TEST=value"]
                assert clone_config["Hostname"] == "test-host"

                # Verify cloned container was started
                cloned_container.start.assert_called_once()

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for DockerSandboxManager with real Docker.

These tests use real Docker containers and verify the new implementation.
Tests are designed to reuse containers where possible while maintaining isolation.

Run with: pytest -vx -m docker_integration tests/integration/test_docker_manager_integration.py
Skip with: pytest -vx -m "not docker_integration"
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from aiodocker import Docker
from prompt_siren.sandbox_managers.abstract import AbstractSandboxManager
from prompt_siren.sandbox_managers.docker.manager import (
    DockerSandboxConfig,
    DockerSandboxManager,
)
from prompt_siren.sandbox_managers.image_spec import (
    BuildImageSpec,
    BuildStage,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from prompt_siren.sandbox_managers.sandbox_state import SandboxState
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    TaskSetup,
)

pytestmark = pytest.mark.anyio


# ==================== Shared Fixtures for Container Reuse ====================


@pytest.fixture(scope="module")
async def basic_sandbox_manager(
    test_image: str,
) -> AsyncIterator[tuple[AbstractSandboxManager, TaskSetup]]:
    """Create a sandbox manager with batch context for basic tests.

    Module-scoped to reuse across tests for performance.
    """
    config = DockerSandboxConfig(network_enabled=False)
    manager = DockerSandboxManager(config)

    container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
    agent_container = ContainerSetup(name="agent", spec=container_spec)
    task_setup = TaskSetup(
        task_id="basic-test",
        agent_container=agent_container,
        service_containers={},
        network_config=None,
    )

    async with manager.setup_batch([task_setup]):
        yield manager, task_setup


@pytest.fixture(scope="module")
async def shared_container(
    basic_sandbox_manager: tuple[AbstractSandboxManager, TaskSetup],
) -> AsyncIterator[tuple[AbstractSandboxManager, SandboxState]]:
    """Create a shared container for read-only tests.

    Module-scoped to allow many tests to share the same container.
    Tests must use unique temp paths to avoid conflicts.
    """
    manager, task_setup = basic_sandbox_manager

    async with manager.setup_task(task_setup) as sandbox_state:
        yield manager, sandbox_state


@pytest.fixture
async def test_tmp_path(
    request: pytest.FixtureRequest,
    shared_container: tuple[AbstractSandboxManager, SandboxState],
) -> Path:
    """Generate a unique temporary path for each test.

    Ensures test isolation when using shared containers.
    """
    manager, sandbox_state = shared_container

    # Sanitize test name for use in path
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
    # Generate unique path with UUID to avoid conflicts
    unique_id = uuid4().hex[:8]
    path = Path("/tmp") / f"test_{test_name}_{unique_id}"

    # Create the directory in the container
    await manager.exec(
        sandbox_state.agent_container_id,
        ["mkdir", "-p", str(path)],
    )

    return path


# ==================== Basic Container Tests ====================


@pytest.mark.docker_integration
class TestBasicContainerOperations:
    """Tests for basic container creation, execution, and cleanup."""

    async def test_create_and_exec_in_container(
        self,
        shared_container: tuple[AbstractSandboxManager, SandboxState],
        test_tmp_path: Path,
    ):
        """Test creating a container and executing commands in it."""
        manager, sandbox_state = shared_container

        # Execute a simple command
        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["echo", "Hello from Docker"],
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "Hello from Docker" in result.stdout

        # Execute command with working directory
        await manager.exec(
            sandbox_state.agent_container_id,
            ["touch", f"{test_tmp_path}/test.txt"],
        )
        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["ls", str(test_tmp_path)],
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "test.txt" in result.stdout

    async def test_exec_with_special_characters(
        self,
        shared_container: tuple[AbstractSandboxManager, SandboxState],
    ):
        """Test executing commands with special characters."""
        manager, sandbox_state = shared_container

        # Test with spaces
        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["echo", "hello world with spaces"],
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "hello world with spaces" in result.stdout

        # Test with quotes
        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["echo", "it's a 'quoted' string"],
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "it's a 'quoted' string" in result.stdout

        # Test that semicolons are literal (not command separator)
        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["echo", "hello; rm -rf /"],
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "hello; rm -rf /" in result.stdout

    async def test_exec_with_stdin(
        self,
        shared_container: tuple[AbstractSandboxManager, SandboxState],
    ):
        """Test executing command with stdin input."""
        manager, sandbox_state = shared_container

        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["cat"],
            stdin="test input",
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "test input" in result.stdout

    async def test_exec_with_env_vars(
        self,
        shared_container: tuple[AbstractSandboxManager, SandboxState],
    ):
        """Test executing command with environment variables."""
        manager, sandbox_state = shared_container

        result = await manager.exec(
            sandbox_state.agent_container_id,
            ["sh", "-c", "echo $TEST_VAR"],
            env={"TEST_VAR": "test_value"},
        )
        assert result.exit_code == 0
        assert result.stdout is not None
        assert "test_value" in result.stdout


# ==================== Multi-Container Networking Tests ====================


@pytest.mark.docker_integration
class TestMultiContainerNetworking:
    """Tests for multi-container setups with networking."""

    async def test_multi_container_dns_resolution_and_communication(self, test_image: str):
        """Test that containers on the same network can resolve each other by hostname."""
        config = DockerSandboxConfig(network_enabled=True)
        manager = DockerSandboxManager(config)

        # Use custom test image with netcat pre-installed
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        network_image_spec = BuildImageSpec(
            context_path=fixtures_dir,
            dockerfile_path="Dockerfile.network",
            tag="prompt-siren-network-test:latest",
        )

        # Create two containers with specific hostnames
        agent_spec = ContainerSpec(
            image_spec=network_image_spec,
            hostname="benign-host",
        )
        service_spec = ContainerSpec(
            image_spec=network_image_spec,
            hostname="service-host",
        )
        agent_container = ContainerSetup(name="agent", spec=agent_spec)
        service_container = ContainerSetup(name="service", spec=service_spec)

        task_setup = TaskSetup(
            task_id="dns-test",
            agent_container=agent_container,
            service_containers={"service": service_container},
            network_config=NetworkConfig(name="test-dns", internal=True),
        )

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as sandbox_state:
                agent_id = sandbox_state.agent_container_id
                service_id = sandbox_state.service_containers["service"]

                # Verify network was created
                assert sandbox_state.network_id is not None

                # Test DNS resolution (agent -> service)
                result = await manager.exec(agent_id, ["getent", "ahosts", "service-host"])
                assert result.exit_code == 0
                assert result.stdout is not None
                assert "service-host" in result.stdout

                # Test actual network connectivity using netcat
                # Start listener on service container
                await manager.exec(
                    service_id,
                    ["sh", "-c", "nc -l -p 8080 > /tmp/received.txt &"],
                )
                await asyncio.sleep(0.5)

                # Send data from agent to service
                result = await manager.exec(
                    agent_id,
                    ["sh", "-c", "echo 'test-message' | nc -w 2 service-host 8080"],
                )
                assert result.exit_code == 0

                # Verify message was received
                await asyncio.sleep(0.5)
                result = await manager.exec(service_id, ["cat", "/tmp/received.txt"])
                assert result.exit_code == 0
                assert result.stdout is not None
                assert "test-message" in result.stdout

    async def test_network_disabled_creates_internal_network(self, test_image: str):
        """Test that network_enabled=False creates internal-only network for multi-container."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(
            image_spec=PullImageSpec(tag=test_image),
            hostname="test-host",
        )
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        service_container = ContainerSetup(name="service", spec=container_spec)

        task_setup = TaskSetup(
            task_id="internal-network-test",
            agent_container=agent_container,
            service_containers={"service": service_container},
            network_config=None,  # Will create default internal network
        )

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as sandbox_state:
                # Verify network was created (needed for multi-container)
                assert sandbox_state.network_id is not None

                # Verify network is internal by inspecting it
                docker = Docker()
                try:
                    network = await docker.networks.get(sandbox_state.network_id)
                    network_info = await network.show()
                    assert network_info["Internal"] is True
                finally:
                    await docker.close()


# ==================== Container Cloning Tests ====================


@pytest.mark.docker_integration
class TestContainerCloning:
    """Tests for container cloning functionality."""

    async def test_clone_single_container(self, test_image: str):
        """Test cloning a single container creates snapshot."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="clone-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as source_state:
                source_id = source_state.agent_container_id

                # Make a change in source container
                await manager.exec(
                    source_id,
                    ["sh", "-c", "echo 'source content' > /tmp/test.txt"],
                )

                # Clone the container
                cloned_state = await manager.clone_sandbox_state(source_state)
                cloned_id = cloned_state.agent_container_id

                # Verify clone has different container ID but same execution_id
                # (execution_id identifies the TaskSandboxContext, not individual containers)
                assert cloned_id != source_id
                assert cloned_state.execution_id == source_state.execution_id

                # Verify clone has the same content (snapshot)
                result = await manager.exec(cloned_id, ["cat", "/tmp/test.txt"])
                assert result.exit_code == 0
                assert result.stdout is not None
                assert "source content" in result.stdout

                # Modify source after cloning
                await manager.exec(
                    source_id,
                    ["sh", "-c", "echo 'modified' > /tmp/test.txt"],
                )

                # Clone should still have original content
                result = await manager.exec(cloned_id, ["cat", "/tmp/test.txt"])
                assert result.exit_code == 0
                assert result.stdout is not None
                assert "source content" in result.stdout
                assert "modified" not in result.stdout

    async def test_clone_container_with_custom_command(self, test_image: str):
        """Test cloning a container with custom command preserves and runs the command."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        # Create container with custom command that keeps running
        custom_command = [
            "/bin/bash",
            "-c",
            "while true; do echo 'custom-process-marker'; sleep 2; done",
        ]
        container_spec = ContainerSpec(
            image_spec=PullImageSpec(tag=test_image),
            command=custom_command,
        )
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="clone-custom-cmd-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        docker = Docker()
        try:
            async with manager.setup_batch([task_setup]):
                async with manager.setup_task(task_setup) as source_state:
                    source_id = source_state.agent_container_id

                    # Verify source container's command via Docker API
                    source_container = await docker.containers.get(source_id)
                    source_info = await source_container.show()
                    source_cmd = source_info["Config"]["Cmd"]
                    assert source_cmd == custom_command

                    # Verify command is actually running in source container
                    # Use /proc filesystem which is available in all Linux containers
                    await asyncio.sleep(0.5)  # Give process time to start
                    cmdline_result = await manager.exec(
                        source_id,
                        ["sh", "-c", "cat /proc/*/cmdline | tr '\\0' '\\n'"],
                    )
                    assert cmdline_result.exit_code == 0
                    assert cmdline_result.stdout is not None
                    assert "custom-process-marker" in cmdline_result.stdout

                    # Clone the container
                    cloned_state = await manager.clone_sandbox_state(source_state)
                    cloned_id = cloned_state.agent_container_id

                    # Verify clone has different container ID
                    assert cloned_id != source_id

                    # Verify cloned container's command via Docker API
                    cloned_container = await docker.containers.get(cloned_id)
                    cloned_info = await cloned_container.show()
                    cloned_cmd = cloned_info["Config"]["Cmd"]
                    assert cloned_cmd == custom_command

                    # Verify both containers are running
                    assert source_info["State"]["Running"] is True
                    assert cloned_info["State"]["Running"] is True

                    # Verify command is actually running in cloned container
                    cloned_cmdline_result = await manager.exec(
                        cloned_id,
                        ["sh", "-c", "cat /proc/*/cmdline | tr '\\0' '\\n'"],
                    )
                    assert cloned_cmdline_result.exit_code == 0
                    assert cloned_cmdline_result.stdout is not None
                    assert "custom-process-marker" in cloned_cmdline_result.stdout
        finally:
            await docker.close()

    async def test_clone_multi_container_with_network(self, test_image: str):
        """Test cloning multi-container setup clones network too."""
        config = DockerSandboxConfig(network_enabled=True)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(
            image_spec=PullImageSpec(tag=test_image),
            hostname="test-host",
        )
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        service_container = ContainerSetup(name="service", spec=container_spec)

        task_setup = TaskSetup(
            task_id="clone-network-test",
            agent_container=agent_container,
            service_containers={"service": service_container},
            network_config=NetworkConfig(name="test-net", internal=False),
        )

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as source_state:
                # Clone the entire setup
                cloned_state = await manager.clone_sandbox_state(source_state)

                # Verify all containers cloned
                assert cloned_state.agent_container_id != source_state.agent_container_id
                assert (
                    cloned_state.service_containers["service"]
                    != source_state.service_containers["service"]
                )

                # Verify new network created
                assert cloned_state.network_id != source_state.network_id
                assert cloned_state.network_id is not None

                # Verify cloned containers can communicate on new network
                # (Use Docker API to verify network attachment)
                docker = Docker()
                try:
                    clone_agent = await docker.containers.get(cloned_state.agent_container_id)
                    clone_info = await clone_agent.show()
                    networks = clone_info["NetworkSettings"]["Networks"]
                    assert len(networks) > 0
                finally:
                    await docker.close()

    async def test_clone_cleanup_removes_temp_images(self, test_image: str):
        """Test that cloning cleanup removes temporary images."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="clone-cleanup-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        docker = Docker()
        try:
            async with manager.setup_batch([task_setup]):
                async with manager.setup_task(task_setup) as source_state:
                    # Clone multiple times
                    cloned_states = []
                    for _ in range(3):
                        cloned_state = await manager.clone_sandbox_state(source_state)
                        cloned_states.append(cloned_state)

                    # Verify temp images exist
                    images = await docker.images.list()
                    image_tags = [tag for img in images for tag in img.get("RepoTags", [])]
                    temp_images_count = sum(1 for tag in image_tags if tag and "temp-clone-" in tag)
                    assert temp_images_count >= 3

                # After task cleanup, temp images should be gone
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]

                # Filter for temp images from this specific execution
                # (may have temp images from other tests)
                for cloned_state in cloned_states:
                    # Verify this specific clone's temp image is gone
                    temp_image_pattern = f"temp-clone-{cloned_state.execution_id}"
                    assert not any(temp_image_pattern in (tag or "") for tag in image_tags)
        finally:
            await docker.close()


# ==================== Concurrent Execution Tests ====================


@pytest.mark.docker_integration
class TestConcurrentExecution:
    """Tests for concurrent task execution."""

    async def test_parallel_tasks_with_same_task_id(self, test_image: str):
        """Test that parallel tasks with same task_id are independent."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
        agent_container = ContainerSetup(name="agent", spec=container_spec)

        # Same task_id for all
        task_setup = TaskSetup(
            task_id="parallel-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        async def run_independent_task(task_num: int) -> str:
            """Run a task and return its container ID."""
            async with manager.setup_task(task_setup) as sandbox_state:
                container_id = sandbox_state.agent_container_id

                # Write unique content
                await manager.exec(
                    container_id,
                    ["sh", "-c", f"echo 'task-{task_num}' > /tmp/id.txt"],
                )

                # Verify isolation
                result = await manager.exec(container_id, ["cat", "/tmp/id.txt"])
                assert result.stdout is not None
                assert f"task-{task_num}" in result.stdout

                return container_id

        async with manager.setup_batch([task_setup]):
            # Run 5 tasks in parallel with same task_id
            container_ids = await asyncio.gather(*[run_independent_task(i) for i in range(5)])

            # All should have unique IDs
            assert len(container_ids) == 5
            assert len(set(container_ids)) == 5

    async def test_concurrent_cloning(self, test_image: str):
        """Test that concurrent cloning operations are safe."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="concurrent-clone-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        async with manager.setup_batch([task_setup]):
            async with manager.setup_task(task_setup) as source_state:
                # Clone 5 times concurrently
                cloned_states = await asyncio.gather(
                    *[manager.clone_sandbox_state(source_state) for _ in range(5)]
                )

                # All should have unique container IDs but share execution_id
                container_ids = [s.agent_container_id for s in cloned_states]
                execution_ids = [s.execution_id for s in cloned_states]

                assert len(set(container_ids)) == 5
                # All clones from same source share execution_id
                assert len(set(execution_ids)) == 1
                assert execution_ids[0] == source_state.execution_id

                # All should be functional
                for state in cloned_states:
                    result = await manager.exec(state.agent_container_id, ["echo", "test"])
                    assert result.exit_code == 0


# ==================== Image Building Tests ====================


@pytest.mark.docker_integration
class TestImageBuilding:
    """Tests for building Docker images from Dockerfiles."""

    async def test_build_image_from_dockerfile(self, test_image: str):
        """Test building an image from a Dockerfile using BuildImageSpec."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        build_spec = BuildImageSpec(
            context_path="tests/integration/fixtures",
            tag="prompt-siren-test-build:latest",
        )

        container_spec = ContainerSpec(image_spec=build_spec)
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="build-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        docker = Docker()
        try:
            # Cleanup any existing test image
            try:
                await docker.images.delete("prompt-siren-test-build:latest", force=True)
            except Exception:
                pass

            async with manager.setup_batch([task_setup]):
                # Verify image was built
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]
                assert any("prompt-siren-test-build:latest" in (tag or "") for tag in image_tags)

                # Create container and verify build marker
                async with manager.setup_task(task_setup) as sandbox_state:
                    result = await manager.exec(
                        sandbox_state.agent_container_id,
                        ["cat", "/test-marker.txt"],
                    )
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "Test build successful" in result.stdout
        finally:
            # Cleanup
            try:
                await docker.images.delete("prompt-siren-test-build:latest", force=True)
            except Exception:
                pass
            await docker.close()

    async def test_build_with_build_args(self):
        """Test building an image with build_args and custom dockerfile_path."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        build_spec = BuildImageSpec(
            context_path="tests/integration/fixtures",
            dockerfile_path="Dockerfile.dev",
            tag="prompt-siren-test-build-args:latest",
            build_args={"TEST_ARG": "custom_value"},
        )

        container_spec = ContainerSpec(image_spec=build_spec)
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="build-args-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        docker = Docker()
        try:
            # Cleanup any existing test image
            try:
                await docker.images.delete("prompt-siren-test-build-args:latest", force=True)
            except Exception:
                pass

            async with manager.setup_batch([task_setup]):
                async with manager.setup_task(task_setup) as sandbox_state:
                    # Verify build arg was used
                    result = await manager.exec(
                        sandbox_state.agent_container_id,
                        ["cat", "/build-arg-test.txt"],
                    )
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "custom_value" in result.stdout
        finally:
            # Cleanup
            try:
                await docker.images.delete("prompt-siren-test-build-args:latest", force=True)
            except Exception:
                pass
            await docker.close()

    async def test_mixed_pull_and_build_specs(self, test_image: str):
        """Test using both PullImageSpec and BuildImageSpec in the same batch."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        pull_spec = PullImageSpec(tag=test_image)
        build_spec = BuildImageSpec(
            context_path="tests/integration/fixtures",
            tag="prompt-siren-test-mixed:latest",
        )

        pull_container = ContainerSetup(name="agent", spec=ContainerSpec(image_spec=pull_spec))
        build_container = ContainerSetup(name="agent", spec=ContainerSpec(image_spec=build_spec))

        task_setups = [
            TaskSetup(
                task_id="pulled-task",
                agent_container=pull_container,
                service_containers={},
                network_config=None,
            ),
            TaskSetup(
                task_id="built-task",
                agent_container=build_container,
                service_containers={},
                network_config=None,
            ),
        ]

        docker = Docker()
        try:
            # Cleanup any existing test image
            try:
                await docker.images.delete("prompt-siren-test-mixed:latest", force=True)
            except Exception:
                pass

            async with manager.setup_batch(task_setups):
                # Create containers from both images
                async with manager.setup_task(task_setups[0]) as pulled_state:
                    result1 = await manager.exec(
                        pulled_state.agent_container_id,
                        ["echo", "pulled"],
                    )
                    assert result1.exit_code == 0

                async with manager.setup_task(task_setups[1]) as built_state:
                    result2 = await manager.exec(
                        built_state.agent_container_id,
                        ["cat", "/test-marker.txt"],
                    )
                    assert result2.exit_code == 0
                    assert result2.stdout is not None
                    assert "Test build successful" in result2.stdout
        finally:
            # Cleanup
            try:
                await docker.images.delete("prompt-siren-test-mixed:latest", force=True)
            except Exception:
                pass
            await docker.close()


# ==================== Multi-Stage Build Tests ====================


@pytest.mark.docker_integration
class TestMultiStageBuild:
    """Tests for multi-stage Docker builds with caching."""

    async def test_multi_stage_build_creates_all_stages(self):
        """Test that multi-stage build creates all three stages correctly."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        base_tag = "test-multistage-base:latest"
        env_tag = "test-multistage-env:latest"
        instance_tag = "test-multistage-instance:latest"

        stages = [
            BuildStage(
                tag=base_tag,
                context_path="tests/integration/fixtures/multistage/base",
                cache_key=base_tag,
            ),
            BuildStage(
                tag=env_tag,
                context_path="tests/integration/fixtures/multistage/env",
                parent_tag=base_tag,
                cache_key=env_tag,
            ),
            BuildStage(
                tag=instance_tag,
                context_path="tests/integration/fixtures/multistage/instance",
                parent_tag=env_tag,
            ),
        ]

        multi_stage_spec = MultiStageBuildImageSpec(stages=stages, final_tag=instance_tag)

        container_spec = ContainerSpec(image_spec=multi_stage_spec)
        agent_container = ContainerSetup(name="agent", spec=container_spec)
        task_setup = TaskSetup(
            task_id="multistage-test",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        docker = Docker()
        try:
            # Cleanup any existing images
            for tag in [base_tag, env_tag, instance_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass

            async with manager.setup_batch([task_setup]):
                # Verify all three images were created
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]

                assert any(base_tag in (tag or "") for tag in image_tags)
                assert any(env_tag in (tag or "") for tag in image_tags)
                assert any(instance_tag in (tag or "") for tag in image_tags)

                # Create container and verify all stages executed
                async with manager.setup_task(task_setup) as sandbox_state:
                    container_id = sandbox_state.agent_container_id

                    # Verify base stage
                    result = await manager.exec(container_id, ["cat", "/base-marker.txt"])
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "Base stage built" in result.stdout

                    # Verify env stage
                    result = await manager.exec(container_id, ["cat", "/env-marker.txt"])
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "Environment ready" in result.stdout

                    # Verify instance stage
                    result = await manager.exec(container_id, ["cat", "/instance-marker.txt"])
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "Instance ready" in result.stdout
        finally:
            # Cleanup
            for tag in [base_tag, env_tag, instance_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass
            await docker.close()

    async def test_multi_stage_build_caching(self):
        """Test that multi-stage build properly caches intermediate stages."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        base_tag = "test-multistage-cache-base:latest"
        env_tag = "test-multistage-cache-env:latest"
        instance_tag = "test-multistage-cache-instance:latest"

        stages = [
            BuildStage(
                tag=base_tag,
                context_path="tests/integration/fixtures/multistage/base",
                cache_key=base_tag,
            ),
            BuildStage(
                tag=env_tag,
                context_path="tests/integration/fixtures/multistage/env",
                parent_tag=base_tag,
                cache_key=env_tag,
            ),
            BuildStage(
                tag=instance_tag,
                context_path="tests/integration/fixtures/multistage/instance",
                parent_tag=env_tag,
            ),
        ]

        multi_stage_spec = MultiStageBuildImageSpec(stages=stages, final_tag=instance_tag)

        container_spec = ContainerSpec(image_spec=multi_stage_spec)
        agent_container = ContainerSetup(name="agent", spec=container_spec)

        docker = Docker()
        try:
            # Cleanup any existing images
            for tag in [base_tag, env_tag, instance_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass

            # First build - all stages should be built
            task_setup1 = TaskSetup(
                task_id="cache-test-1",
                agent_container=agent_container,
                service_containers={},
                network_config=None,
            )

            async with manager.setup_batch([task_setup1]):
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]

                assert any(base_tag in (tag or "") for tag in image_tags)
                assert any(env_tag in (tag or "") for tag in image_tags)
                assert any(instance_tag in (tag or "") for tag in image_tags)

            # Delete only instance image
            await docker.images.delete(instance_tag, force=True)

            # Second build - base and env should be cached
            task_setup2 = TaskSetup(
                task_id="cache-test-2",
                agent_container=agent_container,
                service_containers={},
                network_config=None,
            )

            async with manager.setup_batch([task_setup2]):
                # Verify all images exist again
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]

                assert any(base_tag in (tag or "") for tag in image_tags)
                assert any(env_tag in (tag or "") for tag in image_tags)
                assert any(instance_tag in (tag or "") for tag in image_tags)

                # Verify functionality
                async with manager.setup_task(task_setup2) as sandbox_state:
                    result = await manager.exec(
                        sandbox_state.agent_container_id,
                        ["cat", "/instance-marker.txt"],
                    )
                    assert result.exit_code == 0
                    assert result.stdout is not None
                    assert "Instance ready" in result.stdout
        finally:
            # Cleanup
            for tag in [base_tag, env_tag, instance_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass
            await docker.close()

    async def test_shared_base_and_env_stages(self):
        """Test multiple instances sharing base and env stages."""
        config = DockerSandboxConfig(network_enabled=False)
        manager = DockerSandboxManager(config)

        base_tag = "test-shared-base:latest"
        env_tag = "test-shared-env:latest"
        instance1_tag = "test-shared-instance1:latest"
        instance2_tag = "test-shared-instance2:latest"

        # Two specs sharing base and env
        spec1 = MultiStageBuildImageSpec(
            stages=[
                BuildStage(
                    tag=base_tag,
                    context_path="tests/integration/fixtures/multistage/base",
                    cache_key=base_tag,
                ),
                BuildStage(
                    tag=env_tag,
                    context_path="tests/integration/fixtures/multistage/env",
                    parent_tag=base_tag,
                    cache_key=env_tag,
                ),
                BuildStage(
                    tag=instance1_tag,
                    context_path="tests/integration/fixtures/multistage/instance",
                    parent_tag=env_tag,
                ),
            ],
            final_tag=instance1_tag,
        )

        spec2 = MultiStageBuildImageSpec(
            stages=[
                BuildStage(
                    tag=base_tag,
                    context_path="tests/integration/fixtures/multistage/base",
                    cache_key=base_tag,
                ),
                BuildStage(
                    tag=env_tag,
                    context_path="tests/integration/fixtures/multistage/env",
                    parent_tag=base_tag,
                    cache_key=env_tag,
                ),
                BuildStage(
                    tag=instance2_tag,
                    context_path="tests/integration/fixtures/multistage/instance",
                    parent_tag=env_tag,
                ),
            ],
            final_tag=instance2_tag,
        )

        task_setups = [
            TaskSetup(
                task_id="shared-test-1",
                agent_container=ContainerSetup(
                    name="agent",
                    spec=ContainerSpec(image_spec=spec1),
                ),
                service_containers={},
                network_config=None,
            ),
            TaskSetup(
                task_id="shared-test-2",
                agent_container=ContainerSetup(
                    name="agent",
                    spec=ContainerSpec(image_spec=spec2),
                ),
                service_containers={},
                network_config=None,
            ),
        ]

        docker = Docker()
        try:
            # Cleanup
            for tag in [base_tag, env_tag, instance1_tag, instance2_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass

            async with manager.setup_batch(task_setups):
                # Verify all images created
                images = await docker.images.list()
                image_tags = [tag for img in images for tag in img.get("RepoTags", [])]

                # Base and env should exist (built once)
                assert any(base_tag in (tag or "") for tag in image_tags)
                assert any(env_tag in (tag or "") for tag in image_tags)

                # Both instances should exist
                assert any(instance1_tag in (tag or "") for tag in image_tags)
                assert any(instance2_tag in (tag or "") for tag in image_tags)

                # Verify both work
                async with manager.setup_task(task_setups[0]) as state1:
                    result = await manager.exec(
                        state1.agent_container_id,
                        ["cat", "/base-marker.txt"],
                    )
                    assert result.exit_code == 0

                async with manager.setup_task(task_setups[1]) as state2:
                    result = await manager.exec(
                        state2.agent_container_id,
                        ["cat", "/env-marker.txt"],
                    )
                    assert result.exit_code == 0
        finally:
            # Cleanup
            for tag in [base_tag, env_tag, instance1_tag, instance2_tag]:
                try:
                    await docker.images.delete(tag, force=True)
                except Exception:  # noqa: PERF203
                    pass
            await docker.close()

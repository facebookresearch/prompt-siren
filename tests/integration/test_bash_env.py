# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for BashEnvironment multi-container support.

Run with: pytest -vx -m docker_integration tests/integration/test_bash_env.py
"""

import asyncio
from pathlib import Path

import pytest
from prompt_siren.environments.bash_env import (
    BashEnvironment,
    BashEnvState,
)
from prompt_siren.sandbox_managers.docker import (
    DockerSandboxConfig,
    DockerSandboxManager,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import ContainerSpec
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple
from pydantic import BaseModel, Field

pytestmark = pytest.mark.anyio


class _TestBenignTaskBashEnvMetadata(BaseModel):
    """Test implementation of benign task metadata."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)


class _TestMaliciousTaskBashEnvMetadata(BaseModel):
    """Test implementation of malicious task metadata."""

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec] = Field(default_factory=dict)
    benign_dockerfile_extra: str | None = None


@pytest.fixture
async def sandbox_manager() -> DockerSandboxManager:
    """Create a sandbox manager for testing."""
    config = DockerSandboxConfig(network_enabled=False)
    return DockerSandboxManager(config)


@pytest.fixture
def bash_env(
    sandbox_manager: DockerSandboxManager,
) -> BashEnvironment[
    DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
]:
    """Create a BashEnvironment instance for testing."""
    return BashEnvironment(sandbox_manager=sandbox_manager, all_injection_ids=[])


@pytest.mark.docker_integration
class TestSingleContainerTask:
    """Tests for single-container benign tasks."""

    async def test_create_env_state_for_single_container_task(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test creating env_state for a single-container benign task."""
        # Create a benign task
        image_spec = PullImageSpec(tag="alpine:latest")
        container_spec = ContainerSpec(image_spec=image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=container_spec)

        benign_task = BenignTask(
            id="test-benign-single",
            prompt="Do something",
            evaluators={},
            metadata=benign_metadata,
        )

        # Create batch and task context
        async with bash_env.create_batch_context([benign_task]):
            async with bash_env.create_task_context(benign_task) as env_state:
                # Verify env_state structure
                assert isinstance(env_state, BashEnvState)
                assert env_state.agent_container_id is not None
                assert env_state.sandbox_state.service_containers.get("attack_server") is None

                # Verify single container state (no network)
                assert env_state.sandbox_state.service_containers == {}
                assert env_state.sandbox_state.network_id is None

                # Verify container is actually running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client
                container = await docker.containers.get(env_state.agent_container_id)
                container_info = await container.show()
                assert container_info["State"]["Running"] is True

    async def test_clone_single_container_env_state(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test cloning single-container env_state."""
        image_spec = PullImageSpec(tag="alpine:latest")
        container_spec = ContainerSpec(image_spec=image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=container_spec)

        benign_task = BenignTask(
            id="test-benign-clone", prompt="Do something", evaluators={}, metadata=benign_metadata
        )

        async with bash_env.create_batch_context([benign_task]):
            async with bash_env.create_task_context(benign_task) as env_state:
                original_container_id = env_state.agent_container_id

                # Clone the env_state
                cloned_env_state = await bash_env.copy_env_state(env_state)

                # Verify cloned env_state has different container ID
                assert cloned_env_state.agent_container_id != original_container_id
                assert (
                    cloned_env_state.sandbox_state.service_containers.get("attack_server") is None
                )

                # Verify single container state (no network)
                assert cloned_env_state.sandbox_state.service_containers == {}
                assert cloned_env_state.sandbox_state.network_id is None

                # Verify both containers are running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client
                original_container = await docker.containers.get(original_container_id)
                cloned_container = await docker.containers.get(cloned_env_state.agent_container_id)

                original_info = await original_container.show()
                cloned_info = await cloned_container.show()

                assert original_info["State"]["Running"] is True
                assert cloned_info["State"]["Running"] is True


@pytest.mark.docker_integration
class TestMultiContainerTaskCouple:
    """Tests for multi-container task couples."""

    async def test_create_env_state_for_task_couple(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test creating env_state for a multi-container task couple."""
        # Create benign task
        benign_image_spec = PullImageSpec(tag="alpine:latest")
        benign_container_spec = ContainerSpec(image_spec=benign_image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=benign_container_spec)

        benign_task = BenignTask(
            id="test-benign", prompt="Do something", evaluators={}, metadata=benign_metadata
        )

        # Create malicious task
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(image_spec=attack_image_spec, hostname="attack")
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"attack_server": attack_container_spec},
        )

        malicious_task = MaliciousTask(
            id="test-malicious", goal="Exfiltrate data", evaluators={}, metadata=malicious_metadata
        )

        task_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        # Create batch and task context
        async with bash_env.create_batch_context([task_couple]):
            async with bash_env.create_task_context(task_couple) as env_state:
                # Verify env_state structure
                assert isinstance(env_state, BashEnvState)
                assert env_state.agent_container_id is not None
                attack_container_id = env_state.sandbox_state.service_containers.get(
                    "attack_server"
                )
                assert attack_container_id is not None

                # Verify multi-container state with network
                assert "attack_server" in env_state.sandbox_state.service_containers
                assert env_state.sandbox_state.network_id is not None

                # Verify both containers are actually running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client

                benign_container = await docker.containers.get(env_state.agent_container_id)
                attack_container = await docker.containers.get(attack_container_id)

                benign_info = await benign_container.show()
                attack_info = await attack_container.show()

                assert benign_info["State"]["Running"] is True
                assert attack_info["State"]["Running"] is True

                # Verify both containers are on the same network
                benign_networks = benign_info["NetworkSettings"]["Networks"]
                attack_networks = attack_info["NetworkSettings"]["Networks"]

                assert len(benign_networks) > 0
                assert len(attack_networks) > 0

                # Find common network
                benign_network_ids = {net["NetworkID"] for net in benign_networks.values()}
                attack_network_ids = {net["NetworkID"] for net in attack_networks.values()}
                assert benign_network_ids & attack_network_ids, "Containers not on same network"

    async def test_clone_multi_container_env_state(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test cloning multi-container env_state preserves network topology."""
        # Create benign task
        benign_image_spec = PullImageSpec(tag="alpine:latest")
        benign_container_spec = ContainerSpec(image_spec=benign_image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=benign_container_spec)

        benign_task = BenignTask(
            id="test-benign-clone-multi",
            prompt="Do something",
            evaluators={},
            metadata=benign_metadata,
        )

        # Create malicious task
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(image_spec=attack_image_spec, hostname="attack")
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"attack_server": attack_container_spec},
        )

        malicious_task = MaliciousTask(
            id="test-malicious-clone-multi",
            goal="Exfiltrate data",
            evaluators={},
            metadata=malicious_metadata,
        )

        task_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        # Create batch and task context
        async with bash_env.create_batch_context([task_couple]):
            async with bash_env.create_task_context(task_couple) as env_state:
                original_agent_id = env_state.agent_container_id
                original_attack_id = env_state.sandbox_state.service_containers.get("attack_server")
                assert original_attack_id is not None

                # Get original network ID
                original_network_id = env_state.sandbox_state.network_id

                # Clone the env_state
                cloned_env_state = await bash_env.copy_env_state(env_state)

                # Verify cloned env_state has different container and network IDs
                assert cloned_env_state.agent_container_id != original_agent_id
                cloned_attack_id = cloned_env_state.sandbox_state.service_containers.get(
                    "attack_server"
                )
                assert cloned_attack_id is not None
                assert cloned_attack_id != original_attack_id

                # Verify cloned state has different network
                assert cloned_env_state.sandbox_state.network_id != original_network_id
                assert cloned_env_state.sandbox_state.network_id is not None
                assert "attack_server" in cloned_env_state.sandbox_state.service_containers

                # Verify all containers are running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client

                cloned_benign = await docker.containers.get(cloned_env_state.agent_container_id)
                cloned_attack = await docker.containers.get(cloned_attack_id)

                cloned_benign_info = await cloned_benign.show()
                cloned_attack_info = await cloned_attack.show()

                assert cloned_benign_info["State"]["Running"] is True
                assert cloned_attack_info["State"]["Running"] is True

                # Verify cloned containers are on the same NEW network
                cloned_benign_networks = cloned_benign_info["NetworkSettings"]["Networks"]
                cloned_attack_networks = cloned_attack_info["NetworkSettings"]["Networks"]

                cloned_benign_net_ids = {
                    net["NetworkID"] for net in cloned_benign_networks.values()
                }
                cloned_attack_net_ids = {
                    net["NetworkID"] for net in cloned_attack_networks.values()
                }

                # Should have a common network
                assert cloned_benign_net_ids & cloned_attack_net_ids

                # And that network should be different from the original
                assert original_network_id not in cloned_benign_net_ids

    async def test_build_modified_image(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test building modified images with dockerfile_extra."""
        # Create benign task
        benign_image_spec = PullImageSpec(tag="alpine:latest")
        benign_container_spec = ContainerSpec(image_spec=benign_image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=benign_container_spec)

        benign_task = BenignTask(
            id="test-benign-modified",
            prompt="Do something",
            evaluators={},
            metadata=benign_metadata,
        )

        # Create malicious task with dockerfile_extra
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(image_spec=attack_image_spec, hostname="attack")
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"attack_server": attack_container_spec},
            benign_dockerfile_extra="RUN echo 'modified' > /tmp/test.txt",
        )

        malicious_task = MaliciousTask(
            id="test-malicious-modified",
            goal="Exfiltrate data",
            evaluators={},
            metadata=malicious_metadata,
        )

        task_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        # Create batch and task context
        async with bash_env.create_batch_context([task_couple]):
            async with bash_env.create_task_context(task_couple) as env_state:
                # Verify the agent container is using the modified image
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client

                agent_container = await docker.containers.get(env_state.agent_container_id)
                container_info = await agent_container.show()
                image_name = container_info["Config"]["Image"]

                # Should be the modified image tag (contains "modified" in the tag)
                assert "modified" in image_name

    async def test_multiple_service_containers(self, sandbox_manager: DockerSandboxManager):
        """Test creating a task couple with multiple service containers."""
        bash_env = BashEnvironment(sandbox_manager, all_injection_ids=[])

        # Create benign task
        benign_image_spec = PullImageSpec(tag="alpine:latest")
        benign_container_spec = ContainerSpec(image_spec=benign_image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=benign_container_spec)

        benign_task = BenignTask(
            id="test-benign-multi-service",
            prompt="Do something benign",
            evaluators={},
            metadata=benign_metadata,
        )

        # Create malicious task with MULTIPLE service containers
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={
                "attack_server": ContainerSpec(image_spec=attack_image_spec, hostname="attack"),
                "database": ContainerSpec(
                    image_spec=PullImageSpec(tag="alpine:latest"), hostname="db"
                ),
                "cache": ContainerSpec(
                    image_spec=PullImageSpec(tag="alpine:latest"), hostname="cache"
                ),
            },
        )

        malicious_task = MaliciousTask(
            id="test-malicious-multi-service",
            goal="Exfiltrate data using multiple services",
            evaluators={},
            metadata=malicious_metadata,
        )

        task_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        # Create batch and task context
        async with bash_env.create_batch_context([task_couple]):
            async with bash_env.create_task_context(task_couple) as env_state:
                # Verify all service containers are present
                assert "attack_server" in env_state.sandbox_state.service_containers
                assert "database" in env_state.sandbox_state.service_containers
                assert "cache" in env_state.sandbox_state.service_containers

                # Verify network is created
                assert env_state.sandbox_state.network_id is not None

                # Verify all containers are running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client

                agent_container = await docker.containers.get(env_state.agent_container_id)
                attack_container = await docker.containers.get(
                    env_state.sandbox_state.service_containers["attack_server"]
                )
                db_container = await docker.containers.get(
                    env_state.sandbox_state.service_containers["database"]
                )
                cache_container = await docker.containers.get(
                    env_state.sandbox_state.service_containers["cache"]
                )

                # Check all containers are running
                for container in [agent_container, attack_container, db_container, cache_container]:
                    info = await container.show()
                    assert info["State"]["Running"] is True

                # Verify all containers are on the same network
                agent_info = await agent_container.show()
                attack_info = await attack_container.show()
                db_info = await db_container.show()
                cache_info = await cache_container.show()

                agent_networks = set(agent_info["NetworkSettings"]["Networks"].keys())
                attack_networks = set(attack_info["NetworkSettings"]["Networks"].keys())
                db_networks = set(db_info["NetworkSettings"]["Networks"].keys())
                cache_networks = set(cache_info["NetworkSettings"]["Networks"].keys())

                # All containers should share at least one network
                common_network = agent_networks & attack_networks & db_networks & cache_networks
                assert len(common_network) > 0

    async def test_dns_resolution_between_containers(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test that agent container can resolve service container hostname via DNS."""
        # Create benign task
        benign_image_spec = PullImageSpec(tag="alpine:latest")
        benign_container_spec = ContainerSpec(image_spec=benign_image_spec)
        benign_metadata = _TestBenignTaskBashEnvMetadata(agent_container_spec=benign_container_spec)

        benign_task = BenignTask(
            id="test-benign-dns", prompt="Do something", evaluators={}, metadata=benign_metadata
        )

        # Create malicious task with service container that has a hostname
        attack_hostname = "test-attack-server.example.com"
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(
            image_spec=attack_image_spec, hostname=attack_hostname
        )
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"attack_server": attack_container_spec},
        )

        malicious_task = MaliciousTask(
            id="test-malicious-dns",
            goal="Test DNS resolution",
            evaluators={},
            metadata=malicious_metadata,
        )

        task_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        # Create batch and task context
        async with bash_env.create_batch_context([task_couple]):
            async with bash_env.create_task_context(task_couple) as env_state:
                # Verify containers are running
                assert bash_env._sandbox_manager._batch_state is not None

                # Try to resolve the attack server hostname from the agent container
                result = await bash_env._sandbox_manager.exec(
                    env_state.agent_container_id,
                    ["getent", "hosts", attack_hostname],
                    stdin=None,
                    cwd="/",
                    env=None,
                    user=None,
                    timeout=10,
                    shell_path=Path("/bin/sh"),
                )

                # DNS resolution should succeed
                assert result.exit_code == 0, f"DNS resolution failed: {result.stderr}"
                assert result.stdout is not None
                assert attack_hostname in result.stdout

                # Also test with ping (to verify network connectivity)
                ping_result = await bash_env._sandbox_manager.exec(
                    env_state.agent_container_id,
                    ["ping", "-c", "1", attack_hostname],
                    stdin=None,
                    cwd="/",
                    env=None,
                    user=None,
                    timeout=10,
                    shell_path=Path("/bin/sh"),
                )

                # Ping should succeed (exit code 0)
                assert ping_result.exit_code == 0, f"Ping failed: {ping_result.stderr}"

    async def test_dns_resolution_standalone_malicious_task(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test DNS resolution in standalone malicious task (no task couple)."""
        # Create a standalone malicious task with service container
        attack_hostname = "standalone-attack.example.com"
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(
            image_spec=attack_image_spec, hostname=attack_hostname
        )
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"attack_server": attack_container_spec},
        )

        malicious_task = MaliciousTask(
            id="test-standalone-malicious-dns",
            goal="Test DNS resolution in standalone task",
            evaluators={},
            metadata=malicious_metadata,
        )

        # Create batch and task context (no task couple, just malicious task)
        async with bash_env.create_batch_context([malicious_task]):
            async with bash_env.create_task_context(malicious_task) as env_state:
                # Verify containers are running
                assert bash_env._sandbox_manager._batch_state is not None

                # Try to resolve the attack server hostname from the agent container

                result = await bash_env._sandbox_manager.exec(
                    env_state.agent_container_id,
                    ["getent", "hosts", attack_hostname],
                    stdin=None,
                    cwd="/",
                    env=None,
                    user=None,
                    timeout=10,
                    shell_path=Path("/bin/sh"),
                )

                # DNS resolution should succeed
                assert result.exit_code == 0, f"DNS resolution failed: {result.stderr}"
                assert result.stdout is not None
                assert attack_hostname in result.stdout

                # Also test with ping to verify network connectivity
                ping_result = await bash_env._sandbox_manager.exec(
                    env_state.agent_container_id,
                    ["ping", "-c", "1", attack_hostname],
                    stdin=None,
                    cwd="/",
                    env=None,
                    user=None,
                    timeout=10,
                    shell_path=Path("/bin/sh"),
                )

                # Ping should succeed (exit code 0)
                assert ping_result.exit_code == 0, f"Ping failed: {ping_result.stderr}"

    async def test_http_request_to_service_container(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test that agent can make HTTP requests to service container."""
        # Create a standalone malicious task with HTTP server
        attack_hostname = "test-server.example.com"
        # Use python image that already has python installed
        # This avoids needing external network access to install packages
        attack_image_spec = PullImageSpec(tag="python:3.12-alpine")
        attack_container_spec = ContainerSpec(
            image_spec=attack_image_spec,
            hostname=attack_hostname,
            command=["python3", "-m", "http.server", "8080"],
        )
        # Agent container - use image with curl already installed
        agent_image_spec = PullImageSpec(tag="alpine:latest")
        agent_container_spec = ContainerSpec(image_spec=agent_image_spec)
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"http_server": attack_container_spec},
            benign_dockerfile_extra="RUN apk add --no-cache curl",
        )

        malicious_task = MaliciousTask(
            id="test-http-to-service",
            goal="Test HTTP connectivity",
            evaluators={},
            metadata=malicious_metadata,
        )

        # Create batch and task context
        async with bash_env.create_batch_context([malicious_task]):
            async with bash_env.create_task_context(malicious_task) as env_state:
                # Wait a moment for HTTP server to start
                await asyncio.sleep(0.5)

                # Verify service container is running
                assert bash_env._sandbox_manager._batch_state is not None
                docker = bash_env._sandbox_manager._batch_state.docker_client
                service_id = env_state.sandbox_state.service_containers["http_server"]
                service_container = await docker.containers.get(service_id)
                service_info = await service_container.show()

                # If container crashed, get logs to understand why
                if not service_info["State"]["Running"]:
                    logs = await service_container.log(stdout=True, stderr=True)
                    log_text = "".join(logs)
                    raise AssertionError(
                        f"Service container crashed with exit code {service_info['State']['ExitCode']}. "
                        f"Logs:\n{log_text}"
                    )

                # Try to make HTTP request from agent to service
                result = await bash_env._sandbox_manager.exec(
                    env_state.agent_container_id,
                    ["curl", "-s", f"http://{attack_hostname}:8080/"],
                    stdin=None,
                    cwd="/",
                    env=None,
                    user=None,
                    timeout=10,
                    shell_path=Path("/bin/sh"),
                )

                # HTTP request should succeed
                assert result.exit_code == 0, f"Curl failed with exit code {result.exit_code}"
                assert result.stdout is not None
                # Should get HTML directory listing
                assert "<!DOCTYPE" in result.stdout or "<html>" in result.stdout.lower()

    async def test_container_exit_raises_error(
        self,
        bash_env: BashEnvironment[
            DockerSandboxManager, _TestBenignTaskBashEnvMetadata, _TestMaliciousTaskBashEnvMetadata
        ],
    ):
        """Test that container exiting immediately raises RuntimeError with logs."""
        # Create a malicious task where the service container will exit immediately
        attack_image_spec = PullImageSpec(tag="alpine:latest")
        attack_container_spec = ContainerSpec(
            image_spec=attack_image_spec,
            hostname="failing-server",
            # This command will fail immediately (try to access internet which is blocked)
            command=["sh", "-c", "cat /abcdefg.txt"],
        )
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))
        malicious_metadata = _TestMaliciousTaskBashEnvMetadata(
            agent_container_spec=agent_container_spec,
            service_containers={"failing_server": attack_container_spec},
        )

        malicious_task = MaliciousTask(
            id="test-failing-container",
            goal="Test container failure",
            evaluators={},
            metadata=malicious_metadata,
        )

        # Create batch context
        async with bash_env.create_batch_context([malicious_task]):
            # Creating task context should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                async with bash_env.create_task_context(malicious_task):
                    pass

            # Verify error message contains useful information
            error_msg = str(exc_info.value)
            assert "exited immediately after starting" in error_msg
            assert "exit code" in error_msg
            # Should include logs showing the DNS lookup error
            assert "no such file or directory" in error_msg.lower()

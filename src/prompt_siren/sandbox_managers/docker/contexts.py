# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Batch and task-level contexts for Docker sandbox management."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    import aiodocker
    from aiodocker.containers import DockerContainer
except ImportError as e:
    raise ImportError(
        "Docker sandbox manager requires the 'docker' optional dependency. "
        "Install with: pip install 'prompt-siren[docker]'"
    ) from e

from ..sandbox_state import ContainerID, SandboxState
from ..sandbox_task_setup import ContainerSetup, TaskSetup
from .image_cache import ImageCache


@dataclass
class ContainerInfo:
    """Information about a tracked container."""

    container: DockerContainer
    temp_image: str | None = None  # Temporary image created during cloning


@dataclass
class BatchState:
    """Shared state for an entire batch of tasks.

    Attributes:
        batch_id: Unique identifier for this batch
        docker_client: Shared Docker client for all operations
        image_cache: Cache for built/pulled images
        contexts: Map of execution_id to TaskSandboxContext for resource tracking
        _lock: Lock for thread-safe modifications to contexts dict
    """

    batch_id: str
    docker_client: aiodocker.Docker
    image_cache: ImageCache
    contexts: dict[str, TaskSandboxContext] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class TaskSandboxContext:
    """Manages container lifecycle for one task execution.

    Each setup_task call creates a unique TaskSandboxContext with a unique execution_id,
    even if multiple calls use the same task_id. This ensures complete independence
    between concurrent executions.

    Responsibilities:
    - Create containers and network
    - Clone containers (commit + new container from image)
    - Track all containers (original + clones) for cleanup
    - Clean up all resources on exit
    """

    task_id: str
    execution_id: str  # Unique per setup_task call (UUID)
    batch_state: BatchState
    _containers: dict[ContainerID, ContainerInfo] = field(default_factory=dict)
    _networks: set[str] = field(default_factory=set)  # All networks for cleanup
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def create_containers(
        self,
        task_setup: TaskSetup,
        network_enabled: bool = True,
    ) -> SandboxState:
        """Create all containers for this task.

        Args:
            task_setup: Task setup specification
            network_enabled: Whether networking is enabled globally

        Returns:
            SandboxState with container IDs and network ID
        """
        # Create network if needed (multi-container setup)
        network_id = None
        if task_setup.service_containers:
            network_id = await self._create_network(task_setup, network_enabled)
            # Track network for cleanup
            async with self._lock:
                self._networks.add(network_id)

        # Create agent container
        agent_container_id = await self._create_single_container(
            container_setup=task_setup.agent_container,
            task_id=task_setup.task_id,
            network_id=network_id,
            network_enabled=network_enabled,
        )

        # Create service containers in parallel
        service_items = list(task_setup.service_containers.items())
        service_ids = await asyncio.gather(
            *[
                self._create_single_container(
                    container_setup=service_setup,
                    task_id=task_setup.task_id,
                    network_id=network_id,
                    network_enabled=network_enabled,
                )
                for service_name, service_setup in service_items
            ]
        )
        service_container_ids = dict(
            zip([name for name, _ in service_items], service_ids, strict=False)
        )

        return SandboxState(
            agent_container_id=agent_container_id,
            service_containers=service_container_ids,
            network_id=network_id,
            execution_id=self.execution_id,
        )

    async def clone(self, source_state: SandboxState) -> SandboxState:
        """Clone all containers from source_state.

        Creates new containers by committing source containers to temporary images,
        then creating new containers from those images. All cloned containers are
        tracked by this context for cleanup.

        Args:
            source_state: Source sandbox state to clone

        Returns:
            New SandboxState with cloned container IDs
        """
        # Create new network if source has one
        new_network_id = None
        if source_state.network_id:
            new_network_id = await self._clone_network(source_state.network_id)
            # Track cloned network for cleanup
            async with self._lock:
                self._networks.add(new_network_id)

        # Clone agent container
        new_agent_id = await self._clone_container(
            source_state.agent_container_id,
            network_id=new_network_id,
        )

        # Clone service containers in parallel
        service_items = list(source_state.service_containers.items())
        cloned_ids = await asyncio.gather(
            *[
                self._clone_container(
                    service_id,
                    network_id=new_network_id,
                )
                for service_name, service_id in service_items
            ]
        )
        new_service_ids = dict(zip([name for name, _ in service_items], cloned_ids, strict=False))

        return SandboxState(
            agent_container_id=new_agent_id,
            service_containers=new_service_ids,
            network_id=new_network_id,
            execution_id=self.execution_id,
        )

    async def cleanup(self) -> None:
        """Clean up all containers and networks for this task.

        Idempotent and safe - can be called multiple times.
        """
        # Cleanup all containers
        async with self._lock:
            container_ids = list(self._containers.keys())
            network_ids = list(self._networks)

        # Cleanup containers in parallel
        if container_ids:
            await asyncio.gather(
                *[self._cleanup_container(container_id) for container_id in container_ids],
                return_exceptions=True,  # Continue cleanup even if one fails
            )

        # Cleanup networks in parallel (after containers are removed)
        if network_ids:
            await asyncio.gather(
                *[self._cleanup_network(network_id) for network_id in network_ids],
                return_exceptions=True,  # Continue cleanup even if one fails
            )

    async def _create_network(self, task_setup: TaskSetup, network_enabled: bool) -> str:
        """Create a network for multi-container setup.

        Args:
            task_setup: Task setup with network configuration
            network_enabled: Whether networking is enabled globally

        Returns:
            Network ID
        """
        if not network_enabled:
            # Network disabled, but we still need to create one for container linking
            # It will be created as internal-only
            network_config = {
                "Name": f"{self.batch_state.batch_id}-{self.execution_id}-network",
                "Driver": "bridge",
                "Internal": True,
            }
        elif task_setup.network_config:
            network_config = {
                "Name": f"{self.batch_state.batch_id}-{self.execution_id}-{task_setup.network_config.name}",
                "Driver": "bridge",
                "Internal": task_setup.network_config.internal,
            }
        else:
            # Default network for multi-container
            network_config = {
                "Name": f"{self.batch_state.batch_id}-{self.execution_id}-network",
                "Driver": "bridge",
                "Internal": False,
            }

        network = await self.batch_state.docker_client.networks.create(config=network_config)
        network_info = await network.show()
        return network_info["Id"]

    async def _clone_network(self, source_network_id: str) -> str:
        """Clone a network (create a new one with same config).

        Args:
            source_network_id: Source network ID

        Returns:
            New network ID
        """
        # Get source network config
        source_network = await self.batch_state.docker_client.networks.get(source_network_id)
        source_info = await source_network.show()

        # Create new network with same config
        clone_id = uuid.uuid4().hex[:8]
        network_config = {
            "Name": f"{self.batch_state.batch_id}-{self.execution_id}-clone-{clone_id}",
            "Driver": source_info.get("Driver", "bridge"),
            "Internal": source_info.get("Internal", False),
        }

        network = await self.batch_state.docker_client.networks.create(config=network_config)
        network_info = await network.show()
        return network_info["Id"]

    async def _create_single_container(
        self,
        container_setup: ContainerSetup,
        task_id: str,
        network_id: str | None,
        network_enabled: bool,
    ) -> ContainerID:
        """Create and start a single container.

        Args:
            container_setup: Container setup specification
            task_id: Task identifier
            network_id: Optional network ID to attach to
            network_enabled: Whether networking is enabled

        Returns:
            Container ID
        """
        # Get image tag from cache
        image_tag = await self.batch_state.image_cache.get_image_for_container(
            container_setup, task_id
        )

        # Build container config
        config = self._build_container_config(
            container_setup=container_setup,
            image_tag=image_tag,
            network_id=network_id,
            network_enabled=network_enabled,
        )

        # Generate unique container name
        safe_task_id = task_id.replace(":", "-").replace("/", "-")
        unique_suffix = uuid.uuid4().hex[:8]
        container_name = (
            f"{self.batch_state.batch_id}-{self.execution_id}-"
            f"{container_setup.name}-{safe_task_id}-{unique_suffix}"
        )

        # Create and start container
        container = await self.batch_state.docker_client.containers.create(
            config, name=container_name
        )
        await container.start()
        container_id: ContainerID = (await container.show())["Id"]

        # Check if container exited immediately after starting
        container_info = await container.show()
        if not container_info["State"]["Running"]:
            exit_code = container_info["State"].get("ExitCode", "unknown")
            # Get container logs to help diagnose the issue
            logs = await container.log(stdout=True, stderr=True)
            log_text = "".join(logs) if logs else "(no logs)"
            # Clean up the failed container
            try:
                await container.delete()
            except Exception:
                pass  # Best effort cleanup
            raise RuntimeError(
                f"Container {container_name} exited immediately after starting "
                f"with exit code {exit_code}. Logs:\n{log_text}"
            )

        # Track container
        async with self._lock:
            self._containers[container_id] = ContainerInfo(container=container, temp_image=None)

        return container_id

    async def _clone_container(
        self,
        source_id: ContainerID,
        network_id: str | None = None,
    ) -> ContainerID:
        """Clone a container by committing to image and creating new container.

        Args:
            source_id: Source container ID
            network_id: Optional network ID to attach clone to

        Returns:
            Cloned container ID
        """
        source_container = await self.batch_state.docker_client.containers.get(source_id)

        # Generate unique names
        clone_id = uuid.uuid4().hex[:8]
        clone_name = f"{self.batch_state.batch_id}-{self.execution_id}-clone-{clone_id}"
        temp_image_name = f"temp-clone-{self.execution_id}-{clone_id}"

        # Commit container to temporary image
        await source_container.commit(repository=temp_image_name, tag="latest")
        temp_image_full = f"{temp_image_name}:latest"

        cloned_container = None
        try:
            # Get source container config
            source_info = await source_container.show()

            # Build config for cloned container
            config: dict[str, Any] = {"Image": temp_image_full}

            # Copy settings from source
            if source_info["Config"].get("Env"):
                config["Env"] = source_info["Config"]["Env"]

            if source_info["Config"].get("Cmd"):
                config["Cmd"] = source_info["Config"]["Cmd"]

            if source_info["Config"].get("Hostname"):
                config["Hostname"] = source_info["Config"]["Hostname"]

            # Only copy NetworkMode if NOT using custom network
            # (custom networks use NetworkingConfig instead)
            if not network_id and source_info["HostConfig"].get("NetworkMode"):
                config["HostConfig"] = {"NetworkMode": source_info["HostConfig"]["NetworkMode"]}

            # Add network configuration if provided
            if network_id:
                hostname = config.get("Hostname")
                if hostname:
                    config["NetworkingConfig"] = {
                        "EndpointsConfig": {
                            network_id: {
                                "Aliases": [hostname],
                            }
                        }
                    }

            # Create and start cloned container
            cloned_container = await self.batch_state.docker_client.containers.create(
                config=config, name=clone_name
            )
            await cloned_container.start()

            container_id: ContainerID = (await cloned_container.show())["Id"]

            # Track cloned container
            async with self._lock:
                self._containers[container_id] = ContainerInfo(
                    container=cloned_container, temp_image=temp_image_full
                )

            return container_id

        except Exception:
            # Cleanup on failure
            if cloned_container:
                try:
                    await cloned_container.stop()
                    await cloned_container.delete()
                except Exception:
                    pass  # Best effort

            try:
                await self.batch_state.docker_client.images.delete(temp_image_full, force=True)
            except Exception:
                pass  # Best effort

            raise

    def _build_container_config(
        self,
        container_setup: ContainerSetup,
        image_tag: str,
        network_id: str | None,
        network_enabled: bool,
    ) -> dict[str, Any]:
        """Build Docker container configuration.

        Args:
            container_setup: Container setup specification
            image_tag: Docker image tag to use
            network_id: Optional network ID to attach to
            network_enabled: Whether networking is enabled

        Returns:
            Docker container configuration dict
        """
        config: dict[str, Any] = {"Image": image_tag}

        # Set command (default to sleep infinity to keep container running)
        if container_setup.spec.command:
            config["Cmd"] = container_setup.spec.command
        else:
            config["Cmd"] = ["sleep", "infinity"]

        # Set environment variables
        if container_setup.spec.environment:
            config["Env"] = [
                f"{key}={value}" for key, value in container_setup.spec.environment.items()
            ]

        # Set hostname
        if container_setup.spec.hostname:
            config["Hostname"] = container_setup.spec.hostname

        # Apply network configuration
        # Network isolation has two modes:
        # 1. Full isolation: network_enabled=False AND no network_id -> NetworkMode: "none"
        # 2. Internal-only: network_enabled=False AND network_id present -> attached to internal network
        #    This allows multi-container tasks to communicate internally while blocking external access
        if network_id:
            # Attach to task network (with hostname for DNS if specified)
            # When network_id is present, we ALWAYS attach to it, even if network_enabled=False.
            # This is intentional: multi-container tasks need internal communication even when
            # external networking is disabled. The network itself is created as internal-only
            # in _create_network() when network_enabled=False.
            endpoint_config: dict[str, Any] = {}
            if container_setup.spec.hostname:
                endpoint_config["Aliases"] = [container_setup.spec.hostname]

            config["NetworkingConfig"] = {
                "EndpointsConfig": {
                    network_id: endpoint_config,
                }
            }
        elif not network_enabled:
            # No network_id and networking disabled: isolate container completely
            config["HostConfig"] = {"NetworkMode": "none"}

        return config

    async def _cleanup_container(self, container_id: ContainerID) -> None:
        """Clean up a single container.

        Args:
            container_id: Container ID to clean up
        """
        # Remove from tracking
        async with self._lock:
            info = self._containers.pop(container_id, None)

        if info is None:
            return

        # Stop and remove container
        try:
            await info.container.stop()
            await info.container.delete()
        except Exception:
            pass  # Best effort cleanup

        # Remove temporary image if exists
        if info.temp_image:
            try:
                await self.batch_state.docker_client.images.delete(info.temp_image, force=True)
            except Exception:
                pass  # Best effort cleanup

    async def _cleanup_network(self, network_id: str) -> None:
        """Clean up a network.

        Args:
            network_id: Network ID to clean up
        """
        try:
            network = await self.batch_state.docker_client.networks.get(network_id)
            await network.delete()
        except Exception:
            pass  # Best effort cleanup

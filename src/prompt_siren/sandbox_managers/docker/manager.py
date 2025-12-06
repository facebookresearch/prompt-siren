# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Docker-based sandbox manager implementation."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel, Field

try:
    import aiodocker
except ImportError as e:
    raise ImportError(
        "Docker sandbox manager requires the 'docker' optional dependency. "
        "Install with: pip install 'prompt-siren[docker]'"
    ) from e

from ..abstract import ExecOutput
from ..sandbox_state import ContainerID, SandboxState
from ..sandbox_task_setup import ContainerSetup, SandboxTaskSetup
from .contexts import BatchState, TaskSandboxContext
from .exec_utils import exec_in_container
from .image_cache import ImageCache


class DockerSandboxConfig(BaseModel):
    """Configuration for Docker sandbox manager."""

    network_enabled: bool = Field(
        default=False,
        description="Whether to enable network access in containers",
    )
    batch_id_prefix: str = "workbench"


class DockerSandboxManager:
    """Docker-based implementation of AbstractSandboxManager.

    Provides isolated execution environments using Docker containers with support for:
    - Single and multi-container tasks
    - Image caching and modification (dockerfile_extra)
    - Container cloning for state snapshots
    - Concurrent task execution within a batch

    Architecture:
    - BatchState: Shared state for entire batch (Docker client, image cache, contexts)
    - TaskSandboxContext: Per-task resource manager (containers, network, cleanup)
    - ImageCache: Sequential image building and caching
    """

    def __init__(self, config: DockerSandboxConfig):
        """Initialize Docker sandbox manager.

        Args:
            batch_id_prefix: Prefix for batch ID generation
            network_enabled: Whether networking is enabled for containers
        """
        self._batch_id_prefix = config.batch_id_prefix
        self._network_enabled = config.network_enabled
        self._batch_state: BatchState | None = None

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        """Prepare all images and resources for the batch.

        Creates Docker client, builds/pulls all images sequentially,
        and tracks all task contexts for cleanup.

        Args:
            task_setups: All task setups for this batch

        Yields:
            Control for task execution
        """
        # Generate unique batch ID
        batch_id = f"{self._batch_id_prefix}-{uuid.uuid4().hex[:8]}"

        # Create Docker client
        docker_client = aiodocker.Docker()

        try:
            # Create image cache
            image_cache = ImageCache(docker_client, batch_id)

            # Create batch state
            self._batch_state = BatchState(
                batch_id=batch_id,
                docker_client=docker_client,
                image_cache=image_cache,
                contexts={},
            )

            # Collect all container setups for image preparation
            all_container_setups: list[ContainerSetup] = []
            for task_setup in task_setups:
                all_container_setups.append(task_setup.agent_container)
                all_container_setups.extend(task_setup.service_containers.values())

            # Build/pull all base images sequentially
            await image_cache.ensure_all_base_images(all_container_setups)

            # Yield control for task execution
            yield

        finally:
            # Cleanup all task contexts
            if self._batch_state:
                async with self._batch_state._lock:
                    contexts = list(self._batch_state.contexts.values())

                for context in contexts:
                    await context.cleanup()

            # Close Docker client
            await docker_client.close()
            self._batch_state = None

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        """Create containers and network for a task.

        Creates a TaskSandboxContext, sets up all containers and network,
        and yields the SandboxState. Cleans up resources on exit.

        Args:
            task_setup: Task setup specification

        Yields:
            SandboxState with container IDs and network ID
        """
        if self._batch_state is None:
            raise RuntimeError("setup_task called outside of setup_batch context")

        # Generate unique execution ID
        execution_id = uuid.uuid4().hex

        # Create task context
        context = TaskSandboxContext(
            task_id=task_setup.task_id,
            execution_id=execution_id,
            batch_state=self._batch_state,
        )

        # Register context in batch state
        async with self._batch_state._lock:
            self._batch_state.contexts[execution_id] = context

        try:
            # Create containers and network
            sandbox_state = await context.create_containers(
                task_setup, network_enabled=self._network_enabled
            )

            # Yield sandbox state
            yield sandbox_state

        finally:
            # Cleanup all resources for this task
            await context.cleanup()

            # Unregister context
            async with self._batch_state._lock:
                self._batch_state.contexts.pop(execution_id, None)

    async def clone_sandbox_state(self, source_state: SandboxState) -> SandboxState:
        """Clone all containers and network from source state.

        Delegates to the TaskSandboxContext that owns the source containers.

        Args:
            source_state: Source sandbox state to clone

        Returns:
            New SandboxState with cloned container IDs
        """
        if self._batch_state is None:
            raise RuntimeError("clone_sandbox_state called outside of setup_batch context")

        # Look up the context that owns the source containers
        async with self._batch_state._lock:
            context = self._batch_state.contexts.get(source_state.execution_id)

        if context is None:
            raise ValueError(
                f"Cannot clone sandbox state: execution_id {source_state.execution_id} not found"
            )

        # Delegate cloning to the context
        return await context.clone(source_state)

    async def exec(
        self,
        container_id: ContainerID,
        cmd: str | list[str],
        stdin: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        shell_path: Path | None = None,
    ) -> ExecOutput:
        """Execute a command in a container.

        Args:
            container_id: Container ID to execute in
            cmd: Command to execute
            stdin: Optional stdin data
            cwd: Optional working directory
            env: Optional environment variables
            user: Optional user to run as
            timeout: Optional timeout in seconds
            shell_path: Optional path to shell executable (defaults to /bin/bash)

        Returns:
            ExecOutput with stdout/stderr chunks and exit code

        Raises:
            ExecTimeoutError: If execution times out
        """
        if self._batch_state is None:
            raise RuntimeError("exec called outside of setup_batch context")

        return await exec_in_container(
            docker=self._batch_state.docker_client,
            container_id=container_id,
            cmd=cmd,
            stdin=stdin,
            cwd=cwd,
            env=env,
            user=user,
            timeout=timeout,
            shell_path=shell_path,
        )


def create_docker_sandbox_manager(
    config: DockerSandboxConfig, context: None = None
) -> DockerSandboxManager:
    """Factory function to create a Docker sandbox manager.

    The Docker client is created lazily in setup_batch() and automatically
    closed when the batch context exits.

    Args:
        config: Configuration for the Docker sandbox
        context: Optional context parameter (unused by sandbox managers, for registry compatibility)

    Returns:
        Configured DockerSandboxManager instance
    """
    return DockerSandboxManager(config)

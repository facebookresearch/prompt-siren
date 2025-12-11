# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Docker-based sandbox manager implementation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..abstract import ExecOutput
from ..sandbox_state import ContainerID, SandboxState
from ..sandbox_task_setup import ContainerSetup, SandboxTaskSetup
from .contexts import BatchState, TaskSandboxContext
from .exec_utils import exec_in_container
from .image_cache import ImageCache
from .plugins import (
    AbstractDockerClient,
    create_docker_client,
    get_docker_client_config_class,
)

logger = logging.getLogger(__name__)


def create_docker_client_from_config(client_type: str, config: dict) -> AbstractDockerClient:
    """Create a Docker client instance from configuration.

    Follows the same pattern as registry_bridge.py:
    1. Look up the config class from the registry
    2. Validate the config dict against the Pydantic model
    3. Create the component using the factory

    Args:
        client_type: The Docker client type (e.g., "local")
        config: Configuration dictionary for the client

    Returns:
        Configured Docker client instance

    Raises:
        RuntimeError: If client type is not registered
        ValidationError: If configuration is invalid
    """
    config_class = get_docker_client_config_class(client_type)
    validated_config = config_class.model_validate(config)
    return create_docker_client(client_type, validated_config)


class DockerSandboxConfig(BaseModel):
    """Configuration for Docker sandbox manager."""

    network_enabled: bool = Field(
        default=False,
        description="Whether to enable network access in containers",
    )
    batch_id_prefix: str = "workbench"
    docker_client: str = Field(
        default="local",
        description="Name of the Docker client plugin to use",
    )
    docker_client_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the Docker client plugin",
    )


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
            config: Docker sandbox configuration
        """
        self._config = config
        self._batch_state: BatchState | None = None

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        """Prepare all images and resources for the batch.

        Creates Docker client based on config, builds/pulls all images
        sequentially, and tracks all task contexts for cleanup.

        Args:
            task_setups: All task setups for this batch

        Yields:
            Control for task execution
        """
        logger.debug(
            f"[DockerSandboxManager] setup_batch: Starting with {len(task_setups)} task setups"
        )
        # Generate unique batch ID
        batch_id = f"{self._config.batch_id_prefix}-{uuid.uuid4().hex[:8]}"
        logger.debug(f"[DockerSandboxManager] setup_batch: Generated batch_id: {batch_id}")

        # Create appropriate Docker client based on configuration
        logger.debug(
            f"[DockerSandboxManager] setup_batch: Creating Docker client '{self._config.docker_client}'"
        )

        docker_client = create_docker_client_from_config(
            self._config.docker_client,
            self._config.docker_client_config,
        )
        try:
            # Create image cache
            logger.debug("[DockerSandboxManager] setup_batch: Creating image cache")
            image_cache = ImageCache(docker_client, batch_id)

            # Create batch state
            logger.debug("[DockerSandboxManager] setup_batch: Creating batch state")
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

            logger.debug(
                f"[DockerSandboxManager] setup_batch: Collected {len(all_container_setups)} container setups"
            )
            for idx, setup in enumerate(all_container_setups):
                logger.debug(
                    f"[DockerSandboxManager] setup_batch: Container setup {idx}: name='{setup.name}', image_spec={type(setup.spec.image_spec).__name__}"
                )

            # Build/pull all base images sequentially
            logger.debug("[DockerSandboxManager] setup_batch: Calling ensure_all_base_images")
            await image_cache.ensure_all_base_images(all_container_setups)
            logger.debug("[DockerSandboxManager] setup_batch: ensure_all_base_images completed")

            # Yield control for task execution
            logger.debug("[DockerSandboxManager] setup_batch: Yielding control for task execution")
            yield
            logger.debug(
                "[DockerSandboxManager] setup_batch: Task execution completed, starting cleanup"
            )

        finally:
            # Cleanup all task contexts
            if self._batch_state:
                async with self._batch_state._lock:
                    contexts = list(self._batch_state.contexts.values())

                logger.debug(
                    f"[DockerSandboxManager] setup_batch: Cleaning up {len(contexts)} task contexts"
                )
                for context in contexts:
                    await context.cleanup()

            # Close Docker client
            logger.debug("[DockerSandboxManager] setup_batch: Closing Docker client")
            await docker_client.close()
            self._batch_state = None
            logger.debug("[DockerSandboxManager] setup_batch: Cleanup completed")

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
                task_setup, network_enabled=self._config.network_enabled
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

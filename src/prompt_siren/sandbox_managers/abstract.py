# Copyright (c) Meta Platforms, Inc. and affiliates.
from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, TypeAlias

from .sandbox_state import ContainerID, SandboxState
from .sandbox_task_setup import SandboxTaskSetup


@dataclass(frozen=True)
class StdoutChunk:
    """Represents a chunk of stdout output."""

    content: str


@dataclass(frozen=True)
class StderrChunk:
    """Represents a chunk of stderr output."""

    content: str


Output: TypeAlias = StdoutChunk | StderrChunk


@dataclass(frozen=True)
class ExecOutput:
    """Output from executing a command in a container.

    The outputs list contains chunks in chronological order as they were received,
    preserving both timing and stream information. Use the cached properties for
    convenient access to common formats.
    """

    outputs: list[Output]
    exit_code: int

    @cached_property
    def stdout(self) -> str | None:
        """Get all stdout chunks joined together, or None if no stdout."""
        chunks = [chunk.content for chunk in self.outputs if isinstance(chunk, StdoutChunk)]
        return "".join(chunks) if chunks else None

    @cached_property
    def stderr(self) -> str | None:
        """Get all stderr chunks joined together, or None if no stderr."""
        chunks = [chunk.content for chunk in self.outputs if isinstance(chunk, StderrChunk)]
        return "".join(chunks) if chunks else None

    @cached_property
    def combined(self) -> str | None:
        """Get all output chunks in chronological order, or None if no output."""
        chunks = [chunk.content for chunk in self.outputs]
        return "".join(chunks) if chunks else None


class ExecTimeoutError(TimeoutError):
    """Raised when command execution times out in a container."""

    def __init__(self, container_id: str, command: str | list[str], timeout: int) -> None:
        self.container_id = container_id
        self.command = command
        self.timeout = timeout
        super().__init__(
            f"Command timed out after {timeout}s in container {container_id[:12]}: {command}"
        )


class AbstractSandboxManager(Protocol):
    """Protocol for sandbox managers that provide isolated execution environments.

    Concurrency Safety:
        Implementations MUST be async-safe for concurrent task execution within a batch.
        Multiple tasks can run concurrently within a single batch, so shared state must be
        protected (e.g., using asyncio.Lock for dict modifications).

        Only one batch runs at a time, so batch-level operations (setup_batch) do not
        need concurrency protection.

    Typical usage pattern:
        task_setups = [
            TaskSetup(task_id="task1", containers=[...]),
            TaskSetup(task_id="task2", containers=[...])
        ]
        async with manager.setup_batch(task_setups):  # One batch at a time
            async with manager.setup_task(task_setups[0]) as state1:  # Concurrent
                async with manager.setup_task(task_setups[1]) as state2:  # Concurrent
                    # Execute commands in containers concurrently
                    await manager.exec(state1.benign_container_id, "command")
                    await manager.exec(state2.benign_container_id, "command")
    """

    @asynccontextmanager
    @abstractmethod
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        """Prepare all images (with modifications) for the batch.

        Responsibilities:
        - Build/pull all required images
        - Apply dockerfile modifications and cache them
        - Initialize shared resources (e.g., Docker client, network tracking)
        """
        raise NotImplementedError()
        yield  # needed to make type checker happy about the @asynccontextmanager decorator

    @asynccontextmanager
    @abstractmethod
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        """Create sandbox (containers and network) for a task.

        Uses configuration from task_setup to create containers and network.
        Yields SandboxState with container IDs and network ID.
        Cleans up containers and network on exit.
        """
        raise NotImplementedError()
        yield  # needed to make type checker happy about the @asynccontextmanager decorator

    @abstractmethod
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
            cmd: Command to execute (string or list of arguments)
            stdin: Optional stdin data to pass to the command
            cwd: Optional working directory for command execution
            env: Optional environment variables for command execution
            user: Optional user to run command as
            timeout: Optional timeout in seconds
            shell_path: Optional path to shell executable (defaults to /bin/bash)

        Returns:
            ExecOutput containing stdout/stderr chunks and exit code

        Raises:
            ExecTimeoutError: If command execution exceeds timeout
        """
        raise NotImplementedError()

    @abstractmethod
    async def clone_sandbox_state(self, source_state: SandboxState) -> SandboxState:
        """Clone an entire sandbox (all containers + network if multi-container).

        Creates fresh copies of all containers in the sandbox state. For multi-container
        setups, creates a new network and attaches all cloned containers to it.

        Args:
            source_state: The sandbox state to clone

        Returns:
            New SandboxState with fresh container IDs (and network ID for multi-container)
        """
        raise NotImplementedError()

"""Abstract Docker client interface for supporting multiple execution backends.

This module provides abstractions for Docker operations that can be implemented
by different backends (local Docker via aiodocker, or remote Docker via DES).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..abstract import ExecOutput


class AbstractContainer(ABC):
    """Abstract interface for Docker containers."""

    @abstractmethod
    async def start(self) -> None:
        """Start the container."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the container."""
        ...

    @abstractmethod
    async def delete(self) -> None:
        """Delete the container."""
        ...

    @abstractmethod
    async def show(self) -> dict[str, Any]:
        """Get container details.

        Returns:
            Dict with container information including Id, State, Config, HostConfig
        """
        ...

    @abstractmethod
    async def exec(
        self,
        cmd: list[str],
        stdin: str | bytes | None,
        user: str,
        environment: dict[str, str] | None,
        workdir: str | None,
        timeout: int,
    ) -> ExecOutput:
        """Execute a command in the container.

        Args:
            cmd: Command to execute (already in bash -c format)
            stdin: Optional stdin data to pass to the command
            user: User to run as
            environment: Environment variables
            workdir: Working directory
            timeout: Timeout in seconds

        Returns:
            ExecOutput containing stdout/stderr chunks and exit code
        """
        ...

    @abstractmethod
    async def log(self, stdout: bool, stderr: bool) -> list[str]:
        """Get container logs.

        Args:
            stdout: Include stdout
            stderr: Include stderr

        Returns:
            List of log lines
        """
        ...

    @abstractmethod
    async def commit(self, repository: str, tag: str) -> None:
        """Commit container to an image.

        Args:
            repository: Repository name
            tag: Image tag
        """
        ...


class AbstractNetwork(ABC):
    """Abstract interface for Docker networks."""

    @abstractmethod
    async def show(self) -> dict[str, Any]:
        """Get network details.

        Returns:
            Dict with network information including Id, Driver, Internal
        """
        ...

    @abstractmethod
    async def delete(self) -> None:
        """Delete the network."""
        ...


class AbstractDockerClient(ABC):
    """Abstract Docker client interface.

    This interface provides all Docker operations needed by the workbench,
    abstracting away the underlying implementation (local aiodocker or remote DES).
    """

    @abstractmethod
    async def close(self) -> None:
        """Close the client and clean up resources."""
        ...

    # Image operations

    @abstractmethod
    async def inspect_image(self, tag: str) -> dict[str, Any]:
        """Inspect an image.

        Args:
            tag: Image tag

        Returns:
            Image details

        Raises:
            DockerClientError: If image doesn't exist
        """
        ...

    @abstractmethod
    async def pull_image(self, tag: str) -> None:
        """Pull an image from registry.

        Args:
            tag: Image tag to pull
        """
        ...

    @abstractmethod
    async def build_image(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        buildargs: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Build an image from a build context directory.

        The implementation handles creating tar archives and transferring files
        as needed for the specific backend (local or remote).

        Args:
            context_path: Path to the build context directory
            tag: Tag for built image
            dockerfile_path: Path to Dockerfile relative to context (defaults to "Dockerfile")
            buildargs: Build arguments

        Yields:
            Build log entries (dicts with "stream", "error", etc.)
        """
        # This is an abstract async generator - implementations must yield log entries
        yield {}  # type: ignore[unreachable]
        ...

    @abstractmethod
    async def delete_image(self, tag: str, force: bool = False) -> None:
        """Delete an image.

        Args:
            tag: Image tag to delete
            force: Force deletion
        """
        ...

    # Container operations

    @abstractmethod
    async def create_container(self, config: dict[str, Any], name: str) -> AbstractContainer:
        """Create a container.

        Args:
            config: Container configuration
            name: Container name

        Returns:
            Container instance
        """
        ...

    @abstractmethod
    async def get_container(self, container_id: str) -> AbstractContainer:
        """Get a container by ID.

        Args:
            container_id: Container ID

        Returns:
            Container instance
        """
        ...

    # Network operations

    @abstractmethod
    async def create_network(self, config: dict[str, Any]) -> AbstractNetwork:
        """Create a network.

        Args:
            config: Network configuration

        Returns:
            Network instance
        """
        ...

    @abstractmethod
    async def get_network(self, network_id: str) -> AbstractNetwork:
        """Get a network by ID.

        Args:
            network_id: Network ID

        Returns:
            Network instance
        """
        ...

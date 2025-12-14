"""Local Docker client implementation using aiodocker."""

from __future__ import annotations

import io
import tarfile
from collections.abc import AsyncIterator
from typing import Any

try:
    import aiodocker
    from aiodocker.containers import DockerContainer
    from aiodocker.exceptions import DockerError
    from aiodocker.networks import DockerNetwork
except ImportError as e:
    raise ImportError(
        "The local Docker client requires the 'docker' optional dependency. "
        "Install with: pip install 'prompt-siren[docker]'"
    ) from e

from aiohttp import ClientTimeout
from pydantic import BaseModel

from .. import ExecOutput, StderrChunk, StdoutChunk
from .plugins import AbstractContainer, AbstractDockerClient, AbstractNetwork
from .plugins.errors import DockerClientError


class LocalDockerClientConfig(BaseModel):
    """Configuration for local Docker client (no special config needed)."""


class LocalContainer(AbstractContainer):
    """Local implementation of container using aiodocker."""

    def __init__(self, container: DockerContainer):
        self._container = container

    async def start(self) -> None:
        """Start the container."""
        try:
            await self._container.start()
        except DockerError as e:
            raise DockerClientError(f"Failed to start container: {e.message}") from e

    async def stop(self) -> None:
        """Stop the container."""
        try:
            await self._container.stop()
        except DockerError as e:
            raise DockerClientError(f"Failed to stop container: {e.message}") from e

    async def delete(self) -> None:
        """Delete the container."""
        try:
            await self._container.delete()
        except DockerError as e:
            raise DockerClientError(f"Failed to delete container: {e.message}") from e

    async def show(self) -> dict[str, Any]:
        """Get container details."""
        try:
            return await self._container.show()
        except DockerError as e:
            raise DockerClientError(f"Failed to get container details: {e.message}") from e

    async def exec(
        self,
        cmd: list[str],
        stdin: str | bytes | None,
        user: str,
        environment: dict[str, str] | None,
        workdir: str | None,
        timeout: int,
    ) -> ExecOutput:
        """Execute a command in the container with streaming I/O.

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
        try:
            # Create exec instance
            exec_instance = await self._container.exec(
                cmd=cmd,
                stdout=True,
                stderr=True,
                stdin=stdin is not None,
                user=user,
                environment=environment,
                workdir=workdir,
            )

            # Start execution
            stream = exec_instance.start(detach=False, timeout=ClientTimeout(total=timeout))

            # Write stdin if provided
            if stdin is not None:
                if isinstance(stdin, str):
                    stdin_bytes = stdin.encode()
                else:
                    stdin_bytes = stdin
                await stream.write_in(stdin_bytes)
                # Signal EOF on stdin without closing stdout/stderr
                # Access transport directly like aiodocker's write_in() does
                if stream._resp is not None and stream._resp.connection is not None:
                    transport = stream._resp.connection.transport
                    if transport and transport.can_write_eof():
                        transport.write_eof()

            # Read output
            outputs = []
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break
                decoded = msg.data.decode("utf-8", errors="replace")
                if msg.stream == 1:  # stdout
                    outputs.append(StdoutChunk(content=decoded))
                elif msg.stream == 2:  # stderr
                    outputs.append(StderrChunk(content=decoded))

            # Get exit code
            exit_code = (await exec_instance.inspect())["ExitCode"]

            return ExecOutput(outputs=outputs, exit_code=exit_code)
        except DockerError as e:
            raise DockerClientError(f"Failed to execute command in container: {e.message}") from e

    async def log(self, stdout: bool, stderr: bool) -> list[str]:
        """Get container logs."""
        try:
            return await self._container.log(stdout=stdout, stderr=stderr)
        except DockerError as e:
            raise DockerClientError(f"Failed to get container logs: {e.message}") from e

    async def commit(self, repository: str, tag: str) -> None:
        """Commit container to an image."""
        try:
            await self._container.commit(repository=repository, tag=tag)
        except DockerError as e:
            raise DockerClientError(
                f"Failed to commit container to {repository}:{tag}: {e.message}"
            ) from e


class LocalNetwork(AbstractNetwork):
    """Local implementation of network using aiodocker."""

    def __init__(self, network: DockerNetwork):
        self._network = network

    async def show(self) -> dict[str, Any]:
        """Get network details."""
        try:
            return await self._network.show()
        except DockerError as e:
            raise DockerClientError(f"Failed to get network details: {e.message}") from e

    async def delete(self) -> None:
        """Delete the network."""
        try:
            await self._network.delete()
        except DockerError as e:
            raise DockerClientError(f"Failed to delete network: {e.message}") from e


class LocalDockerClient(AbstractDockerClient):
    """Local Docker client implementation using aiodocker.

    This implementation wraps aiodocker to provide Docker operations on the local machine.
    """

    def __init__(self):
        """Initialize local Docker client."""
        self._docker = aiodocker.Docker()

    async def close(self) -> None:
        """Close the Docker client."""
        await self._docker.close()

    # Image operations

    async def inspect_image(self, tag: str) -> dict[str, Any]:
        """Inspect an image."""
        try:
            return await self._docker.images.inspect(tag)
        except DockerError as e:
            raise DockerClientError(f"Failed to inspect image {tag}: {e.message}") from e

    async def pull_image(self, tag: str) -> None:
        """Pull an image from registry."""
        try:
            await self._docker.images.pull(from_image=tag)
        except DockerError as e:
            raise DockerClientError(f"Failed to pull image {tag}: {e.message}") from e

    async def build_image(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        buildargs: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Build an image from a build context directory.

        Creates a tar archive of the context and sends it to Docker.
        """
        # Create tar archive of build context with normalized ownership
        tar_stream = io.BytesIO()

        def reset_ownership(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
            """Reset ownership to root to avoid user namespace issues."""
            tarinfo.uid = 0
            tarinfo.gid = 0
            tarinfo.uname = "root"
            tarinfo.gname = "root"
            return tarinfo

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(context_path, arcname=".", filter=reset_ownership)
        tar_stream.seek(0)

        # Build image from tar archive
        try:
            async for log_line in self._docker.images.build(
                fileobj=tar_stream,
                tag=tag,
                encoding="application/x-tar",
                stream=True,
                path_dockerfile=dockerfile_path or "Dockerfile",
                buildargs=buildargs,
            ):
                yield log_line
        except DockerError as e:
            raise DockerClientError(f"Failed to build image {tag}: {e.message}") from e

    async def delete_image(self, tag: str, force: bool = False) -> None:
        """Delete an image."""
        try:
            await self._docker.images.delete(tag, force=force)
        except DockerError as e:
            raise DockerClientError(f"Failed to delete image {tag}: {e.message}") from e

    # Container operations

    async def create_container(self, config: dict[str, Any], name: str) -> AbstractContainer:
        """Create a container."""
        try:
            container = await self._docker.containers.create(config, name=name)
            return LocalContainer(container)
        except DockerError as e:
            raise DockerClientError(f"Failed to create container {name}: {e.message}") from e

    async def get_container(self, container_id: str) -> AbstractContainer:
        """Get a container by ID."""
        try:
            container = await self._docker.containers.get(container_id)
            return LocalContainer(container)
        except DockerError as e:
            raise DockerClientError(f"Failed to get container {container_id}: {e.message}") from e

    # Network operations

    async def create_network(self, config: dict[str, Any]) -> AbstractNetwork:
        """Create a network."""
        try:
            network = await self._docker.networks.create(config=config)
            return LocalNetwork(network)
        except DockerError as e:
            raise DockerClientError(f"Failed to create network: {e.message}") from e

    async def get_network(self, network_id: str) -> AbstractNetwork:
        """Get a network by ID."""
        try:
            network = await self._docker.networks.get(network_id)
            return LocalNetwork(network)
        except DockerError as e:
            raise DockerClientError(f"Failed to get network {network_id}: {e.message}") from e


def create_local_docker_client(
    config: LocalDockerClientConfig, context: None = None
) -> LocalDockerClient:
    """Factory function to create a LocalDockerClient instance."""
    return LocalDockerClient()

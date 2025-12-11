"""Unit tests for LocalDockerClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_siren.sandbox_managers.abstract import ExecOutput
from prompt_siren.sandbox_managers.docker.local_client import (
    LocalContainer,
    LocalDockerClient,
    LocalNetwork,
)
from prompt_siren.sandbox_managers.docker.manager import (
    create_docker_client_from_config,
)
from prompt_siren.sandbox_managers.docker.plugins import (
    get_registered_docker_clients,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_docker():
    """Create a mock aiodocker.Docker instance."""
    docker = MagicMock()
    docker.close = AsyncMock()
    docker.images = MagicMock()
    docker.images.inspect = AsyncMock()
    docker.images.pull = AsyncMock()
    docker.images.build = AsyncMock()
    docker.images.delete = AsyncMock()
    docker.containers = MagicMock()
    docker.containers.create = AsyncMock()
    docker.containers.get = AsyncMock()
    docker.networks = MagicMock()
    docker.networks.create = AsyncMock()
    docker.networks.get = AsyncMock()
    return docker


@pytest.fixture
def client(mock_docker):
    """Create a LocalDockerClient with mocked aiodocker."""
    with patch("prompt_siren.sandbox_managers.docker.local_client.aiodocker.Docker") as mock_cls:
        mock_cls.return_value = mock_docker
        client = LocalDockerClient()
        client._docker = mock_docker
        return client


class TestLocalDockerClient:
    """Tests for LocalDockerClient."""

    async def test_close(self, client, mock_docker):
        """Test closing the client."""
        await client.close()
        mock_docker.close.assert_awaited_once()

    async def test_inspect_image(self, client, mock_docker):
        """Test inspecting an image."""
        mock_docker.images.inspect.return_value = {"Id": "sha256:abc123"}

        result = await client.inspect_image("python:3.12")

        mock_docker.images.inspect.assert_awaited_once_with("python:3.12")
        assert result == {"Id": "sha256:abc123"}

    async def test_pull_image(self, client, mock_docker):
        """Test pulling an image."""
        await client.pull_image("python:3.12")

        mock_docker.images.pull.assert_awaited_once_with(from_image="python:3.12")

    async def test_build_image(self, client, mock_docker):
        """Test building an image."""

        # Mock build stream
        async def mock_build(*args, **kwargs):
            yield {"stream": "Step 1/2 : FROM python:3.12"}
            yield {"stream": "Step 2/2 : RUN echo hello"}

        mock_docker.images.build = mock_build

        # Mock tarfile to avoid filesystem access
        with patch(
            "prompt_siren.sandbox_managers.docker.local_client.tarfile.open"
        ) as mock_tar_open:
            mock_tar = MagicMock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            logs = [
                log
                async for log in client.build_image(
                    context_path="/tmp/test",
                    tag="test:latest",
                    dockerfile_path="Dockerfile",
                    buildargs={"ARG1": "value1"},
                )
            ]

            assert len(logs) == 2
            assert "Step 1/2" in logs[0]["stream"]
            assert "Step 2/2" in logs[1]["stream"]
            # Verify tar.add was called with the context path
            mock_tar.add.assert_called_once()

    async def test_delete_image(self, client, mock_docker):
        """Test deleting an image."""
        await client.delete_image("test:latest", force=True)

        mock_docker.images.delete.assert_awaited_once_with("test:latest", force=True)

    async def test_create_container(self, client, mock_docker):
        """Test creating a container."""
        mock_container = MagicMock()
        mock_docker.containers.create.return_value = mock_container

        config = {"Image": "python:3.12"}
        container = await client.create_container(config, name="test-container")

        mock_docker.containers.create.assert_awaited_once_with(config, name="test-container")
        assert isinstance(container, LocalContainer)

    async def test_get_container(self, client, mock_docker):
        """Test getting a container."""
        mock_container = MagicMock()
        mock_docker.containers.get.return_value = mock_container

        container = await client.get_container("container123")

        mock_docker.containers.get.assert_awaited_once_with("container123")
        assert isinstance(container, LocalContainer)

    async def test_create_network(self, client, mock_docker):
        """Test creating a network."""
        mock_network = MagicMock()
        mock_docker.networks.create.return_value = mock_network

        config = {"Name": "test-network", "Driver": "bridge"}
        network = await client.create_network(config)

        mock_docker.networks.create.assert_awaited_once_with(config=config)
        assert isinstance(network, LocalNetwork)

    async def test_get_network(self, client, mock_docker):
        """Test getting a network."""
        mock_network = MagicMock()
        mock_docker.networks.get.return_value = mock_network

        network = await client.get_network("network123")

        mock_docker.networks.get.assert_awaited_once_with("network123")
        assert isinstance(network, LocalNetwork)


class TestLocalContainer:
    """Tests for LocalContainer wrapper."""

    async def test_start(self):
        """Test starting a container."""
        mock_container = MagicMock()
        mock_container.start = AsyncMock()
        container = LocalContainer(mock_container)

        await container.start()

        mock_container.start.assert_awaited_once()

    async def test_stop(self):
        """Test stopping a container."""
        mock_container = MagicMock()
        mock_container.stop = AsyncMock()
        container = LocalContainer(mock_container)

        await container.stop()

        mock_container.stop.assert_awaited_once()

    async def test_delete(self):
        """Test deleting a container."""
        mock_container = MagicMock()
        mock_container.delete = AsyncMock()
        container = LocalContainer(mock_container)

        await container.delete()

        mock_container.delete.assert_awaited_once()

    async def test_show(self):
        """Test showing container details."""
        mock_container = MagicMock()
        mock_container.show = AsyncMock(return_value={"Id": "abc123", "State": {"Running": True}})
        container = LocalContainer(mock_container)

        info = await container.show()

        assert info["Id"] == "abc123"
        assert info["State"]["Running"] is True

    async def test_exec(self):
        """Test executing a command."""
        # Mock the exec instance and stream
        mock_msg = MagicMock()
        mock_msg.stream = 1
        mock_msg.data = b"hello\n"

        mock_stream = MagicMock()
        mock_stream.read_out = AsyncMock(side_effect=[mock_msg, None])
        mock_stream.write_in = AsyncMock()
        mock_stream._resp = MagicMock()
        mock_stream._resp.connection = MagicMock()
        mock_stream._resp.connection.transport = None

        mock_exec = MagicMock()
        mock_exec.start = MagicMock(return_value=mock_stream)
        mock_exec.inspect = AsyncMock(return_value={"ExitCode": 0})

        mock_container = MagicMock()
        mock_container.exec = AsyncMock(return_value=mock_exec)
        container = LocalContainer(mock_container)

        result = await container.exec(
            cmd=["bash", "-c", "echo hello"],
            stdin=None,
            user="root",
            environment={"VAR": "value"},
            workdir="/app",
            timeout=30,
        )

        assert isinstance(result, ExecOutput)
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        mock_container.exec.assert_awaited_once()

    async def test_log(self):
        """Test getting container logs."""
        mock_container = MagicMock()
        mock_container.log = AsyncMock(return_value=["line1\n", "line2\n"])
        container = LocalContainer(mock_container)

        logs = await container.log(stdout=True, stderr=True)

        assert logs == ["line1\n", "line2\n"]

    async def test_commit(self):
        """Test committing a container."""
        mock_container = MagicMock()
        mock_container.commit = AsyncMock()
        container = LocalContainer(mock_container)

        await container.commit(repository="test-image", tag="latest")

        mock_container.commit.assert_awaited_once_with(repository="test-image", tag="latest")


class TestLocalNetwork:
    """Tests for LocalNetwork wrapper."""

    async def test_show(self):
        """Test showing network details."""
        mock_network = MagicMock()
        mock_network.show = AsyncMock(return_value={"Id": "net123", "Driver": "bridge"})
        network = LocalNetwork(mock_network)

        info = await network.show()

        assert info["Id"] == "net123"
        assert info["Driver"] == "bridge"

    async def test_delete(self):
        """Test deleting a network."""
        mock_network = MagicMock()
        mock_network.delete = AsyncMock()
        network = LocalNetwork(mock_network)

        await network.delete()

        mock_network.delete.assert_awaited_once()


class TestLocalDockerClientRegistry:
    """Tests for LocalDockerClient registry integration."""

    def test_local_client_can_be_created_via_registry(self):
        """Verify local client can be instantiated through the registry."""
        assert "local" in get_registered_docker_clients()

        # Mock aiodocker.Docker to avoid needing a running event loop
        with patch("prompt_siren.sandbox_managers.docker.local_client.aiodocker.Docker"):
            client = create_docker_client_from_config("local", {})
            assert isinstance(client, LocalDockerClient)


@pytest.mark.docker_integration
class TestLocalDockerClientIntegration:
    """Integration tests for LocalDockerClient that use actual Docker daemon."""

    async def test_client_lifecycle(self):
        """Test creating and closing a client."""
        client = LocalDockerClient()

        try:
            # Verify client is initialized
            assert client._docker is not None
        finally:
            await client.close()

    async def test_container_lifecycle(self):
        """Test creating, starting, stopping, and deleting a container."""
        client = LocalDockerClient()

        try:
            # Create container
            config = {
                "Image": "debian:bookworm-slim",
                "Cmd": ["sleep", "300"],
            }
            container = await client.create_container(config, name="test-local-container-lifecycle")

            try:
                # Start container
                await container.start()

                # Check container is running
                info = await container.show()
                assert info["State"]["Running"] is True

                # Stop container
                await container.stop()

                # Verify stopped
                info = await container.show()
                assert info["State"]["Running"] is False
            finally:
                # Clean up container
                await container.delete()
        finally:
            await client.close()

    async def test_exec_command(self):
        """Test executing a command in a container."""
        client = LocalDockerClient()

        try:
            # Create and start container
            config = {
                "Image": "debian:bookworm-slim",
                "Cmd": ["sleep", "300"],
            }
            container = await client.create_container(config, name="test-local-exec-command")

            try:
                await container.start()

                # Execute command
                result = await container.exec(
                    cmd=["echo", "hello", "world"],
                    stdin=None,
                    user="root",
                    environment=None,
                    workdir=None,
                    timeout=30,
                )

                # Verify output contains "hello world"
                assert result.stdout is not None
                assert "hello world" in result.stdout
                assert result.exit_code == 0
            finally:
                await container.delete()
        finally:
            await client.close()

    async def test_network_lifecycle(self):
        """Test creating and deleting a network."""
        client = LocalDockerClient()

        try:
            # Create network
            config = {
                "Name": "test-local-network-lifecycle",
                "Driver": "bridge",
                "Internal": False,
            }
            network = await client.create_network(config)

            try:
                # Verify network exists
                info = await network.show()
                assert info["Name"] == "test-local-network-lifecycle"
                assert info["Driver"] == "bridge"
            finally:
                # Clean up network
                await network.delete()
        finally:
            await client.close()

    async def test_image_operations(self):
        """Test pulling and inspecting images."""
        client = LocalDockerClient()

        try:
            # Pull a small test image
            await client.pull_image("alpine:latest")

            # Inspect the image
            image_info = await client.inspect_image("alpine:latest")
            assert image_info["Id"] is not None
            assert "alpine" in image_info["RepoTags"][0].lower()
        finally:
            await client.close()

    async def test_container_logs(self):
        """Test getting container logs."""
        client = LocalDockerClient()

        try:
            # Create and start container with a command that produces output
            config = {
                "Image": "debian:bookworm-slim",
                "Cmd": ["sh", "-c", "'echo \"test log output\" && sleep 300'"],
            }
            container = await client.create_container(config, name="test-local-container-logs")

            try:
                await container.start()

                # Wait a bit for the command to execute
                await asyncio.sleep(2)

                # Get logs
                logs = await container.log(stdout=True, stderr=True)

                # Verify logs contain expected output
                log_text = "\n".join(logs)
                assert "test log output" in log_text
            finally:
                await container.delete()
        finally:
            await client.close()

    async def test_container_commit(self):
        """Test committing a container to an image."""
        client = LocalDockerClient()

        try:
            # Create and start container
            config = {
                "Image": "debian:bookworm-slim",
                "Cmd": ["sleep", "300"],
            }
            container = await client.create_container(config, name="test-local-container-commit")

            try:
                await container.start()

                # Make a change in the container
                await container.exec(
                    cmd=["sh", "-c", "echo 'test' > /tmp/testfile"],
                    stdin=None,
                    user="root",
                    environment=None,
                    workdir=None,
                    timeout=30,
                )

                # Commit container to new image
                await container.commit(repository="test-local-committed-image", tag="v1")

                # Verify image exists
                image_info = await client.inspect_image("test-local-committed-image:v1")
                assert image_info["Id"] is not None
            finally:
                # Clean up
                await container.delete()
                try:
                    await client.delete_image("test-local-committed-image:v1", force=True)
                except Exception:
                    pass  # Ignore cleanup errors
        finally:
            await client.close()

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the build_images script."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prompt_siren.build_images import ImageBuilder
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError
from prompt_siren.sandbox_managers.image_spec import BuildImageSpec


class MockDockerClient:
    """Mock Docker client for testing."""

    def __init__(self) -> None:
        self.inspect_image = AsyncMock()
        self.delete_image = AsyncMock()
        self.tag_image = AsyncMock()
        self.push_image = AsyncMock()
        self.pull_image = AsyncMock()
        self.build_image = AsyncMock()


class TestSeederCallbackExecution:
    """Tests for seeder callback execution in ImageBuilder."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def builder(self, mock_docker: MockDockerClient, tmp_path: Path) -> ImageBuilder:
        return ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

    def _create_build_context(self, tmp_path: Path) -> Path:
        """Create a minimal build context with Dockerfile."""
        context_path = tmp_path / "build_context"
        context_path.mkdir()
        (context_path / "Dockerfile").write_text("FROM alpine:latest\n")
        return context_path

    @staticmethod
    async def _mock_build_gen(*_args: Any, **_kwargs: Any):
        """Async generator that yields successful build logs."""
        yield {"stream": "Step 1/1 : FROM alpine:latest"}
        yield {"stream": "Successfully built abc123"}

    @pytest.mark.anyio
    async def test_seeder_called_before_build(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that seeder is called before the Docker build."""
        context_path = self._create_build_context(tmp_path)
        call_order: list[str] = []

        async def mock_seeder() -> None:
            call_order.append("seeder")

        # Mock inspect_image to indicate image doesn't exist (404)
        mock_docker.inspect_image.side_effect = DockerClientError(message="Not found", status=404)

        # Track when build is called using an async generator
        async def tracking_build(*_args: Any, **_kwargs: Any):
            call_order.append("build")
            yield {"stream": "Step 1/1 : FROM alpine:latest"}
            yield {"stream": "Successfully built abc123"}

        mock_docker.build_image = tracking_build

        spec = BuildImageSpec(
            context_path=str(context_path),
            tag="test-seeder:latest",
            seeder=mock_seeder,
        )

        await builder._build_single_spec(spec)

        # Verify seeder was called before build
        assert call_order == ["seeder", "build"], f"Expected seeder before build, got: {call_order}"

    @pytest.mark.anyio
    async def test_seeder_exception_prevents_build(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that seeder exceptions propagate and prevent the build."""
        context_path = self._create_build_context(tmp_path)

        async def failing_seeder() -> None:
            raise ValueError("Seeder failed to connect to database")

        # Mock inspect_image to indicate image doesn't exist (404)
        mock_docker.inspect_image.side_effect = DockerClientError(message="Not found", status=404)
        build_called = False

        async def tracking_build(*_args: Any, **_kwargs: Any):
            nonlocal build_called
            build_called = True
            yield {"stream": "Should not be called"}

        mock_docker.build_image = tracking_build

        spec = BuildImageSpec(
            context_path=str(context_path),
            tag="test-failing-seeder:latest",
            seeder=failing_seeder,
        )

        with pytest.raises(ValueError, match="Seeder failed to connect"):
            await builder._build_single_spec(spec)

        # Verify build was never called
        assert not build_called, "Build should not be called when seeder fails"

    @pytest.mark.anyio
    async def test_seeder_not_called_when_image_exists(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that seeder is not called when image already exists."""
        context_path = self._create_build_context(tmp_path)
        seeder_called = False

        async def mock_seeder() -> None:
            nonlocal seeder_called
            seeder_called = True

        # Mock inspect_image to indicate image exists
        mock_docker.inspect_image.return_value = {"Id": "sha256:abc123"}

        spec = BuildImageSpec(
            context_path=str(context_path),
            tag="existing-image:latest",
            seeder=mock_seeder,
        )

        await builder._build_single_spec(spec)

        # Verify seeder was NOT called since image exists
        assert not seeder_called, "Seeder should not be called when image already exists"

    @pytest.mark.anyio
    async def test_build_without_seeder(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that builds work correctly without a seeder."""
        context_path = self._create_build_context(tmp_path)

        # Mock inspect_image to indicate image doesn't exist (404)
        mock_docker.inspect_image.side_effect = DockerClientError(message="Not found", status=404)
        mock_docker.build_image = self._mock_build_gen

        spec = BuildImageSpec(
            context_path=str(context_path),
            tag="no-seeder:latest",
            # No seeder specified
        )

        # Should complete without error
        await builder._build_single_spec(spec)

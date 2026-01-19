# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for the ImageBuilder class."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.build_images import ImageBuilder
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError


class MockDockerClient:
    """Mock Docker client for testing."""

    def __init__(self) -> None:
        self.inspect_image = AsyncMock()
        self.delete_image = AsyncMock()
        self.tag_image = AsyncMock()
        self.push_image = AsyncMock()
        self._build_results: list[dict[str, Any]] = []

    def set_build_results(self, results: list[dict[str, Any]]) -> None:
        """Set the results to yield from build_image."""
        self._build_results = results

    async def build_image(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        buildargs: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Mock build_image that yields configured results."""
        for result in self._build_results:
            yield result


class TestImageBuilderImageExists:
    """Tests for ImageBuilder.image_exists method."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def builder(self, mock_docker: MockDockerClient, tmp_path: Path) -> ImageBuilder:
        return ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

    @pytest.mark.anyio
    async def test_image_exists_returns_false_on_404(
        self, builder: ImageBuilder, mock_docker: MockDockerClient
    ) -> None:
        """Test that image_exists returns False when image not found (404)."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)

        result = await builder.image_exists("test:latest")

        assert result is False

    @pytest.mark.anyio
    async def test_image_exists_raises_on_other_errors(
        self, builder: ImageBuilder, mock_docker: MockDockerClient
    ) -> None:
        """Test that image_exists re-raises non-404 errors."""
        mock_docker.inspect_image.side_effect = DockerClientError("Docker daemon error", status=500)

        with pytest.raises(DockerClientError) as exc_info:
            await builder.image_exists("test:latest")

        assert exc_info.value.status == 500


class TestImageBuilderPushToRegistry:
    """Tests for ImageBuilder.push_to_registry method."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.mark.anyio
    async def test_push_to_registry_does_nothing_without_registry(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that push_to_registry does nothing when no registry configured."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry=None,
        )

        await builder.push_to_registry("test:latest")

        mock_docker.tag_image.assert_not_called()
        mock_docker.push_image.assert_not_called()

    @pytest.mark.anyio
    async def test_push_to_registry_tags_and_pushes(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that push_to_registry tags and pushes when registry configured."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry="my-registry.com/repo",
        )

        await builder.push_to_registry("test:latest")

        mock_docker.tag_image.assert_called_once_with(
            "test:latest", "my-registry.com/repo/test:latest"
        )
        mock_docker.push_image.assert_called_once_with("my-registry.com/repo/test:latest")


class TestImageBuilderBuildFromContext:
    """Tests for ImageBuilder.build_from_context method."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def builder(self, mock_docker: MockDockerClient, tmp_path: Path) -> ImageBuilder:
        return ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

    @pytest.mark.anyio
    async def test_build_skips_already_built_in_session(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that build skips images already built in this session."""
        # Manually add to built images
        builder._built_images.add("test:latest")

        await builder.build_from_context(
            context_path=str(tmp_path),
            tag="test:latest",
        )

        # Should not call inspect or build
        mock_docker.inspect_image.assert_not_called()

    @pytest.mark.anyio
    async def test_build_skips_existing_image_by_default(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that build skips existing images when rebuild_existing=False."""
        mock_docker.inspect_image.return_value = {"Id": "sha256:abc123"}

        await builder.build_from_context(
            context_path=str(tmp_path),
            tag="test:latest",
        )

        # Should check if exists but not build
        mock_docker.inspect_image.assert_called_once_with("test:latest")
        # Image should be tracked
        assert "test:latest" in builder._built_images

    @pytest.mark.anyio
    async def test_build_deletes_and_rebuilds_when_rebuild_existing(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that build deletes and rebuilds when rebuild_existing=True."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            rebuild_existing=True,
        )
        mock_docker.inspect_image.return_value = {"Id": "sha256:abc123"}
        mock_docker.set_build_results([{"stream": "Step 1/1 : FROM alpine\n"}])

        # Create a minimal context
        (tmp_path / "Dockerfile").write_text("FROM alpine")

        await builder.build_from_context(
            context_path=str(tmp_path),
            tag="test:latest",
        )

        # Should delete existing image
        mock_docker.delete_image.assert_called_once_with("test:latest", force=True)
        # Image should be tracked
        assert "test:latest" in builder._built_images

    @pytest.mark.anyio
    async def test_build_raises_on_build_error(
        self, builder: ImageBuilder, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Test that build raises RuntimeError on build error."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)
        mock_docker.set_build_results([{"error": "Build failed: no such file"}])

        (tmp_path / "Dockerfile").write_text("FROM alpine")

        with pytest.raises(RuntimeError) as exc_info:
            await builder.build_from_context(
                context_path=str(tmp_path),
                tag="test:latest",
            )

        assert "Failed to build image test:latest" in str(exc_info.value)

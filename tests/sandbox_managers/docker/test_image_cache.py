# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for the ImageCache class."""

from unittest.mock import AsyncMock

import pytest
from prompt_siren.sandbox_managers.docker.image_cache import (
    ImageCache,
    ImageNotFoundError,
)
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError
from prompt_siren.sandbox_managers.image_spec import (
    BuildImageSpec,
    BuildStage,
    MultiStageBuildImageSpec,
)


class MockDockerClient:
    """Mock Docker client for testing."""

    def __init__(self) -> None:
        self.inspect_image = AsyncMock()
        self.pull_image = AsyncMock()


class TestImageNotFoundError:
    """Tests for ImageNotFoundError exception."""

    def test_error_message_contains_tag(self) -> None:
        """Test that error message contains the image tag."""
        error = ImageNotFoundError("my-image:v1")
        assert "my-image:v1" in str(error)

    def test_error_message_mentions_build_command(self) -> None:
        """Test that error message mentions the build command."""
        error = ImageNotFoundError("test:latest")
        assert "prompt-siren-build-images" in str(error)


class TestImageCacheVerifyImageExists:
    """Tests for ImageCache._verify_image_exists method."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def cache(self, mock_docker: MockDockerClient) -> ImageCache:
        return ImageCache(
            docker=mock_docker,  # type: ignore[arg-type]
            batch_id="test-batch",
        )

    @pytest.mark.anyio
    async def test_verify_returns_tag_when_image_exists(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that _verify_image_exists returns tag when image exists."""
        mock_docker.inspect_image.return_value = {"Id": "sha256:abc123"}

        result = await cache._verify_image_exists("test:latest")

        assert result == "test:latest"
        mock_docker.inspect_image.assert_called_once_with("test:latest")

    @pytest.mark.anyio
    async def test_verify_raises_image_not_found_error(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that _verify_image_exists raises ImageNotFoundError when image doesn't exist."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)

        with pytest.raises(ImageNotFoundError) as exc_info:
            await cache._verify_image_exists("missing:latest")

        assert exc_info.value.tag == "missing:latest"
        assert "missing:latest" in str(exc_info.value)
        assert "prompt-siren-build-images" in str(exc_info.value)


class TestImageCacheEnsureImageAvailable:
    """Tests for ImageCache._ensure_image_available with BuildImageSpec and MultiStageBuildImageSpec."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.fixture
    def cache(self, mock_docker: MockDockerClient) -> ImageCache:
        return ImageCache(
            docker=mock_docker,  # type: ignore[arg-type]
            batch_id="test-batch",
        )

    @pytest.mark.anyio
    async def test_build_image_spec_raises_when_not_prebuilt(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that BuildImageSpec raises ImageNotFoundError when image is not pre-built."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)
        spec = BuildImageSpec(context_path="/some/path", tag="my-build:latest")

        with pytest.raises(ImageNotFoundError) as exc_info:
            await cache._ensure_image_available(spec)

        assert exc_info.value.tag == "my-build:latest"

    @pytest.mark.anyio
    async def test_multi_stage_build_spec_raises_when_not_prebuilt(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that MultiStageBuildImageSpec raises ImageNotFoundError when final image is not pre-built."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)
        spec = MultiStageBuildImageSpec(
            stages=[
                BuildStage(context_path="/stage1", tag="stage1:latest"),
                BuildStage(context_path="/stage2", tag="stage2:latest", parent_tag="stage1:latest"),
            ],
            final_tag="stage2:latest",
        )

        with pytest.raises(ImageNotFoundError) as exc_info:
            await cache._ensure_image_available(spec)

        assert exc_info.value.tag == "stage2:latest"

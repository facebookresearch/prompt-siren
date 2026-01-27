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
    DerivedImageSpec,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from pydantic import ValidationError


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

    @pytest.mark.anyio
    async def test_derived_image_spec_raises_when_not_prebuilt(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that DerivedImageSpec raises ImageNotFoundError when image is not pre-built."""
        mock_docker.inspect_image.side_effect = DockerClientError("Image not found", status=404)
        spec = DerivedImageSpec(
            base_image_tag="base:latest",
            dockerfile_extra="RUN pip install something",
            tag="derived:latest",
        )

        with pytest.raises(ImageNotFoundError) as exc_info:
            await cache._ensure_image_available(spec)

        assert exc_info.value.tag == "derived:latest"


class TestImageCacheNon404Errors:
    """Tests for proper handling of non-404 Docker errors."""

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
    async def test_verify_image_raises_docker_error_for_non_404(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that non-404 DockerClientError is re-raised, not converted to ImageNotFoundError."""
        mock_docker.inspect_image.side_effect = DockerClientError("Permission denied", status=403)

        with pytest.raises(DockerClientError) as exc_info:
            await cache._verify_image_exists("test:latest")

        assert exc_info.value.status == 403
        assert "Permission denied" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_pull_image_raises_docker_error_for_non_404(
        self, cache: ImageCache, mock_docker: MockDockerClient
    ) -> None:
        """Test that non-404 errors during pull are propagated, not silently ignored."""
        mock_docker.inspect_image.side_effect = DockerClientError("Daemon not running", status=500)
        spec = PullImageSpec(tag="test:latest")

        with pytest.raises(DockerClientError) as exc_info:
            await cache._pull_image(spec)

        assert exc_info.value.status == 500


class TestImageCacheGetCacheKey:
    """Tests for ImageCache._get_cache_key static method."""

    def test_get_cache_key_build_image_spec(self) -> None:
        """Test cache key generation for BuildImageSpec."""
        spec = BuildImageSpec(context_path="/some/path", tag="build:v1")
        key = ImageCache._get_cache_key(spec)
        assert key == "build:build:v1"

    def test_get_cache_key_multi_stage_spec(self) -> None:
        """Test cache key generation for MultiStageBuildImageSpec."""
        spec = MultiStageBuildImageSpec(
            stages=[BuildStage(context_path="/stage1", tag="final:v1")],
            final_tag="final:v1",
        )
        key = ImageCache._get_cache_key(spec)
        assert key == "multistage:final:v1"

    def test_get_cache_key_derived_image_spec(self) -> None:
        """Test cache key generation for DerivedImageSpec."""
        spec = DerivedImageSpec(
            base_image_tag="base:v1",
            dockerfile_extra="RUN echo test",
            tag="derived:v1",
        )
        key = ImageCache._get_cache_key(spec)
        assert key == "derived:derived:v1"

    def test_get_cache_key_pull_image_spec(self) -> None:
        """Test cache key generation for PullImageSpec."""
        spec = PullImageSpec(tag="alpine:3.18")
        key = ImageCache._get_cache_key(spec)
        assert key == "pull:alpine:3.18"


class TestDerivedImageSpecValidation:
    """Tests for DerivedImageSpec validators."""

    def test_empty_dockerfile_extra_raises(self) -> None:
        """Empty dockerfile_extra should be rejected."""
        with pytest.raises(ValidationError):
            DerivedImageSpec(
                base_image_tag="base:v1",
                dockerfile_extra="",
                tag="derived:v1",
            )

    def test_tag_same_as_base_image_tag_raises(self) -> None:
        """tag must differ from base_image_tag."""
        with pytest.raises(ValidationError, match="must differ from"):
            DerivedImageSpec(
                base_image_tag="same:v1",
                dockerfile_extra="RUN echo x",
                tag="same:v1",
            )

    def test_valid_derived_image_spec(self) -> None:
        """Valid construction succeeds."""
        spec = DerivedImageSpec(
            base_image_tag="base:v1",
            dockerfile_extra="RUN echo x",
            tag="derived:v1",
        )
        assert spec.tag == "derived:v1"
        assert spec.base_image_tag == "base:v1"


class TestMultiStageBuildImageSpecValidation:
    """Tests for MultiStageBuildImageSpec validators."""

    def test_empty_stages_raises(self) -> None:
        """Empty stages list should be rejected."""
        with pytest.raises(ValidationError, match="stages must not be empty"):
            MultiStageBuildImageSpec(stages=[], final_tag="app:latest")

    def test_final_tag_mismatch_raises(self) -> None:
        """final_tag must match the last stage's tag."""
        with pytest.raises(ValidationError, match="final_tag must match"):
            MultiStageBuildImageSpec(
                stages=[
                    BuildStage(context_path="/stage1", tag="stage1:latest"),
                ],
                final_tag="wrong:latest",
            )

    def test_valid_multi_stage_spec(self) -> None:
        """Valid construction succeeds."""
        spec = MultiStageBuildImageSpec(
            stages=[
                BuildStage(context_path="/base", tag="base:latest"),
                BuildStage(
                    context_path="/app",
                    tag="app:latest",
                    parent_tag="base:latest",
                ),
            ],
            final_tag="app:latest",
        )
        assert spec.tag == "app:latest"
        assert len(spec.stages) == 2

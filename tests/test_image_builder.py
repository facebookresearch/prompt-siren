# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for the ImageBuilder class."""

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from prompt_siren.build_images import _handle_build_failures, BuildError, ImageBuilder
from prompt_siren.sandbox_managers.docker.plugins.errors import DockerClientError
from prompt_siren.sandbox_managers.image_spec import (
    BuildImageSpec,
    BuildStage,
    DerivedImageSpec,
    MultiStageBuildImageSpec,
)


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


class TestHandleBuildFailures:
    """Tests for _handle_build_failures function."""

    def test_empty_list_does_not_raise(self) -> None:
        """Test that an empty list does not raise any exception."""
        _handle_build_failures([])  # Should not raise

    def test_single_error_raises_exception_group(self) -> None:
        """Test that a single error raises an ExceptionGroup."""
        error = ValueError("Build failed")
        build_error = BuildError(image_tag="test:latest", error=error)

        with pytest.raises(ExceptionGroup) as exc_info:
            _handle_build_failures([build_error])

        assert len(exc_info.value.exceptions) == 1
        assert exc_info.value.exceptions[0] is error
        assert "1 image build(s) failed" in str(exc_info.value)
        assert "test:latest" in str(exc_info.value)

    def test_multiple_errors_raises_exception_group_with_all(self) -> None:
        """Test that multiple errors raises an ExceptionGroup with all errors."""
        error1 = ValueError("Build 1 failed")
        error2 = RuntimeError("Build 2 failed")
        build_errors = [
            BuildError(image_tag="image1:latest", error=error1),
            BuildError(image_tag="image2:v1", error=error2),
        ]

        with pytest.raises(ExceptionGroup) as exc_info:
            _handle_build_failures(build_errors)

        assert len(exc_info.value.exceptions) == 2
        assert error1 in exc_info.value.exceptions
        assert error2 in exc_info.value.exceptions
        assert "2 image build(s) failed" in str(exc_info.value)
        assert "image1:latest" in str(exc_info.value)
        assert "image2:v1" in str(exc_info.value)


class TestImageBuilderBuildAllSpecs:
    """Tests for ImageBuilder.build_all_specs method."""

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
    async def test_build_all_specs_orders_base_before_derived(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Verify derived specs are built after all base specs complete."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

        # Track build order at the _build_single_spec and build_modified_image level
        build_order: list[str] = []

        # Create context dir with Dockerfile
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        # Set up mock to track calls and simulate successful builds
        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)
        mock_docker.set_build_results([{"stream": "Done\n"}])

        original_build_single = builder._build_single_spec

        async def tracking_build_single(
            spec: BuildImageSpec | MultiStageBuildImageSpec,
        ) -> None:
            build_order.append(f"base:{spec.tag}")
            await original_build_single(spec)

        original_modified = builder.build_modified_image

        async def tracking_modified(base_tag: str, dockerfile_extra: str, output_tag: str) -> None:
            build_order.append(f"derived:{output_tag}")
            await original_modified(base_tag, dockerfile_extra, output_tag)

        builder._build_single_spec = tracking_build_single  # type: ignore[method-assign]
        builder.build_modified_image = tracking_modified  # type: ignore[method-assign]

        # Create specs: 2 base specs and 2 derived specs
        specs = [
            BuildImageSpec(
                context_path=str(context_dir),
                tag="base1:latest",
            ),
            DerivedImageSpec(
                base_image_tag="base1:latest",
                dockerfile_extra="RUN echo 'derived1'",
                tag="derived1:latest",
            ),
            BuildImageSpec(
                context_path=str(context_dir),
                tag="base2:latest",
            ),
            DerivedImageSpec(
                base_image_tag="base2:latest",
                dockerfile_extra="RUN echo 'derived2'",
                tag="derived2:latest",
            ),
        ]

        errors = await builder.build_all_specs(specs)

        assert errors == []
        # All base specs should be built before any derived specs
        assert build_order == [
            "base:base1:latest",
            "base:base2:latest",
            "derived:derived1:latest",
            "derived:derived2:latest",
        ]

    @pytest.mark.anyio
    async def test_build_all_specs_handles_multi_stage_specs(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Verify multi-stage build specs are handled correctly."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

        # Create context dir with Dockerfile
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)
        mock_docker.set_build_results([{"stream": "Done\n"}])

        specs = [
            MultiStageBuildImageSpec(
                stages=[
                    BuildStage(
                        context_path=str(context_dir),
                        tag="stage1:latest",
                        dockerfile_path=None,
                        parent_tag=None,
                        build_args=None,
                    ),
                    BuildStage(
                        context_path=str(context_dir),
                        tag="stage2:latest",
                        dockerfile_path=None,
                        parent_tag="stage1:latest",
                        build_args=None,
                    ),
                ],
            ),
        ]

        errors = await builder.build_all_specs(specs)

        assert errors == []
        # Both stages should be built
        assert "stage1:latest" in builder._built_images
        assert "stage2:latest" in builder._built_images

    @pytest.mark.anyio
    async def test_build_all_specs_collects_errors_without_stopping(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Verify build errors are collected but don't stop other builds."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

        # Create context dir with Dockerfile
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)

        # First build fails, second succeeds
        async def build_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            tag = kwargs.get("tag", args[1] if len(args) > 1 else None)
            if tag == "fail:latest":
                yield {"error": "Build failed"}
            else:
                yield {"stream": "Done\n"}

        mock_docker.build_image = build_side_effect  # type: ignore[method-assign]

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="fail:latest"),
            BuildImageSpec(context_path=str(context_dir), tag="success:latest"),
        ]

        errors = await builder.build_all_specs(specs)

        assert len(errors) == 1
        assert errors[0].image_tag == "fail:latest"
        # Second image should still have been built
        assert "success:latest" in builder._built_images

    @pytest.mark.anyio
    async def test_build_all_specs_skips_derived_when_base_fails(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """Verify derived specs are skipped when their base image fails to build."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )

        # Create context dir with Dockerfile
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)

        # First base build fails, second succeeds
        async def build_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            tag = kwargs.get("tag", args[1] if len(args) > 1 else None)
            if tag == "fail-base:latest":
                yield {"error": "Build failed"}
            else:
                yield {"stream": "Done\n"}

        mock_docker.build_image = build_side_effect  # type: ignore[method-assign]

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="fail-base:latest"),
            BuildImageSpec(context_path=str(context_dir), tag="success-base:latest"),
            DerivedImageSpec(
                base_image_tag="fail-base:latest",
                dockerfile_extra="RUN echo derived-from-fail",
                tag="derived-from-fail:latest",
            ),
            DerivedImageSpec(
                base_image_tag="success-base:latest",
                dockerfile_extra="RUN echo derived-from-success",
                tag="derived-from-success:latest",
            ),
        ]

        errors = await builder.build_all_specs(specs)

        # Should have 2 errors: fail-base and derived-from-fail (skipped because base failed)
        assert len(errors) == 2
        error_tags = {e.image_tag for e in errors}
        assert "fail-base:latest" in error_tags
        assert "derived-from-fail:latest" in error_tags
        # The derived-from-fail error should mention the base image
        derived_error = next(e for e in errors if e.image_tag == "derived-from-fail:latest")
        assert "fail-base:latest" in str(derived_error.error)

        # success-base and derived-from-success should have been built
        assert "success-base:latest" in builder._built_images
        assert "derived-from-success:latest" in builder._built_images


class TestBuildAllSpecsPushToRegistry:
    """Tests for push-to-registry behavior in build_all_specs."""

    @pytest.fixture
    def mock_docker(self) -> MockDockerClient:
        return MockDockerClient()

    @pytest.mark.anyio
    async def test_push_called_for_each_successful_spec(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """push_image is called for each successfully built spec."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry="test-registry",
        )

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)
        mock_docker.set_build_results([{"stream": "Done\n"}])

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="img1:latest"),
            BuildImageSpec(context_path=str(context_dir), tag="img2:latest"),
        ]

        errors = await builder.build_all_specs(specs)

        assert errors == []
        assert mock_docker.push_image.call_count == 2
        pushed_tags = [call.args[0] for call in mock_docker.push_image.call_args_list]
        assert "test-registry/img1:latest" in pushed_tags
        assert "test-registry/img2:latest" in pushed_tags

    @pytest.mark.anyio
    async def test_push_not_called_for_failed_specs(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """push_image is NOT called for specs that failed to build."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry="test-registry",
        )

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)

        async def build_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            tag = kwargs.get("tag", args[1] if len(args) > 1 else None)
            if tag == "fail:latest":
                yield {"error": "Build failed"}
            else:
                yield {"stream": "Done\n"}

        mock_docker.build_image = build_side_effect  # type: ignore[method-assign]

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="fail:latest"),
            BuildImageSpec(context_path=str(context_dir), tag="success:latest"),
        ]

        errors = await builder.build_all_specs(specs)

        assert len(errors) == 1
        assert errors[0].image_tag == "fail:latest"
        # Only pushed for the successful spec
        assert mock_docker.push_image.call_count == 1
        mock_docker.push_image.assert_called_once_with("test-registry/success:latest")

    @pytest.mark.anyio
    async def test_push_not_called_for_skipped_derived_specs(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """push_image is NOT called for derived specs skipped due to base failure."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry="test-registry",
        )

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)

        # Base build always fails
        async def build_side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {"error": "Build failed"}

        mock_docker.build_image = build_side_effect  # type: ignore[method-assign]

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="base:latest"),
            DerivedImageSpec(
                base_image_tag="base:latest",
                dockerfile_extra="RUN echo derived",
                tag="derived:latest",
            ),
        ]

        errors = await builder.build_all_specs(specs)

        # Both should have errors (base failed, derived skipped)
        assert len(errors) == 2
        # push_image should never have been called
        mock_docker.push_image.assert_not_called()

    @pytest.mark.anyio
    async def test_push_failure_does_not_mark_base_as_failed(
        self, mock_docker: MockDockerClient, tmp_path: Path
    ) -> None:
        """A push failure should NOT cause derived images to be skipped."""
        builder = ImageBuilder(
            docker_client=mock_docker,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            registry="test-registry",
        )

        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "Dockerfile").write_text("FROM alpine")

        mock_docker.inspect_image.side_effect = DockerClientError("Not found", status=404)
        mock_docker.set_build_results([{"stream": "Done\n"}])
        # Push always fails
        mock_docker.push_image.side_effect = RuntimeError("Push failed")

        specs = [
            BuildImageSpec(context_path=str(context_dir), tag="base:latest"),
            DerivedImageSpec(
                base_image_tag="base:latest",
                dockerfile_extra="RUN echo derived",
                tag="derived:latest",
            ),
        ]

        errors = await builder.build_all_specs(specs)

        # Both push errors, but the derived image should have been built (not skipped)
        error_tags = [e.image_tag for e in errors]
        assert "base:latest" in error_tags
        assert "derived:latest" in error_tags
        # Derived should be built successfully (the error is from push, not build skip)
        assert "derived:latest" in builder._built_images
        # The derived error should be the push error, not "base image failed to build"
        derived_error = next(e for e in errors if e.image_tag == "derived:latest")
        assert "Push failed" in str(derived_error.error)

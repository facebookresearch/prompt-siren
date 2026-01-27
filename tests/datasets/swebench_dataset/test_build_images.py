# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench image building (image tag functions, build_dataset_images)."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

swebench = pytest.importorskip("swebench")

from prompt_siren.build_images import build_dataset_images, ImageBuilder  # noqa: E402
from prompt_siren.datasets.swebench_dataset.image_tags import (  # noqa: E402
    get_benign_image_tag,
    get_pair_image_tag,
)


class MockDockerClient:
    """Mock Docker client for testing."""

    def __init__(self) -> None:
        self.inspect_image = AsyncMock()
        self.delete_image = AsyncMock()
        self.tag_image = AsyncMock()
        self.push_image = AsyncMock()
        self.build_image = AsyncMock()


class TestImageTagFunctions:
    """Test image tag generation functions."""

    def test_get_benign_image_tag(self) -> None:
        """Test benign image tag generation."""
        tag = get_benign_image_tag("django__django-11179")
        assert tag == "siren-swebench-benign:django__django-11179"

    def test_get_pair_image_tag(self) -> None:
        """Test pair image tag generation."""
        tag = get_pair_image_tag("django__django-11179", "env_exfil_task")
        assert tag == "siren-swebench-pair:django__django-11179__env_exfil_task"


class TestBuildDatasetImagesValidation:
    """Tests for build_dataset_images."""

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
    async def test_uses_default_config(self, builder: ImageBuilder) -> None:
        """Verify build_dataset_images uses default config and returns specs."""
        # build_dataset_images uses default config, so it should return specs
        # without any config overrides
        errors = await build_dataset_images("swebench", builder)
        # With default config and mock docker client, all builds fail with
        # mock errors, but the function should complete without raising
        assert isinstance(errors, list)

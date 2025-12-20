# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the build_images script."""

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.build_images import (
    get_benign_image_tag,
    get_pair_image_tag,
)


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

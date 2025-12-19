# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench image tag utilities.

These tests can run without swebench since image_tags.py is a standalone module.
"""

import pytest

# Skip this entire module if swebench is not installed, as the import chain
# through the package __init__.py requires it
pytest.importorskip("swebench")

from prompt_siren.datasets.swebench_dataset.image_tags import (
    get_basic_agent_image_tag,
    get_benign_image_tag,
    get_malicious_image_tag,
    get_pair_image_tag,
    normalize_tag,
)


class TestNormalizeTag:
    """Test tag normalization."""

    def test_normalize_tag_replaces_slashes(self) -> None:
        """Test that slashes are replaced."""
        assert normalize_tag("astropy/astropy") == "astropy_astropy"

    def test_normalize_tag_replaces_colons(self) -> None:
        """Test that colons are replaced."""
        assert normalize_tag("image:tag") == "image_tag"

    def test_normalize_tag_replaces_spaces(self) -> None:
        """Test that spaces are replaced."""
        assert normalize_tag("my image") == "my_image"

    def test_normalize_tag_lowercases(self) -> None:
        """Test that tags are lowercased."""
        assert normalize_tag("MyImage") == "myimage"

    def test_normalize_tag_combined(self) -> None:
        """Test multiple replacements together."""
        assert normalize_tag("Django/django:11179") == "django_django_11179"


class TestGetBenignImageTag:
    """Test benign image tag generation."""

    def test_get_benign_image_tag(self) -> None:
        """Test benign tag format."""
        tag = get_benign_image_tag("django__django-11179")
        assert tag == "siren-swebench-benign:django__django-11179"

    def test_get_benign_image_tag_with_special_chars(self) -> None:
        """Test tag normalization."""
        tag = get_benign_image_tag("astropy/astropy-12345")
        assert tag == "siren-swebench-benign:astropy_astropy-12345"


class TestGetMaliciousImageTag:
    """Test malicious image tag generation."""

    def test_get_malicious_image_tag(self) -> None:
        """Test malicious tag format."""
        tag = get_malicious_image_tag("env_direct_exfil_task")
        assert tag == "siren-swebench-malicious:env_direct_exfil_task"


class TestGetPairImageTag:
    """Test pair image tag generation."""

    def test_get_pair_image_tag(self) -> None:
        """Test pair tag format."""
        tag = get_pair_image_tag("django__django-11179", "env_direct_exfil_task")
        assert tag == "siren-swebench-pair:django__django-11179__env_direct_exfil_task"

    def test_get_pair_image_tag_with_special_chars(self) -> None:
        """Test pair tag normalization."""
        tag = get_pair_image_tag("astropy/astropy-12345", "some_task")
        assert tag == "siren-swebench-pair:astropy_astropy-12345__some_task"


class TestGetBasicAgentImageTag:
    """Test basic agent image tag generation."""

    def test_get_basic_agent_image_tag(self) -> None:
        """Test basic agent tag format."""
        tag = get_basic_agent_image_tag()
        assert tag == "siren-swebench-basic-agent:latest"

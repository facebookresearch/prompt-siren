# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench image tag utilities.

These tests can run without swebench since image_tags.py is a standalone module.
"""

import pytest

# Skip this entire module if swebench is not installed, as the import chain
# through the package __init__.py requires it
pytest.importorskip("swebench")

from prompt_siren.datasets.swebench_dataset.image_tags import (
    apply_registry_prefix,
    get_basic_agent_image_tag,
    get_base_image_tag,
    get_benign_image_tag,
    get_env_image_tag,
    get_malicious_image_tag,
    get_pair_image_tag,
    get_service_image_tag,
    normalize_tag,
)


class TestNormalizeTag:
    """Test tag normalization."""

    def test_normalize_tag_replaces_slashes(self) -> None:
        """Test that slashes are replaced."""
        assert normalize_tag("astropy/astropy") == "astropy__astropy"

    def test_normalize_tag_replaces_colons(self) -> None:
        """Test that colons are replaced."""
        assert normalize_tag("image:tag") == "image__tag"

    def test_normalize_tag_replaces_spaces(self) -> None:
        """Test that spaces are replaced."""
        assert normalize_tag("my image") == "my__image"

    def test_normalize_tag_lowercases(self) -> None:
        """Test that tags are lowercased."""
        assert normalize_tag("MyImage") == "myimage"

    def test_normalize_tag_combined(self) -> None:
        """Test multiple replacements together."""
        assert normalize_tag("Django/django:11179") == "django__django__11179"


class TestApplyRegistryPrefix:
    """Test registry prefix application."""

    def test_apply_registry_prefix_with_registry(self) -> None:
        """Test that registry prefix is applied."""
        tag = apply_registry_prefix("siren-swebench-benign:instance", "my-registry.com/repo")
        assert tag == "my-registry.com/repo/siren-swebench-benign:instance"

    def test_apply_registry_prefix_without_registry(self) -> None:
        """Test that tag is unchanged when registry is None."""
        tag = apply_registry_prefix("siren-swebench-benign:instance", None)
        assert tag == "siren-swebench-benign:instance"

    def test_apply_registry_prefix_with_empty_string(self) -> None:
        """Test that tag is unchanged when registry is empty string."""
        tag = apply_registry_prefix("siren-swebench-benign:instance", "")
        assert tag == "siren-swebench-benign:instance"

    def test_apply_registry_prefix_does_not_double_prefix(self) -> None:
        """Test that tags with existing registry are not double-prefixed."""
        tag = apply_registry_prefix(
            "existing-registry.com/repo/siren-swebench-benign:instance", "my-registry.com/repo"
        )
        assert tag == "existing-registry.com/repo/siren-swebench-benign:instance"


class TestGetBenignImageTag:
    """Test benign image tag generation."""

    def test_get_benign_image_tag(self) -> None:
        """Test benign tag format."""
        tag = get_benign_image_tag("django__django-11179")
        assert tag == "siren-swebench-benign:django__django-11179"

    def test_get_benign_image_tag_with_special_chars(self) -> None:
        """Test tag normalization."""
        tag = get_benign_image_tag("astropy/astropy-12345")
        assert tag == "siren-swebench-benign:astropy__astropy-12345"

    def test_get_benign_image_tag_with_registry(self) -> None:
        """Test benign tag with registry prefix."""
        tag = get_benign_image_tag("django__django-11179", registry="my-registry.com/repo")
        assert tag == "my-registry.com/repo/siren-swebench-benign:django__django-11179"


class TestGetServiceImageTag:
    """Test service image tag generation."""

    def test_get_service_image_tag(self) -> None:
        """Test service tag format."""
        tag = get_service_image_tag("env_direct_exfil_task")
        assert tag == "siren-swebench-service:env_direct_exfil_task"

    def test_get_service_image_tag_with_registry(self) -> None:
        """Test service tag with registry prefix."""
        tag = get_service_image_tag("env_direct_exfil_task", registry="my-registry.com/repo")
        assert tag == "my-registry.com/repo/siren-swebench-service:env_direct_exfil_task"

    def test_get_malicious_image_tag_is_alias(self) -> None:
        """Test deprecated malicious tag helper delegates to service tags."""
        tag = get_malicious_image_tag("env_direct_exfil_task")
        assert tag == "siren-swebench-service:env_direct_exfil_task"


class TestGetPairImageTag:
    """Test pair image tag generation."""

    def test_get_pair_image_tag(self) -> None:
        """Test pair tag format."""
        tag = get_pair_image_tag("django__django-11179", "env_direct_exfil_task")
        assert tag == "siren-swebench-pair:django__django-11179__env_direct_exfil_task"

    def test_get_pair_image_tag_with_special_chars(self) -> None:
        """Test pair tag normalization."""
        tag = get_pair_image_tag("astropy/astropy-12345", "some_task")
        assert tag == "siren-swebench-pair:astropy__astropy-12345__some_task"

    def test_get_pair_image_tag_with_registry(self) -> None:
        """Test pair tag with registry prefix."""
        tag = get_pair_image_tag(
            "django__django-11179", "env_direct_exfil_task", registry="my-registry.com/repo"
        )
        assert (
            tag
            == "my-registry.com/repo/siren-swebench-pair:django__django-11179__env_direct_exfil_task"
        )


class TestGetBasicAgentImageTag:
    """Test basic agent image tag generation."""

    def test_get_basic_agent_image_tag(self) -> None:
        """Test basic agent tag format."""
        tag = get_basic_agent_image_tag()
        assert tag == "siren-swebench-agent:basic"

    def test_get_basic_agent_image_tag_with_registry(self) -> None:
        """Test basic agent tag with registry prefix."""
        tag = get_basic_agent_image_tag(registry="my-registry.com/repo")
        assert tag == "my-registry.com/repo/siren-swebench-agent:basic"


class TestGetCacheImageTags:
    """Test base/env cache image tag generation."""

    def test_get_base_image_tag(self) -> None:
        """Test base cache tag format."""
        tag = get_base_image_tag("deadbeefcafebabe")
        assert tag == "siren-swebench-base:deadbeefcafebabe"

    def test_get_env_image_tag(self) -> None:
        """Test env cache tag format."""
        tag = get_env_image_tag("deadbeefcafebabe")
        assert tag == "siren-swebench-env:deadbeefcafebabe"

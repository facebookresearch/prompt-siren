# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for shared dataset image tag helpers."""

from prompt_siren.datasets.image_tags import (
    apply_registry_prefix,
    make_dataset_tag,
    normalize_tag_component,
)


class TestNormalizeTagComponent:
    """Tests for tag component normalization."""

    def test_replaces_slashes_colons_spaces_and_lowercases(self) -> None:
        value = "My/Repo:Version 1"
        assert normalize_tag_component(value) == "my__repo__version__1"


class TestApplyRegistryPrefix:
    """Tests for registry prefix application."""

    def test_prefixes_when_registry_set(self) -> None:
        tag = apply_registry_prefix("siren-test-benign:case", "registry.io/repo")
        assert tag == "registry.io/repo/siren-test-benign:case"

    def test_noop_when_registry_none(self) -> None:
        tag = apply_registry_prefix("siren-test-benign:case", None)
        assert tag == "siren-test-benign:case"

    def test_noop_when_registry_empty(self) -> None:
        tag = apply_registry_prefix("siren-test-benign:case", "")
        assert tag == "siren-test-benign:case"

    def test_does_not_double_prefix(self) -> None:
        tag = apply_registry_prefix(
            "existing.io/repo/siren-test-benign:case", "registry.io/repo"
        )
        assert tag == "existing.io/repo/siren-test-benign:case"


class TestMakeDatasetTag:
    """Tests for dataset tag construction."""

    def test_builds_expected_tag(self) -> None:
        tag = make_dataset_tag("siren-test", "benign", "My/Repo:Version 1")
        assert tag == "siren-test-benign:my__repo__version__1"

    def test_builds_expected_tag_with_registry(self) -> None:
        tag = make_dataset_tag("siren-test", "service", "Task Name", registry="registry.io/repo")
        assert tag == "registry.io/repo/siren-test-service:task__name"

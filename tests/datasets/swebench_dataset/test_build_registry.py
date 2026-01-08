# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench malicious task build registry."""

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.datasets.swebench_dataset.malicious_tasks.build_registry import (
    get_all_service_container_build_specs,
    get_service_container_build_spec,
    SERVICE_CONTAINER_BUILD_SPECS,
)
from prompt_siren.sandbox_managers.image_spec import BuildImageSpec


class TestBuildRegistry:
    """Test build registry functionality."""

    def test_service_container_build_specs_not_empty(self) -> None:
        """Test that the registry has entries."""
        assert len(SERVICE_CONTAINER_BUILD_SPECS) > 0

    def test_all_entries_are_build_image_specs(self) -> None:
        """Test that all entries are BuildImageSpec."""
        for tag, spec in SERVICE_CONTAINER_BUILD_SPECS.items():
            assert isinstance(spec, BuildImageSpec), f"{tag} is not a BuildImageSpec"
            assert spec.tag == tag, f"Tag mismatch for {tag}"

    def test_get_all_service_container_build_specs(self) -> None:
        """Test get_all_service_container_build_specs returns correct list."""
        specs = get_all_service_container_build_specs()
        assert len(specs) == len(SERVICE_CONTAINER_BUILD_SPECS)
        assert all(isinstance(s, BuildImageSpec) for s in specs)

    def test_get_service_container_build_spec_found(self) -> None:
        """Test get_service_container_build_spec for existing tag."""
        # Get a tag from the registry
        some_tag = next(iter(SERVICE_CONTAINER_BUILD_SPECS.keys()))
        spec = get_service_container_build_spec(some_tag)
        assert spec is not None
        assert isinstance(spec, BuildImageSpec)
        assert spec.tag == some_tag

    def test_get_service_container_build_spec_not_found(self) -> None:
        """Test get_service_container_build_spec for non-existing tag."""
        spec = get_service_container_build_spec("nonexistent:tag")
        assert spec is None


class TestBuildSpecsHaveValidContextPaths:
    """Test that build specs have valid context paths."""

    def test_context_paths_are_strings(self) -> None:
        """Test that context paths are strings."""
        for tag, spec in SERVICE_CONTAINER_BUILD_SPECS.items():
            assert isinstance(spec.context_path, str), f"{tag} has non-string context_path"

    def test_basic_agent_is_registered(self) -> None:
        """Test that basic agent is in the registry."""
        from prompt_siren.datasets.swebench_dataset.image_tags import (
            get_basic_agent_image_tag,
        )

        basic_agent_tag = get_basic_agent_image_tag()
        spec = get_service_container_build_spec(basic_agent_tag)
        assert spec is not None
        assert spec.tag == basic_agent_tag

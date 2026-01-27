# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for dataset registry image building functions."""

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.datasets import registry
from prompt_siren.datasets.registry import (
    get_datasets_with_image_specs,
    get_image_build_specs,
)
from prompt_siren.datasets.swebench_dataset.config import SwebenchDatasetConfig
from prompt_siren.datasets.swebench_dataset.dataset import SwebenchDataset
from prompt_siren.sandbox_managers.image_spec import (
    BuildImageSpec,
    DerivedImageSpec,
    MultiStageBuildImageSpec,
)


class TestGetDatasetsWithImageSpecs:
    """Tests for get_datasets_with_image_specs function."""

    def test_includes_swebench(self) -> None:
        """Test that swebench is included in image-buildable datasets."""
        datasets = get_datasets_with_image_specs()
        assert "swebench" in datasets


class TestGetImageBuildSpecs:
    """Tests for get_image_build_specs function."""

    def test_raises_for_unsupported_dataset(self) -> None:
        """Verify ValueError when dataset doesn't support image building."""
        with pytest.raises(ValueError, match="does not support image building"):
            get_image_build_specs("nonexistent-dataset", SwebenchDatasetConfig())


class TestSwebenchDatasetGetImageBuildSpecs:
    """Tests for SwebenchDataset.get_image_build_specs classmethod."""

    def test_returns_specs_for_valid_instance(self) -> None:
        """Test that valid instance IDs return image specs."""
        # Use a valid instance ID from INSTANCE_INJECTION_MAPPING
        config = SwebenchDatasetConfig()
        specs = SwebenchDataset.get_image_build_specs(config)

        # Should have at least one multi-stage build spec for benign task
        multi_stage_specs = [s for s in specs if isinstance(s, MultiStageBuildImageSpec)]
        assert len(multi_stage_specs) >= 1

        # Should have derived specs for pairs (benign x malicious with dockerfile_extra)
        derived_specs = [s for s in specs if isinstance(s, DerivedImageSpec)]
        # At least one malicious task should have benign_dockerfile_extra
        assert len(derived_specs) >= 1

    def test_all_specs_are_image_build_spec(self) -> None:
        """Verify all returned specs are valid ImageBuildSpec types."""
        config = SwebenchDatasetConfig(max_instances=1)
        specs = SwebenchDataset.get_image_build_specs(config)

        for spec in specs:
            # ImageBuildSpec = BuildImageSpec | MultiStageBuildImageSpec | DerivedImageSpec
            assert isinstance(spec, (BuildImageSpec, MultiStageBuildImageSpec, DerivedImageSpec))


class TestRegistryEntryPointLoadingErrors:
    """Tests for error handling when entry point loading fails."""

    @pytest.fixture(autouse=True)
    def reset_registry_state(self):
        """Reset registry state before and after each test."""
        # Save original state
        original_loaded = registry._image_buildable_classes_loaded
        original_error = registry._image_buildable_load_error
        original_classes = registry._image_buildable_classes.copy()
        original_failed = registry._image_buildable_failed_entry_points.copy()

        yield

        # Restore original state
        registry._image_buildable_classes_loaded = original_loaded
        registry._image_buildable_load_error = original_error
        registry._image_buildable_classes.clear()
        registry._image_buildable_classes.update(original_classes)
        registry._image_buildable_failed_entry_points.clear()
        registry._image_buildable_failed_entry_points.update(original_failed)

    def test_get_image_build_specs_raises_when_entry_point_loading_failed(
        self,
    ) -> None:
        """Verify RuntimeError raised when entry point loading failed completely."""
        registry._image_buildable_classes_loaded = True
        registry._image_buildable_load_error = ValueError("Metadata error")

        with pytest.raises(RuntimeError, match="entry point loading failed"):
            get_image_build_specs("any-dataset", SwebenchDatasetConfig())

    def test_get_datasets_with_image_specs_raises_when_entry_point_loading_failed(
        self,
    ) -> None:
        """Verify RuntimeError raised when entry point loading failed completely."""
        registry._image_buildable_classes_loaded = True
        registry._image_buildable_load_error = ImportError("System error")

        with pytest.raises(RuntimeError, match="entry point loading failed"):
            get_datasets_with_image_specs()

    def test_error_chaining_preserves_original_error(self) -> None:
        """Verify original error is chained as __cause__."""
        original_error = RuntimeError("Original failure")
        registry._image_buildable_classes_loaded = True
        registry._image_buildable_load_error = original_error

        with pytest.raises(RuntimeError) as exc_info:
            get_datasets_with_image_specs()

        assert exc_info.value.__cause__ is original_error

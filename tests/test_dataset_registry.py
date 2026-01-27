# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for dataset registry image building functions."""

import pytest

# Skip this entire module if swebench is not installed
pytest.importorskip("swebench")

from prompt_siren.datasets.registry import (
    get_datasets_with_image_specs,
    get_image_build_specs,
)
from prompt_siren.datasets.swebench_dataset.config import SwebenchDatasetConfig
from prompt_siren.datasets.swebench_dataset.dataset import SwebenchDataset
from prompt_siren.registry_base import UnknownComponentError
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

    def test_raises_for_nonexistent_dataset(self) -> None:
        """Verify UnknownComponentError when dataset doesn't exist."""
        with pytest.raises(UnknownComponentError):
            get_image_build_specs("nonexistent-dataset", SwebenchDatasetConfig())

    def test_raises_for_dataset_without_image_support(self) -> None:
        """Verify ValueError when dataset exists but doesn't support image building."""
        with pytest.raises(ValueError, match="does not support image building"):
            get_image_build_specs("agentdojo", SwebenchDatasetConfig())


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

    def test_derived_specs_reference_existing_base_tags(self) -> None:
        """Verify derived specs reference tags that are produced by base specs."""
        config = SwebenchDatasetConfig(max_instances=1)
        specs = SwebenchDataset.get_image_build_specs(config)

        # Collect all tags produced by non-derived specs
        base_tags: set[str] = set()
        for spec in specs:
            if isinstance(spec, MultiStageBuildImageSpec):
                base_tags.add(spec.final_tag)
            elif isinstance(spec, BuildImageSpec):
                base_tags.add(spec.tag)

        # Every derived spec's base_image_tag must reference a produced tag
        derived_specs = [s for s in specs if isinstance(s, DerivedImageSpec)]
        assert len(derived_specs) >= 1, "Expected at least one derived spec"
        for spec in derived_specs:
            assert spec.base_image_tag in base_tags, (
                f"Derived spec {spec.tag} references base_image_tag={spec.base_image_tag!r} "
                f"which is not produced by any base spec. Available tags: {base_tags}"
            )

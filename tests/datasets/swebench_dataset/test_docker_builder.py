# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench Docker build context preparation."""

import hashlib
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from prompt_siren.datasets.swebench_dataset.config import SwebenchDatasetConfig
from prompt_siren.datasets.swebench_dataset.docker_builder import prepare_build_context
from prompt_siren.datasets.swebench_dataset.image_tags import (
    get_base_image_tag,
    get_benign_image_tag,
    get_env_image_tag,
)
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.test_spec.test_spec import TestSpec


class TestPrepareBuildContext:
    """Test build context preparation."""

    @pytest.fixture
    def mock_instance(self) -> SWEbenchInstance:
        """Create a mock SWE-bench instance."""
        # Use the real instance from INSTANCE_INJECTION_MAPPING
        return cast(
            SWEbenchInstance,
            {
                "instance_id": "astropy__astropy-12907",
                "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
                "repo": "astropy/astropy",
                "version": "4.3",
                "problem_statement": "Test problem",
                "hints_text": "",
                "environment_setup_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            },
        )

    @pytest.fixture
    def mock_config(self) -> SwebenchDatasetConfig:
        """Create a mock dataset config."""
        return SwebenchDatasetConfig(
            use_cache=False,
        )

    @pytest.fixture
    def build_context_dir(self, tmp_path: Path) -> Path:
        """Create a build context directory path."""
        return tmp_path / "cache"

    @pytest.fixture
    def mock_test_spec(self) -> TestSpec:
        """Create a mock test spec."""
        spec = MagicMock(spec=TestSpec)
        spec.base_image_key = "swebench-base:latest"
        spec.env_image_key = "swebench-env:abc123"
        spec.instance_image_key = "swebench-instance:django__django-11179"
        spec.setup_env_script = "#!/bin/bash\necho 'setup env'"
        spec.install_repo_script = "#!/bin/bash\necho 'install repo'"
        spec.eval_script = "#!/bin/bash\necho 'run tests'"
        return spec

    @patch("prompt_siren.datasets.swebench_dataset.docker_builder.make_test_spec")
    def test_prepare_build_context_returns_multi_stage_spec(
        self,
        mock_make_test_spec: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_config: SwebenchDatasetConfig,
        mock_test_spec: TestSpec,
        build_context_dir: Path,
    ) -> None:
        """Test that prepare_build_context returns MultiStageBuildImageSpec."""
        mock_make_test_spec.return_value = mock_test_spec

        spec, _test_spec = prepare_build_context(mock_instance, mock_config, build_context_dir)

        # Check that spec has three stages
        assert len(spec.stages) == 3
        assert spec.final_tag == get_benign_image_tag(mock_instance["instance_id"])

        # Check stage ordering and dependencies
        base_hash = hashlib.sha256(mock_test_spec.base_image_key.encode()).hexdigest()[:16]
        env_hash = hashlib.sha256(mock_test_spec.env_image_key.encode()).hexdigest()[:16]
        expected_base_tag = get_base_image_tag(base_hash)
        expected_env_tag = get_env_image_tag(env_hash)
        expected_benign_tag = get_benign_image_tag(mock_instance["instance_id"])

        assert spec.stages[0].tag == expected_base_tag
        assert spec.stages[0].parent_tag is None
        assert spec.stages[0].cache_key == mock_test_spec.base_image_key

        assert spec.stages[1].tag == expected_env_tag
        assert spec.stages[1].parent_tag == expected_base_tag
        assert spec.stages[1].cache_key == mock_test_spec.env_image_key

        assert spec.stages[2].tag == expected_benign_tag
        assert spec.stages[2].parent_tag == expected_env_tag
        assert spec.stages[2].cache_key is None  # Instances always rebuild

    @patch("prompt_siren.datasets.swebench_dataset.docker_builder.make_test_spec")
    def test_prepare_build_context_respects_cache_flag(
        self,
        mock_make_test_spec: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_config: SwebenchDatasetConfig,
        mock_test_spec: TestSpec,
        build_context_dir: Path,
    ) -> None:
        """Test that use_cache flag prevents rewriting existing files."""
        mock_make_test_spec.return_value = mock_test_spec

        # Create config with caching enabled
        cached_config = SwebenchDatasetConfig(
            use_cache=True,
        )

        # First call creates files
        _spec1, _ = prepare_build_context(mock_instance, mock_config, build_context_dir)

        # Write custom content to Dockerfile
        base_dirs = list((build_context_dir / "base").iterdir())
        base_dockerfile = base_dirs[0] / "Dockerfile"
        custom_content = "# CUSTOM CONTENT"
        base_dockerfile.write_text(custom_content)

        # Second call with use_cache=True should not overwrite
        _spec2, _ = prepare_build_context(mock_instance, cached_config, build_context_dir)

        # Check that custom content is preserved
        assert base_dockerfile.read_text() == custom_content

    @patch("prompt_siren.datasets.swebench_dataset.docker_builder.make_test_spec")
    def test_prepare_build_context_with_injection_mapping(
        self,
        mock_make_test_spec: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_config: SwebenchDatasetConfig,
        mock_test_spec: TestSpec,
        build_context_dir: Path,
    ) -> None:
        """Test that prepare_build_context looks up and passes injection_spec."""
        mock_make_test_spec.return_value = mock_test_spec

        _spec, _ = prepare_build_context(mock_instance, mock_config, build_context_dir)

        # Verify that make_test_spec was called with the injection_spec
        mock_make_test_spec.assert_called_once()
        call_kwargs = mock_make_test_spec.call_args.kwargs
        assert "injection_spec" in call_kwargs
        # Verify it's the real spec from the mapping
        assert call_kwargs["injection_spec"]["file"] == "astropy/modeling/separable.py"
        assert call_kwargs["injection_spec"]["line"] == 246
        assert "injection_vector_ea2cbaa4" in call_kwargs["injection_spec"]["content"]

    @patch("prompt_siren.datasets.swebench_dataset.docker_builder.INSTANCE_INJECTION_MAPPING", {})
    def test_prepare_build_context_without_injection_mapping(
        self,
        mock_instance: SWEbenchInstance,
        mock_config: SwebenchDatasetConfig,
        build_context_dir: Path,
    ) -> None:
        """Test that prepare_build_context raises error when instance not in mapping."""
        # Instance not in mapping
        with pytest.raises(
            RuntimeError,
            match="does not have a location to place an injection",
        ):
            prepare_build_context(mock_instance, mock_config, build_context_dir)

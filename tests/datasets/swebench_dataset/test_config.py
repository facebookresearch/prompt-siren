# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench dataset configuration defaults."""

from prompt_siren.datasets.swebench_dataset.config import SwebenchDatasetConfig


class TestSwebenchDatasetConfigDefaults:
    """Verify default configuration values."""

    def test_default_registry_is_set(self) -> None:
        config = SwebenchDatasetConfig()
        assert config.registry == "ghcr.io/ethz-spylab/prompt-siren-images"

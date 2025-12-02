# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the registry bridge module."""

import pytest
from prompt_siren.config.experiment_config import DatasetConfig
from prompt_siren.config.registry_bridge import create_dataset_from_config
from prompt_siren.datasets import dataset_registry
from prompt_siren.datasets.agentdojo_dataset import AgentDojoDataset
from prompt_siren.environments.agentdojo_env import AgentDojoEnv
from prompt_siren.registry_base import UnknownComponentError

from ..conftest import (
    create_mock_dataset,
    MockDataset,
    MockDatasetConfig,
    MockEnvironment,
)


class TestDatasetBridge:
    """Tests for dataset-related bridge functions."""

    @pytest.fixture(autouse=True)
    def setup_registries(self):
        """Register mock components for testing."""
        # Register mock dataset
        if "mock" not in dataset_registry._registry:
            dataset_registry.register("mock", MockDatasetConfig, create_mock_dataset)

        yield

        # Clean up
        if "mock" in dataset_registry._registry:
            del dataset_registry._registry["mock"]

    def test_create_dataset_from_config_with_mock(self):
        """Test creating a dataset from config with mock dataset."""
        config = DatasetConfig(
            type="mock",
            config={"name": "test_dataset", "custom_parameter": "test_value"},
        )

        dataset = create_dataset_from_config(config)

        assert isinstance(dataset, MockDataset)
        assert dataset.name == "test_dataset"
        # Datasets now own their environment directly
        assert isinstance(dataset.environment, MockEnvironment)

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_create_dataset_from_config_with_agentdojo(self, suite_name: str):
        """Test creating a dataset from config with real AgentDojo dataset."""
        config = DatasetConfig(
            type="agentdojo",
            config={"suite_name": suite_name, "version": "v1.2.2"},
        )

        dataset = create_dataset_from_config(config)

        assert isinstance(dataset, AgentDojoDataset)
        # Dataset now directly provides environment
        assert isinstance(dataset.environment, AgentDojoEnv)
        assert len(dataset.benign_tasks) > 0
        assert len(dataset.malicious_tasks) > 0

    def test_create_dataset_from_config_invalid_type(self):
        """Test that creating dataset with invalid type raises error."""
        config = DatasetConfig(type="nonexistent_dataset", config={})

        with pytest.raises(
            UnknownComponentError,
            match="Dataset type 'nonexistent_dataset' is not registered",
        ):
            create_dataset_from_config(config)


class TestDatasetEnvironmentIntegration:
    """Integration tests for dataset and environment."""

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_full_pipeline_with_agentdojo(self, suite_name: str):
        """Test the complete pipeline from config to dataset with environment."""
        # 1. Start with a DatasetConfig
        dataset_config = DatasetConfig(
            type="agentdojo",
            config={"suite_name": suite_name, "version": "v1.2.2"},
        )

        # 2. Create dataset from config
        dataset = create_dataset_from_config(dataset_config)

        # 3. Verify dataset properties
        assert isinstance(dataset, AgentDojoDataset)
        assert len(dataset.benign_tasks) > 0
        assert len(dataset.malicious_tasks) > 0
        assert len(dataset.task_couples) > 0

        # 4. Verify dataset provides environment directly
        env = dataset.environment
        assert isinstance(env, AgentDojoEnv)
        assert env.name == f"agentdojo-{suite_name}"

        # 5. Verify dataset has tools
        assert len(dataset.default_toolsets) > 0

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_dataset_environment_is_configured_correctly(self, suite_name: str):
        """Test that dataset's environment is configured with dataset settings."""
        # Create dataset
        dataset_config = DatasetConfig(
            type="agentdojo",
            config={"suite_name": suite_name, "version": "v1.2.2"},
        )
        dataset = create_dataset_from_config(dataset_config)

        # Get environment from dataset
        env = dataset.environment

        # Verify environment was created successfully and has expected properties
        assert isinstance(env, AgentDojoEnv)
        assert env.name == f"agentdojo-{suite_name}"

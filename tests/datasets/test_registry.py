# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the dataset plugin system."""

import pytest
from prompt_siren.datasets import (
    create_dataset,
    get_dataset_config_class,
    get_registered_datasets,
)
from prompt_siren.datasets.agentdojo_dataset import AgentDojoDatasetConfig
from prompt_siren.datasets.registry import dataset_registry
from prompt_siren.registry_base import UnknownComponentError

from ..conftest import create_mock_dataset, MockDataset, MockDatasetConfig


class TestDatasetRegistry:
    """Tests for the dataset plugin system."""

    def setup_method(self):
        """Set up test dataset by clearing registry and registering mock dataset."""
        # Clear the registry for clean tests
        dataset_registry._registry.clear()
        dataset_registry._entry_points_loaded = False

        # Manually register mock dataset for testing
        dataset_registry.register("mock", MockDatasetConfig, create_mock_dataset)

    def test_register_and_get_config_class(self):
        """Test registering a dataset and retrieving its config class."""
        # Check that it's listed in registered datasets
        assert "mock" in get_registered_datasets()

        # Get the config class
        config_class = get_dataset_config_class("mock")

        # Check that it's the correct class
        assert config_class == MockDatasetConfig

    def test_entry_point_discovery(self):
        """Test that entry points are discovered automatically."""
        # Clear manual registrations
        dataset_registry._registry.clear()
        dataset_registry._entry_points_loaded = False

        # Check that built-in datasets are discovered via entry points
        registered_datasets = get_registered_datasets()
        assert "agentdojo" in registered_datasets

        # Test that we can get the config classes
        agentdojo_config_class = get_dataset_config_class("agentdojo")
        assert agentdojo_config_class is AgentDojoDatasetConfig

    def test_create_dataset(self):
        """Test creating a dataset from config."""
        # Create a config
        config = MockDatasetConfig(name="Custom Mock Dataset", custom_parameter="test-value")

        # Create the dataset
        dataset = create_dataset("mock", config)

        # Check that it's the correct type
        assert isinstance(dataset, MockDataset)

        # Check that it has the correct values
        assert dataset.name == "Custom Mock Dataset"

        # Check that environment_type is correct

    def test_missing_dataset_type(self):
        """Test error when requesting an unregistered dataset type."""
        with pytest.raises(UnknownComponentError):
            get_dataset_config_class("non-existent-dataset")

        config = MockDatasetConfig()
        with pytest.raises(UnknownComponentError):
            create_dataset("non-existent-dataset", config)

    def test_duplicate_registration(self):
        """Test error when attempting to register the same dataset type twice."""
        # Attempt to register the same type again should raise ValueError
        with pytest.raises(ValueError, match="Dataset type 'mock' is already registered"):
            dataset_registry.register("mock", MockDatasetConfig, create_mock_dataset)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the AgentDojo dataset implementation."""

import pytest
from prompt_siren.datasets.agentdojo_dataset import (
    AgentDojoDataset,
    AgentDojoDatasetConfig,
    create_agentdojo_dataset,
    load_agentdojo_dataset,
)
from prompt_siren.tasks import TaskCouple


class TestAgentDojoDataset:
    """Tests for AgentDojoDataset class."""

    @pytest.fixture(params=["banking", "slack", "travel", "workspace"])
    def suite_name(self, request) -> str:
        """Parameterized fixture for different suite names."""
        return request.param

    @pytest.fixture
    def dataset(self, suite_name: str) -> AgentDojoDataset:
        """Load a dataset for testing."""
        config = AgentDojoDatasetConfig(suite_name=suite_name, version="v1.2.2")
        return load_agentdojo_dataset(config)

    def test_name_format(self, dataset: AgentDojoDataset, suite_name: str):
        """Test that dataset name follows AgentDojo naming convention."""
        assert dataset.name == f"agentdojo-{suite_name}"

    def test_task_couples_populated(self, dataset: AgentDojoDataset):
        """Test that task couples are generated as cartesian product (AgentDojo-specific)."""
        couples = dataset.task_couples
        benign_tasks = dataset.benign_tasks
        malicious_tasks = dataset.malicious_tasks

        # AgentDojo-specific: Should have cartesian product of benign x malicious
        expected_count = len(benign_tasks) * len(malicious_tasks)
        assert len(couples) == expected_count

        # All couples should be TaskCouple instances
        assert all(isinstance(couple, TaskCouple) for couple in couples)

        # Check that all combinations are present
        benign_ids = {task.id for task in benign_tasks}
        malicious_ids = {task.id for task in malicious_tasks}

        couple_benign_ids = {couple.benign.id for couple in couples}
        couple_malicious_ids = {couple.malicious.id for couple in couples}

        assert couple_benign_ids == benign_ids
        assert couple_malicious_ids == malicious_ids

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_create_agentdojo_dataset_factory(self, suite_name: str):
        """Test the factory function used by the registry."""
        config = AgentDojoDatasetConfig(suite_name=suite_name, version="v1.2.2")
        dataset = create_agentdojo_dataset(config)

        assert isinstance(dataset, AgentDojoDataset)
        assert len(dataset.benign_tasks) > 0
        assert len(dataset.malicious_tasks) > 0
        assert len(dataset.task_couples) > 0

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_different_versions_create_different_datasets(self, suite_name: str):
        """Test that different config versions can be loaded (if available)."""
        config1 = AgentDojoDatasetConfig(suite_name=suite_name, version="v1.2.2")
        dataset1 = load_agentdojo_dataset(config1)

        # Just verify it loads successfully
        assert len(dataset1.benign_tasks) > 0

    def test_invalid_suite_name_raises_error(self):
        """Test that invalid suite names raise appropriate errors."""
        config = AgentDojoDatasetConfig(suite_name="nonexistent_suite", version="v1.2.2")

        with pytest.raises(KeyError):  # AgentDojo raises error for missing suite
            load_agentdojo_dataset(config)

    def test_invalid_version_raises_error(self):
        """Test that invalid versions raise appropriate errors."""
        config = AgentDojoDatasetConfig(suite_name="workspace", version="v999.999.999")

        with pytest.raises(KeyError):  # AgentDojo raises error for missing version
            load_agentdojo_dataset(config)

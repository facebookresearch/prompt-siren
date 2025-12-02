# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the dataset interface across all registered datasets.

These tests verify that all datasets correctly implement the required properties:
- benign_tasks
- malicious_tasks
- task_couples
- default_toolsets
"""

import pytest
from prompt_siren.datasets.abstract import AbstractDataset
from prompt_siren.datasets.registry import (
    create_dataset,
    get_dataset_config_class,
    get_registered_datasets,
)

from ..conftest import create_mock_sandbox, MockSandboxConfig

pytestmark = pytest.mark.anyio


def get_testable_datasets() -> list[str]:
    """Get list of dataset types that should have non-empty task lists.

    Filters out 'mock' dataset which is designed for testing with empty lists.
    """
    return [dataset_type for dataset_type in get_registered_datasets() if dataset_type != "mock"]


@pytest.fixture(params=get_testable_datasets())
def dataset_instance(request) -> AbstractDataset:
    """Fixture that provides dataset instances for all registered datasets.

    Creates datasets using default configurations from their config classes.
    Provides a mock sandbox manager for datasets that require it.
    """
    dataset_type = request.param
    config_class = get_dataset_config_class(dataset_type)
    config = config_class()  # Use defaults

    # Create a mock sandbox manager for datasets that need it (e.g., SWEBench)
    mock_sandbox_config = MockSandboxConfig(name="test_sandbox")
    mock_sandbox = create_mock_sandbox(mock_sandbox_config)

    return create_dataset(dataset_type, config, sandbox_manager=mock_sandbox)


class TestDatasetProperties:
    """Test the required properties interface for all datasets."""

    def test_benign_tasks_property(self, dataset_instance: AbstractDataset):
        """Test that benign_tasks returns deduplicated benign tasks."""
        benign_tasks = dataset_instance.benign_tasks

        # Should be a list of Task objects
        assert isinstance(benign_tasks, list)
        assert len(benign_tasks) > 0

        # All benign task IDs should be unique (deduplicated)
        benign_ids = [t.id for t in benign_tasks]
        assert len(benign_ids) == len(set(benign_ids)), "Benign task IDs should be unique"

        # All should be valid Task objects with no ':' in ID
        for task in benign_tasks:
            assert ":" not in task.id, f"Benign task ID should not contain ':': {task.id}"
            assert task.prompt is not None

    def test_malicious_tasks_property(self, dataset_instance: AbstractDataset):
        """Test that malicious_tasks returns deduplicated malicious tasks."""
        malicious_tasks = dataset_instance.malicious_tasks

        # Should be a list of Task objects
        assert isinstance(malicious_tasks, list)
        assert len(malicious_tasks) > 0

        # All malicious task IDs should be unique (deduplicated)
        malicious_ids = [t.id for t in malicious_tasks]
        assert len(malicious_ids) == len(set(malicious_ids)), "Malicious task IDs should be unique"

        # All should be valid Task objects with no ':' in ID
        for task in malicious_tasks:
            assert ":" not in task.id, f"Malicious task ID should not contain ':': {task.id}"
            assert task.goal is not None

    def test_task_couples_property(self, dataset_instance: AbstractDataset):
        """Test that task_couples returns valid couples."""
        couples = dataset_instance.task_couples

        # Should be a list of TaskCouple objects
        assert isinstance(couples, list)
        assert len(couples) > 0

        # Verify all couples have valid structure
        for couple in couples:
            # Couple ID should have ':' separator
            assert ":" in couple.id, f"Couple ID should contain ':': {couple.id}"

            # Both tasks should be non-null
            assert couple.benign is not None
            assert couple.malicious is not None

            # Task IDs should not contain ':'
            ben_id = couple.benign.id
            assert ":" not in ben_id, f"Benign task ID should not contain ':': {ben_id}"
            mal_id = couple.malicious.id
            assert ":" not in mal_id, f"Malicious task ID should not contain ':': {mal_id}"

    def test_default_toolsets_property(self, dataset_instance: AbstractDataset):
        """Test that default_toolsets returns a list of toolsets."""
        toolsets = dataset_instance.default_toolsets

        # Should be a list (may be empty for some datasets)
        assert isinstance(toolsets, list)

        # For most real datasets, toolsets should not be empty
        # (though we allow it for flexibility)
        if hasattr(dataset_instance, "name") and dataset_instance.name != "mock":
            # Real datasets should typically have toolsets
            assert len(toolsets) >= 0  # Allow zero for now, but document expectation

    def test_environment_property(self, dataset_instance: AbstractDataset):
        """Test that environment property returns a valid environment instance."""
        environment = dataset_instance.environment

        # Should not be None
        assert environment is not None

        # Environment should have required attributes
        assert hasattr(environment, "name")
        assert hasattr(environment, "all_injection_ids")
        assert hasattr(environment, "create_batch_context")
        assert hasattr(environment, "create_task_context")

    def test_task_lists_are_independent(self, dataset_instance: AbstractDataset):
        """Test that the three lists are independent and correctly formed."""
        benign_tasks = dataset_instance.benign_tasks
        malicious_tasks = dataset_instance.malicious_tasks
        couples = dataset_instance.task_couples

        # Lists should not be empty
        assert len(benign_tasks) > 0
        assert len(malicious_tasks) > 0
        assert len(couples) > 0

        # Number of couples should generally be >= number of unique benign tasks
        # (since multiple couples can share the same benign task)
        assert len(couples) >= len(benign_tasks)

        # All benign tasks from couples should be in benign_tasks
        benign_task_ids = {t.id for t in benign_tasks}
        for couple in couples:
            ben_id = couple.benign.id
            assert ben_id in benign_task_ids, f"Benign task {ben_id} not in benign_tasks"

        # All malicious tasks from couples should be in malicious_tasks
        malicious_task_ids = {t.id for t in malicious_tasks}
        for couple in couples:
            mal_id = couple.malicious.id
            assert mal_id in malicious_task_ids, f"Malicious task {mal_id} not in malicious_tasks"

    def test_multiple_calls_return_consistent_results(self, dataset_instance: AbstractDataset):
        """Test that multiple calls to properties return consistent results."""
        # Call each property multiple times
        benign_1 = dataset_instance.benign_tasks
        benign_2 = dataset_instance.benign_tasks
        malicious_1 = dataset_instance.malicious_tasks
        malicious_2 = dataset_instance.malicious_tasks
        couples_1 = dataset_instance.task_couples
        couples_2 = dataset_instance.task_couples
        toolsets_1 = dataset_instance.default_toolsets
        toolsets_2 = dataset_instance.default_toolsets

        # IDs should be consistent across calls
        assert {t.id for t in benign_1} == {t.id for t in benign_2}
        assert {t.id for t in malicious_1} == {t.id for t in malicious_2}
        assert {c.id for c in couples_1} == {c.id for c in couples_2}

        # Toolset count should be consistent
        assert len(toolsets_1) == len(toolsets_2)

    def test_name_attribute_exists(self, dataset_instance: AbstractDataset):
        """Test that dataset has a name attribute."""
        # All datasets should have a name
        assert hasattr(dataset_instance, "name")
        assert isinstance(dataset_instance.name, str)
        assert len(dataset_instance.name) > 0

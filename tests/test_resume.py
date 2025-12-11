# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for resume.py module."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
)
from prompt_siren.resume import (
    filter_incomplete_couples,
    filter_incomplete_tasks,
    load_saved_job_config,
    merge_configs,
    SavedJobConfig,
)
from prompt_siren.run_persistence import INDEX_FILENAME, IndexEntry

from .conftest import create_mock_benign_task, create_mock_task_couple


class TestLoadSavedJobConfig:
    """Tests for load_saved_job_config function."""

    def test_loads_valid_config(self, tmp_path: Path):
        """Test loading a valid config.yaml file."""
        config_yaml = tmp_path / "config.yaml"
        config_content = """# Config hash: abc12345
# Created: 2025-01-01T00:00:00

dataset:
  type: agentdojo
  config:
    suite_name: workspace
    version: v1.0
agent:
  type: plain
  config:
    model: claude-3-5-sonnet
    temperature: 0.0
attack:
  type: agentdojo
  config:
    attack_template: test
"""
        config_yaml.write_text(config_content)

        result = load_saved_job_config(tmp_path)

        assert isinstance(result, SavedJobConfig)
        assert result.dataset.type == "agentdojo"
        assert result.dataset.config["suite_name"] == "workspace"
        assert result.agent.type == "plain"
        assert result.agent.config["model"] == "claude-3-5-sonnet"
        assert result.attack is not None
        assert result.attack.type == "agentdojo"

    def test_loads_benign_config(self, tmp_path: Path):
        """Test loading a config without attack (benign mode)."""
        config_yaml = tmp_path / "config.yaml"
        config_content = """# Config hash: abc12345

dataset:
  type: agentdojo
  config:
    suite_name: workspace
agent:
  type: plain
  config:
    model: test
attack: null
"""
        config_yaml.write_text(config_content)

        result = load_saved_job_config(tmp_path)

        assert result.attack is None

    def test_raises_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised when config.yaml doesn't exist."""
        with pytest.raises(FileNotFoundError, match=r"No config\.yaml found"):
            load_saved_job_config(tmp_path)

    def test_raises_value_error_on_missing_dataset(self, tmp_path: Path):
        """Test that ValueError is raised when dataset key is missing."""
        config_yaml = tmp_path / "config.yaml"
        config_content = """agent:
  type: plain
  config: {}
"""
        config_yaml.write_text(config_content)

        with pytest.raises(ValueError, match="Missing 'dataset' key"):
            load_saved_job_config(tmp_path)

    def test_raises_value_error_on_missing_agent(self, tmp_path: Path):
        """Test that ValueError is raised when agent key is missing."""
        config_yaml = tmp_path / "config.yaml"
        config_content = """dataset:
  type: test
  config: {}
"""
        config_yaml.write_text(config_content)

        with pytest.raises(ValueError, match="Missing 'agent' key"):
            load_saved_job_config(tmp_path)


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_saved_config_overrides_base(self):
        """Test that saved config overrides base config."""
        base_cfg = OmegaConf.create(
            {
                "name": "experiment",
                "dataset": {"type": "base_dataset", "config": {"key": "base_value"}},
                "agent": {"type": "base_agent", "config": {"model": "base_model"}},
                "attack": None,
                "execution": {"concurrency": 4},
                "output": {"trace_dir": "traces"},
            }
        )

        saved_config = SavedJobConfig(
            dataset=DatasetConfig(type="saved_dataset", config={"key": "saved_value"}),
            agent=AgentConfig(type="saved_agent", config={"model": "saved_model"}),
            attack=AttackConfig(type="saved_attack", config={}),
        )

        result = merge_configs(base_cfg, saved_config)

        # Saved values should override base
        assert result["dataset"]["type"] == "saved_dataset"
        assert result["dataset"]["config"]["key"] == "saved_value"
        assert result["agent"]["type"] == "saved_agent"
        assert result["agent"]["config"]["model"] == "saved_model"
        assert result["attack"]["type"] == "saved_attack"

        # Base values not in saved should be preserved
        assert result["execution"]["concurrency"] == 4
        assert result["output"]["trace_dir"] == "traces"
        assert result["name"] == "experiment"

    def test_merge_preserves_base_fields(self):
        """Test that non-component fields from base are preserved."""
        base_cfg = OmegaConf.create(
            {
                "name": "my_experiment",
                "dataset": {"type": "base", "config": {}},
                "agent": {"type": "base", "config": {}},
                "attack": None,
                "execution": {"concurrency": 8},
                "telemetry": {"trace_console": True},
                "usage_limits": {"max_tokens": 1000},
            }
        )

        saved_config = SavedJobConfig(
            dataset=DatasetConfig(type="saved", config={}),
            agent=AgentConfig(type="saved", config={}),
            attack=None,
        )

        result = merge_configs(base_cfg, saved_config)

        # Non-component fields should be preserved
        assert result["name"] == "my_experiment"
        assert result["execution"]["concurrency"] == 8
        assert result["telemetry"]["trace_console"] is True
        assert result["usage_limits"]["max_tokens"] == 1000

    def test_merge_none_attack(self):
        """Test merging with None attack (benign mode)."""
        base_cfg = OmegaConf.create(
            {
                "dataset": {"type": "base", "config": {}},
                "agent": {"type": "base", "config": {}},
                "attack": {"type": "base_attack", "config": {}},
            }
        )

        saved_config = SavedJobConfig(
            dataset=DatasetConfig(type="saved", config={}),
            agent=AgentConfig(type="saved", config={}),
            attack=None,
        )

        result = merge_configs(base_cfg, saved_config)

        # Attack should be None from saved config
        assert result["attack"] is None


class TestFilterIncompleteTasks:
    """Tests for filter_incomplete_tasks function."""

    def test_filters_completed_tasks(self, tmp_path: Path):
        """Test that completed tasks are filtered out."""
        # Create index with some completed tasks
        index_file = tmp_path / INDEX_FILENAME
        entries = [
            IndexEntry(
                execution_id="exec1",
                task_id="task_1",
                timestamp="2025-01-01T00:00:00",
                dataset="test",
                dataset_config={},
                agent_type="plain",
                agent_name="test",
                attack_type=None,
                config_hash="test_hash",
                benign_score=1.0,
                attack_score=None,
                path=Path("test1.json"),
            ),
            IndexEntry(
                execution_id="exec2",
                task_id="task_3",
                timestamp="2025-01-01T00:00:00",
                dataset="test",
                dataset_config={},
                agent_type="plain",
                agent_name="test",
                attack_type=None,
                config_hash="test_hash",
                benign_score=1.0,
                attack_score=None,
                path=Path("test3.json"),
            ),
        ]
        with open(index_file, "w") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")

        # Create tasks
        tasks = [
            create_mock_benign_task("task_1", {}),
            create_mock_benign_task("task_2", {}),
            create_mock_benign_task("task_3", {}),
            create_mock_benign_task("task_4", {}),
        ]

        # Filter incomplete
        incomplete = filter_incomplete_tasks(tasks, tmp_path, "test_hash")

        # Only task_2 and task_4 should remain
        assert len(incomplete) == 2
        assert {t.id for t in incomplete} == {"task_2", "task_4"}

    def test_returns_all_when_no_completed(self, tmp_path: Path):
        """Test that all tasks are returned when none are completed."""
        tasks = [
            create_mock_benign_task("task_1", {}),
            create_mock_benign_task("task_2", {}),
        ]

        incomplete = filter_incomplete_tasks(tasks, tmp_path, "test_hash")

        assert len(incomplete) == 2

    def test_filters_by_config_hash(self, tmp_path: Path):
        """Test that filtering respects config hash."""
        # Create index with different config hashes
        index_file = tmp_path / INDEX_FILENAME
        entries = [
            IndexEntry(
                execution_id="exec1",
                task_id="task_1",
                timestamp="2025-01-01T00:00:00",
                dataset="test",
                dataset_config={},
                agent_type="plain",
                agent_name="test",
                attack_type=None,
                config_hash="hash_a",
                benign_score=1.0,
                attack_score=None,
                path=Path("test1.json"),
            ),
            IndexEntry(
                execution_id="exec2",
                task_id="task_1",
                timestamp="2025-01-01T00:00:00",
                dataset="test",
                dataset_config={},
                agent_type="plain",
                agent_name="test",
                attack_type=None,
                config_hash="hash_b",
                benign_score=1.0,
                attack_score=None,
                path=Path("test1b.json"),
            ),
        ]
        with open(index_file, "w") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")

        tasks = [create_mock_benign_task("task_1", {})]

        # With hash_a, task_1 should be filtered
        incomplete_a = filter_incomplete_tasks(tasks, tmp_path, "hash_a")
        assert len(incomplete_a) == 0

        # With hash_c, task_1 should remain (no entries for this hash)
        incomplete_c = filter_incomplete_tasks(tasks, tmp_path, "hash_c")
        assert len(incomplete_c) == 1


class TestFilterIncompleteCouples:
    """Tests for filter_incomplete_couples function."""

    def test_filters_completed_couples(self, tmp_path: Path):
        """Test that completed couples are filtered out."""
        # Create couples first to get their IDs
        couples = [
            create_mock_task_couple("couple_1", {}, {}),
            create_mock_task_couple("couple_2", {}, {}),
        ]

        # Create index with the first couple as completed
        # Couple ID format is {benign.id}:{malicious.id}
        index_file = tmp_path / INDEX_FILENAME
        entries = [
            IndexEntry(
                execution_id="exec1",
                task_id=couples[0].id,  # Use actual couple ID
                timestamp="2025-01-01T00:00:00",
                dataset="test",
                dataset_config={},
                agent_type="plain",
                agent_name="test",
                attack_type="test",
                config_hash="test_hash",
                benign_score=1.0,
                attack_score=0.5,
                path=Path("test1.json"),
            ),
        ]
        with open(index_file, "w") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")

        # Filter incomplete
        incomplete = filter_incomplete_couples(couples, tmp_path, "test_hash")

        # Only couple_2 should remain
        assert len(incomplete) == 1
        assert incomplete[0].id == couples[1].id

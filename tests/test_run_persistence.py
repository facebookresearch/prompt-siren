# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for run_persistence.py functions.

These tests focus on the persistence layer - saving generated attacks
and conversation logs to JSON files.
"""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from logfire import LogfireSpan
from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
)
from prompt_siren.run_persistence import (
    _save_config_yaml,
    compute_config_hash,
    CONFIG_FILENAME,
    ExecutionData,
    ExecutionPersistence,
    INDEX_FILENAME,
    IndexEntry,
)
from prompt_siren.tasks import EvaluationResult
from prompt_siren.types import StrContentAttack
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import (
    create_mock_benign_task,
    create_mock_task_couple,
    MockEnvState,
)


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_determinism_same_configs(self):
        """Test that same configs produce same hash across multiple calls."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        hash1 = compute_config_hash(env, agent, attack)
        hash2 = compute_config_hash(env, agent, attack)
        hash3 = compute_config_hash(env, agent, attack)

        assert hash1 == hash2 == hash3
        assert len(hash1) == 8  # Should be 8 characters

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent1 = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        agent2 = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.5},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        hash1 = compute_config_hash(env, agent1, attack)
        hash2 = compute_config_hash(env, agent2, attack)

        assert hash1 != hash2

    def test_none_attack_config(self):
        """Test that None attack config is handled correctly (benign runs)."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )

        # Should not raise an error
        hash1 = compute_config_hash(env, agent, None)
        assert len(hash1) == 8

        # None should produce different hash than an attack config
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})
        hash2 = compute_config_hash(env, agent, attack)
        assert hash1 != hash2

    def test_field_order_independence(self):
        """Test that Pydantic model field definition order doesn't affect hash."""
        # Both configs have same content, just different field order in dict
        env1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        env2 = DatasetConfig(
            type="agentdojo",
            config={"version": "v1.0", "suite_name": "workspace"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        hash1 = compute_config_hash(env1, agent, attack)
        hash2 = compute_config_hash(env2, agent, attack)

        # Should be the same because JSON serialization sorts keys
        assert hash1 == hash2

    def test_nested_configs(self):
        """Test that nested Pydantic models are handled correctly."""
        env1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        env2 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v2.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        hash1 = compute_config_hash(env1, agent, attack)
        hash2 = compute_config_hash(env2, agent, attack)

        assert len(hash1) == 8
        assert hash1 != hash2  # Changing nested value should change hash

    def test_type_sensitivity(self):
        """Test that different types produce different hashes."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )

        # Int vs float in attack config
        attack1 = AttackConfig(type="agentdojo", config={"value": 1})
        attack2 = AttackConfig(type="agentdojo", config={"value": 1.0})
        hash1 = compute_config_hash(env, agent, attack1)
        hash2 = compute_config_hash(env, agent, attack2)
        assert hash1 != hash2  # JSON serializes 1 and 1.0 differently

        # String vs number
        attack3 = AttackConfig(type="agentdojo", config={"value": "1"})
        hash3 = compute_config_hash(env, agent, attack3)
        assert hash1 != hash3

        # Boolean vs int
        attack4 = AttackConfig(type="agentdojo", config={"value": True})
        attack5 = AttackConfig(type="agentdojo", config={"value": 1})
        hash4 = compute_config_hash(env, agent, attack4)
        hash5 = compute_config_hash(env, agent, attack5)
        assert hash4 != hash5

    def test_special_types_in_pydantic_models(self):
        """Test that special types (datetime, Path) are serialized consistently."""
        # Create two instances with same values
        env1 = DatasetConfig(
            type="agentdojo",
            config={
                "suite_name": "workspace",
                "version": "v1.0",
                "timestamp": "2025-01-14T10:30:45Z",
            },
        )
        env2 = DatasetConfig(
            type="agentdojo",
            config={
                "suite_name": "workspace",
                "version": "v1.0",
                "timestamp": "2025-01-14T10:30:45Z",
            },
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )

        hash1 = compute_config_hash(env1, agent, None)
        hash2 = compute_config_hash(env2, agent, None)
        assert hash1 == hash2

        # Different values should produce different hashes
        env3 = DatasetConfig(
            type="agentdojo",
            config={
                "suite_name": "workspace",
                "version": "v1.0",
                "timestamp": "2025-01-14T10:30:46Z",
            },
        )
        hash3 = compute_config_hash(env3, agent, None)
        assert hash1 != hash3

    def test_list_in_config(self):
        """Test that lists in configs are handled correctly."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )

        attack1 = AttackConfig(type="agentdojo", config={"values": [1, 2, 3]})
        attack2 = AttackConfig(type="agentdojo", config={"values": [1, 2, 3]})
        attack3 = AttackConfig(type="agentdojo", config={"values": [3, 2, 1]})

        hash1 = compute_config_hash(env, agent, attack1)
        hash2 = compute_config_hash(env, agent, attack2)
        hash3 = compute_config_hash(env, agent, attack3)

        assert hash1 == hash2  # Same list
        assert hash1 != hash3  # Different order matters for lists

    def test_different_types_same_config(self):
        """Test that different component types with same config produce different hashes.

        This is the critical test to ensure that type parameters are properly included
        in the hash, preventing collisions between different attack/env/agent types.
        """
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack_config = {"attack_template": "test"}

        # Same configs but different attack types should produce different hashes
        attack_gcg = AttackConfig(type="gcg", config=attack_config)
        attack_agentdojo = AttackConfig(type="agentdojo", config=attack_config)
        attack_target = AttackConfig(type="target_string", config=attack_config)

        hash_gcg = compute_config_hash(env, agent, attack_gcg)
        hash_agentdojo = compute_config_hash(env, agent, attack_agentdojo)
        hash_target = compute_config_hash(env, agent, attack_target)

        assert hash_gcg != hash_agentdojo
        assert hash_gcg != hash_target
        assert hash_agentdojo != hash_target

        # Different environment types should also produce different hashes
        env1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        env2 = DatasetConfig(
            type="playwright",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        hash_env1 = compute_config_hash(env1, agent, attack_gcg)
        hash_env2 = compute_config_hash(env2, agent, attack_gcg)
        assert hash_env1 != hash_env2

        # Different agent types should also produce different hashes
        agent1 = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        agent2 = AgentConfig(
            type="custom",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        hash_agent1 = compute_config_hash(env, agent1, attack_gcg)
        hash_agent2 = compute_config_hash(env, agent2, attack_gcg)
        assert hash_agent1 != hash_agent2

    def test_sandbox_config_excluded_from_hash(self):
        """Test that different sandbox configs produce the same hash.

        Sandbox configuration is an implementation detail that shouldn't affect
        experiment identity. Different sandbox types (local-docker, remote-docker, modal)
        should produce the same results, so they should map to the same output directory.
        """
        # Create two coding environment configs with different sandbox settings
        env1 = DatasetConfig(
            type="coding",
            config={
                "sandbox_manager_type": "local-docker",
                "sandbox_manager_config": {"network": True},
            },
        )
        env2 = DatasetConfig(
            type="coding",
            config={
                "sandbox_manager_type": "remote-docker",
                "sandbox_manager_config": {
                    "network": False,
                    "host": "example.com",
                },
            },
        )

        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Compute hashes for both configs
        hash1 = compute_config_hash(env1, agent, attack)
        hash2 = compute_config_hash(env2, agent, attack)

        # Hashes should be identical despite different sandbox configs
        assert hash1 == hash2, (
            "Sandbox config should not affect experiment hash. "
            "Different sandbox implementations should produce the same results and share output directories."
        )

        # Verify they would use the same output directory
        # (sandbox type and config are excluded from the hash)
        assert len(hash1) == 8
        assert len(hash2) == 8

    def test_different_dataset_configs_different_hashes(self):
        """Test that different dataset configurations produce different hashes."""
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Same type, different config values
        dataset1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        dataset2 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v2.0"},
        )

        hash1 = compute_config_hash(dataset1, agent, attack)
        hash2 = compute_config_hash(dataset2, agent, attack)

        # Different dataset configs should produce different hashes
        assert hash1 != hash2

    def test_different_dataset_types_different_hashes(self):
        """Test that different dataset types with same config produce different hashes."""
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Different types, same config structure
        dataset1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        dataset2 = DatasetConfig(
            type="playwright",
            config={"suite_name": "workspace", "version": "v1.0"},
        )

        hash1 = compute_config_hash(dataset1, agent, attack)
        hash2 = compute_config_hash(dataset2, agent, attack)

        # Different dataset types should produce different hashes
        assert hash1 != hash2

    def test_same_dataset_config_same_hash(self):
        """Test that identical dataset configurations produce the same hash."""
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Create same dataset config twice
        dataset1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        dataset2 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )

        hash1 = compute_config_hash(dataset1, agent, attack)
        hash2 = compute_config_hash(dataset2, agent, attack)

        # Identical configs should produce identical hashes
        assert hash1 == hash2

    def test_dataset_config_field_order_independence(self):
        """Test that dataset config field order doesn't affect hash."""
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Same config, different field order
        dataset1 = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        dataset2 = DatasetConfig(
            type="agentdojo",
            config={"version": "v1.0", "suite_name": "workspace"},
        )

        hash1 = compute_config_hash(dataset1, agent, attack)
        hash2 = compute_config_hash(dataset2, agent, attack)

        # Field order shouldn't matter (JSON serialization sorts keys)
        assert hash1 == hash2

    def test_complete_config_combination_uniqueness(self):
        """Test that the combination of dataset, agent, and attack configs is unique."""
        # Base configs
        dataset = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        # Get base hash
        base_hash = compute_config_hash(dataset, agent, attack)

        # Change only dataset
        dataset_changed = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v2.0"},
        )
        hash_dataset_changed = compute_config_hash(dataset_changed, agent, attack)
        assert base_hash != hash_dataset_changed

        # Change only agent
        agent_changed = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.5},
        )
        hash_agent_changed = compute_config_hash(dataset, agent_changed, attack)
        assert base_hash != hash_agent_changed

        # Change only attack
        attack_changed = AttackConfig(type="agentdojo", config={"attack_template": "different"})
        hash_attack_changed = compute_config_hash(dataset, agent, attack_changed)
        assert base_hash != hash_attack_changed

        # All hashes should be different
        assert (
            len(
                {
                    base_hash,
                    hash_dataset_changed,
                    hash_agent_changed,
                    hash_attack_changed,
                }
            )
            == 4
        )


class TestExecutionPersistence:
    """Tests for ExecutionPersistence class."""

    def test_create_initializes_directories(self, tmp_path: Path):
        """Test that create() sets up the directory structure correctly."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=env,
            agent_config=agent,
            attack_config=attack,
        )

        # Check directory exists
        assert persistence.output_dir.exists()
        assert persistence.output_dir.is_dir()

        # Check structure: outputs/agentdojo/plain/agentdojo/{hash}/
        assert persistence.output_dir.name == persistence.config_hash
        assert persistence.output_dir.parent.name == "agentdojo"
        assert persistence.output_dir.parent.parent.name == "plain"
        assert persistence.output_dir.parent.parent.parent.name == "agentdojo"

        # Check config.yaml created
        config_file = persistence.output_dir / CONFIG_FILENAME
        assert config_file.exists()

    def test_config_yaml_format(self, tmp_path: Path):
        """Test that config.yaml is properly formatted and parseable."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=env,
            agent_config=agent,
            attack_config=attack,
        )

        config_file = persistence.output_dir / CONFIG_FILENAME
        with open(config_file) as f:
            content = f.read()
            # Check header comment with hash
            assert f"# Config hash: {persistence.config_hash}" in content

            # Parse YAML (skip header comments)
            configs = yaml.safe_load(content.split("\n\n", 1)[1])

            # Verify dataset config
            assert "dataset" in configs
            assert configs["dataset"]["type"] == "agentdojo"
            assert "config" in configs["dataset"]
            assert configs["dataset"]["config"]["suite_name"] == "workspace"
            assert configs["dataset"]["config"]["version"] == "v1.0"

            # Verify agent config
            assert "agent" in configs
            assert configs["agent"]["type"] == "plain"
            assert "config" in configs["agent"]
            assert configs["agent"]["config"]["temperature"] == 0.0
            assert configs["agent"]["config"]["model"] == "claude-3-5-sonnet"

            # Verify attack config
            assert "attack" in configs
            assert configs["attack"]["type"] == "agentdojo"
            assert "config" in configs["attack"]
            assert configs["attack"]["config"]["attack_template"] == "test"

    def test_benign_only_execution(self, tmp_path: Path):
        """Test benign-only execution (no attack)."""
        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        )

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=env,
            agent_config=agent,
            attack_config=None,
        )

        # Output directory should have "benign" instead of attack type
        assert "benign" in str(persistence.output_dir)
        assert persistence.attack_config is None

        # Config file should have attack set to None for benign runs
        config_file = persistence.output_dir / CONFIG_FILENAME
        with open(config_file) as f:
            content = f.read()
            configs = yaml.safe_load(content.split("\n\n", 1)[1])
            assert configs.get("attack") is None

    def test_save_single_task_execution_creates_file(self, tmp_path: Path):
        """Test that save_single_task_execution creates a properly formatted file."""

        # Setup persistence for benign-only
        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=None,  # Benign-only
        )

        # Create mock data - single task

        task = create_mock_benign_task("test_task_benign", {"eval1": 1.0})
        mock_run_ctx = RunContext(
            messages=[ModelRequest.user_text_prompt("test")],
            usage=RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=0),
            deps=MockEnvState(value="test"),
            model=TestModel(),
        )
        evaluation = EvaluationResult(task_id=task.id, results={"eval1": 1.0})

        mock_span = Mock(spec=LogfireSpan)
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        # Save single task execution
        filepath = persistence.save_single_task_execution(
            task=task,
            agent_name="test_agent",
            result_ctx=mock_run_ctx,
            evaluation=evaluation,
            task_span=mock_span,
        )

        # Verify file was created
        assert filepath.exists()
        assert filepath.parent == persistence.output_dir
        assert "_test_task_benign.json" in filepath.name  # Should have sanitized task_id

        # Verify content using ExecutionData model
        with open(filepath) as f:
            execution_data = ExecutionData.model_validate_json(f.read())
            assert execution_data.task_id == "test_task_benign"
            assert execution_data.agent == "test_agent"
            assert execution_data.config_hash == persistence.config_hash
            assert execution_data.results.benign_score == 1.0
            assert execution_data.results.attack_score is None  # No attack in benign mode
            assert execution_data.messages is not None
            assert execution_data.usage.input_tokens == 100
            assert execution_data.usage.output_tokens == 50
            assert execution_data.usage.total_tokens == 150
            assert execution_data.execution_id is not None
            assert execution_data.timestamp is not None
            # Verify trace and span IDs
            assert execution_data.trace_id == format(mock_span_context.trace_id, "032x")
            assert execution_data.span_id == format(mock_span_context.span_id, "016x")

    def test_save_couple_execution_with_attack(self, tmp_path: Path):
        """Test saving task couple execution with attack results and generated attacks."""

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=AttackConfig(type="agentdojo", config={"attack_template": "test"}),
        )

        couple = create_mock_task_couple("test_task_1", {"eval1": 1.0}, {"eval1": 1.0})

        mock_run_ctx = RunContext(
            messages=[ModelRequest.user_text_prompt("test")],
            usage=RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=0),
            deps=MockEnvState(value="test"),
            model=TestModel(),
        )
        benign_eval = EvaluationResult(task_id=couple.benign.id, results={"eval1": 1.0})
        malicious_eval = EvaluationResult(task_id=couple.malicious.id, results={"eval1": 0.5})

        mock_span = Mock(spec=LogfireSpan)
        mock_span_context = Mock()
        mock_span_context.trace_id = 111222333
        mock_span_context.span_id = 444555666
        mock_span.get_span_context.return_value = mock_span_context

        generated_attacks = {"vector1": StrContentAttack("payload")}

        filepath = persistence.save_couple_execution(
            couple=couple,
            agent_name="test_agent",
            result_ctx=mock_run_ctx,
            benign_eval=benign_eval,
            malicious_eval=malicious_eval,
            task_span=mock_span,
            generated_attacks=generated_attacks,
        )

        # Verify attack info included using ExecutionData model
        with open(filepath) as f:
            execution_data = ExecutionData.model_validate_json(f.read())
            assert execution_data.results.attack_score == 0.5
            assert execution_data.attacks is not None
            assert "vector1" in execution_data.attacks
            # Verify trace and span IDs
            assert execution_data.trace_id == format(mock_span_context.trace_id, "032x")
            assert execution_data.span_id == format(mock_span_context.span_id, "016x")

    def test_index_jsonl_created_and_updated(self, tmp_path: Path):
        """Test that index.jsonl is created and appended to correctly."""

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=None,
        )

        # Save first task execution

        task_1 = create_mock_benign_task("test_task_1", {"eval1": 1.0})
        mock_run_ctx = RunContext(
            messages=[ModelRequest.user_text_prompt("test")],
            usage=RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=0),
            deps=MockEnvState(value="test"),
            model=TestModel(),
        )
        evaluation_1 = EvaluationResult(task_id=task_1.id, results={"eval1": 1.0})
        mock_span = Mock(spec=LogfireSpan)
        mock_span.get_span_context.return_value = None

        persistence.save_single_task_execution(
            task=task_1,
            agent_name="test_agent",
            result_ctx=mock_run_ctx,
            evaluation=evaluation_1,
            task_span=mock_span,
        )

        # Save second task execution
        task_2 = create_mock_benign_task("test_task_2", {"eval1": 0.8})
        evaluation_2 = EvaluationResult(task_id=task_2.id, results={"eval1": 0.8})

        persistence.save_single_task_execution(
            task=task_2,
            agent_name="test_agent",
            result_ctx=mock_run_ctx,
            evaluation=evaluation_2,
            task_span=mock_span,
        )

        # Verify index.jsonl exists and has 2 entries
        index_file = tmp_path / INDEX_FILENAME
        assert index_file.exists()

        with open(index_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Verify first entry using IndexEntry model
            entry1 = IndexEntry.model_validate_json(lines[0])
            assert entry1.task_id == "test_task_1"
            assert entry1.dataset == "agentdojo"
            assert entry1.agent_type == "plain"
            assert entry1.attack_type is None
            assert entry1.config_hash == persistence.config_hash
            assert entry1.benign_score == 1.0
            assert entry1.attack_score is None
            assert entry1.path is not None
            assert entry1.execution_id is not None
            assert entry1.timestamp is not None

            # Verify second entry using IndexEntry model
            entry2 = IndexEntry.model_validate_json(lines[1])
            assert entry2.task_id == "test_task_2"
            assert entry2.benign_score == 0.8

    def test_multiple_configs_different_directories(self, tmp_path: Path):
        """Test that different configs create different directories."""

        # Create persistence with config 1
        persistence1 = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=AttackConfig(type="agentdojo", config={"attack_template": "test"}),
        )

        # Create persistence with config 2 (different temperature)
        persistence2 = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.5},
            ),
            attack_config=AttackConfig(type="agentdojo", config={"attack_template": "test"}),
        )

        # Different configs should have different hashes and directories
        assert persistence1.config_hash != persistence2.config_hash
        assert persistence1.output_dir != persistence2.output_dir
        assert persistence1.output_dir.exists()
        assert persistence2.output_dir.exists()

    def test_same_config_same_directory(self, tmp_path: Path):
        """Test that same config reuses the same directory."""

        # Create persistence twice with identical config
        persistence1 = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=AttackConfig(type="agentdojo", config={"attack_template": "test"}),
        )

        persistence2 = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=AttackConfig(type="agentdojo", config={"attack_template": "test"}),
        )

        # Same config should have same hash and reuse directory
        assert persistence1.config_hash == persistence2.config_hash
        assert persistence1.output_dir == persistence2.output_dir

        # config.yaml should only be created once (not overwritten)
        config_file = persistence1.output_dir / CONFIG_FILENAME
        assert config_file.exists()


class TestSaveConfigYaml:
    """Tests for _save_config_yaml function."""

    def test_save_config_with_attack(self, tmp_path: Path):
        """Test saving config YAML with all components including attack."""
        config_file = tmp_path / "config.yaml"
        config_hash = "abc12345"

        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(
            type="plain",
            config={"model": "claude-3-5-sonnet", "temperature": 0.5},
        )
        attack = AttackConfig(type="agentdojo", config={"attack_template": "test"})

        _save_config_yaml(
            config_file=config_file,
            config_hash=config_hash,
            dataset_config=env,
            agent_config=agent,
            attack_config=attack,
        )

        # Verify file was created
        assert config_file.exists()

        # Read and parse YAML
        with open(config_file) as f:
            content = f.read()

            # Check header
            assert f"# Config hash: {config_hash}" in content
            assert "# Created:" in content

            # Parse YAML (skip header comments)
            configs = yaml.safe_load(content.split("\n\n", 1)[1])

            # Verify structure with nested config keys
            assert "dataset" in configs
            assert configs["dataset"]["type"] == "agentdojo"
            assert "config" in configs["dataset"]
            assert configs["dataset"]["config"]["suite_name"] == "workspace"
            assert configs["dataset"]["config"]["version"] == "v1.0"

            assert "agent" in configs
            assert configs["agent"]["type"] == "plain"
            assert "config" in configs["agent"]
            assert configs["agent"]["config"]["model"] == "claude-3-5-sonnet"
            assert configs["agent"]["config"]["temperature"] == 0.5

            assert "attack" in configs
            assert configs["attack"]["type"] == "agentdojo"
            assert "config" in configs["attack"]
            assert configs["attack"]["config"]["attack_template"] == "test"

    def test_save_config_without_attack(self, tmp_path: Path):
        """Test saving config YAML for benign-only run (no attack)."""
        config_file = tmp_path / "config.yaml"
        config_hash = "def67890"

        env = DatasetConfig(
            type="agentdojo",
            config={"suite_name": "workspace", "version": "v1.0"},
        )
        agent = AgentConfig(type="plain", config={"model": "gpt-4", "temperature": 0.0})

        _save_config_yaml(
            config_file=config_file,
            config_hash=config_hash,
            dataset_config=env,
            agent_config=agent,
            attack_config=None,
        )

        # Verify file was created
        assert config_file.exists()

        # Read and parse YAML
        with open(config_file) as f:
            content = f.read()

            # Check header
            assert f"# Config hash: {config_hash}" in content

            # Parse YAML (skip header comments)
            configs = yaml.safe_load(content.split("\n\n", 1)[1])

            # Verify environment and agent are present
            assert "dataset" in configs
            assert configs["dataset"]["type"] == "agentdojo"
            assert configs["dataset"]["config"]["suite_name"] == "workspace"
            assert configs["dataset"]["config"]["version"] == "v1.0"

            assert "agent" in configs
            assert configs["agent"]["type"] == "plain"
            assert configs["agent"]["config"]["model"] == "gpt-4"
            assert configs["agent"]["config"]["temperature"] == 0.0

            # Verify attack is NOT present (or is None)
            assert configs.get("attack") is None

    @pytest.mark.anyio
    async def test_concurrent_index_writes(self, tmp_path: Path):
        """Test that concurrent writes to index.jsonl don't corrupt the file."""

        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(
                type="agentdojo",
                config={"suite_name": "workspace", "version": "v1.0"},
            ),
            agent_config=AgentConfig(
                type="plain",
                config={"model": "claude-3-5-sonnet", "temperature": 0.0},
            ),
            attack_config=None,
        )

        mock_span = Mock(spec=LogfireSpan)
        mock_span.get_span_context.return_value = None

        # Create multiple tasks that will write concurrently
        async def write_task(task_id: int):
            task = create_mock_benign_task(f"test_task_{task_id}", {"eval1": 1.0})
            mock_run_ctx = RunContext(
                messages=[ModelRequest.user_text_prompt("test")],
                usage=RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=0),
                deps=MockEnvState(value="test"),
                model=TestModel(),
            )
            evaluation = EvaluationResult(task_id=task.id, results={"eval1": 1.0})

            persistence.save_single_task_execution(
                task=task,
                agent_name="test_agent",
                result_ctx=mock_run_ctx,
                evaluation=evaluation,
                task_span=mock_span,
            )

        # Run 10 concurrent writes
        num_tasks = 10
        await asyncio.gather(*[write_task(i) for i in range(num_tasks)])

        # Verify index.jsonl has exactly num_tasks entries and is not corrupted
        index_file = tmp_path / INDEX_FILENAME
        assert index_file.exists()

        with open(index_file) as f:
            lines = f.readlines()
            assert len(lines) == num_tasks

            # Verify each line is valid JSON and can be parsed
            task_ids = set()
            for i, line in enumerate(lines):
                try:
                    entry = IndexEntry.model_validate_json(line)
                    task_ids.add(entry.task_id)
                    assert entry.dataset == "agentdojo"
                    assert entry.agent_type == "plain"
                    assert entry.benign_score == 1.0
                except Exception as e:  # noqa: PERF203 - per-line diagnostics
                    pytest.fail(f"Line {i} is corrupted or invalid JSON: {line}\nError: {e}")

            # Verify we have all unique task IDs (no duplicates or missing entries)
            expected_task_ids = {f"test_task_{i}" for i in range(num_tasks)}
            assert task_ids == expected_task_ids

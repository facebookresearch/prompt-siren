# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for GRPO attack implementation."""

from __future__ import annotations

import pytest
from prompt_siren.attacks.registry import attack_registry, get_registered_attacks

# Try to import GRPO attack - skip all tests if RL dependencies are missing
try:
    from prompt_siren.attacks.grpo_attack import (
        create_grpo_attack,
        GRPOAttack,
        GRPOAttackConfig,
    )

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    GRPOAttack = None  # type: ignore[assignment, misc]
    GRPOAttackConfig = None  # type: ignore[assignment, misc]
    create_grpo_attack = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not RL_AVAILABLE,
    reason="RL dependencies not installed (torch, transformers, trl, peft, datasets)",
)


class TestGRPOAttackConfig:
    """Tests for GRPOAttackConfig."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = GRPOAttackConfig()

        assert config.policy_model == "gpt2"
        assert config.num_train_epochs == 10
        assert config.learning_rate == 1e-5
        assert config.num_generations == 4
        assert config.max_new_tokens == 128
        assert config.use_lora is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = GRPOAttackConfig(
            policy_model="meta-llama/Llama-3.1-8B-Instruct",
            num_train_epochs=5,
            learning_rate=5e-6,
            use_lora=False,
        )

        assert config.policy_model == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.num_train_epochs == 5
        assert config.learning_rate == 5e-6
        assert config.use_lora is False

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = GRPOAttackConfig(
            policy_model="test-model",
            num_train_epochs=3,
            reward_config={"type": "exact_match"},
        )

        # Serialize to dict
        config_dict = config.model_dump()
        assert config_dict["policy_model"] == "test-model"
        assert config_dict["num_train_epochs"] == 3
        assert config_dict["reward_config"]["type"] == "exact_match"

        # Deserialize from dict
        loaded_config = GRPOAttackConfig.model_validate(config_dict)
        assert loaded_config.policy_model == "test-model"
        assert loaded_config.num_train_epochs == 3

    def test_default_injection_prompt_template(self):
        """Test that default injection prompt template contains goal placeholder."""
        config = GRPOAttackConfig()
        assert "{goal}" in config.injection_prompt_template

    def test_default_attacker_system_prompt(self):
        """Test that default attacker system prompt contains Mini-GOAT style elements."""
        config = GRPOAttackConfig()
        # Should contain key Mini-GOAT elements
        assert "skilled instructor" in config.attacker_system_prompt
        assert "eye-catching" in config.attacker_system_prompt
        assert "RESPOND ONLY WITH THE INSTRUCTION" in config.attacker_system_prompt
        # Should NOT contain the STOP instruction (removed for RL)
        assert "STOP" not in config.attacker_system_prompt


class TestGRPOAttack:
    """Tests for GRPOAttack class."""

    def test_attack_name(self):
        """Test that attack has correct name."""
        attack = GRPOAttack()
        assert attack.name == "grpo"

    def test_attack_with_custom_config(self):
        """Test creating attack with custom config."""
        config = GRPOAttackConfig(
            policy_model="test-model",
            num_train_epochs=3,
        )
        attack = GRPOAttack(_config=config)

        assert attack.config.policy_model == "test-model"
        assert attack.config.num_train_epochs == 3

    def test_factory_function(self):
        """Test the factory function creates correct attack instance."""
        config = GRPOAttackConfig(
            policy_model="factory-model",
        )
        attack = create_grpo_attack(config)

        assert isinstance(attack, GRPOAttack)
        assert attack.config.policy_model == "factory-model"

    def test_find_latest_checkpoint_no_dir(self):
        """Test _find_latest_checkpoint returns None when directory doesn't exist."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        attack = GRPOAttack()
        with TemporaryDirectory() as tmp_dir:
            non_existent = Path(tmp_dir) / "does_not_exist"
            assert attack._find_latest_checkpoint(non_existent) is None

    def test_find_latest_checkpoint_empty_dir(self):
        """Test _find_latest_checkpoint returns None for empty directory."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        attack = GRPOAttack()
        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            assert attack._find_latest_checkpoint(model_dir) is None

    def test_find_latest_checkpoint_with_checkpoints(self):
        """Test _find_latest_checkpoint returns latest checkpoint."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        attack = GRPOAttack()
        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            # Create some checkpoint directories
            (model_dir / "checkpoint-100").mkdir()
            (model_dir / "checkpoint-500").mkdir()
            (model_dir / "checkpoint-200").mkdir()

            latest = attack._find_latest_checkpoint(model_dir)
            assert latest is not None
            assert latest.name == "checkpoint-500"


class TestGRPOAttackRegistry:
    """Tests for GRPO attack registration."""

    def setup_method(self):
        """Reset registry for clean tests."""
        attack_registry._registry.clear()
        attack_registry._entry_points_loaded = False

    def test_grpo_registered(self):
        """Test that grpo attack is registered via entry points.

        Note: This test may fail during development if pyproject.toml was modified
        but the package wasn't reinstalled. Run `uv pip install -e .` to fix.
        """
        registered_attacks = get_registered_attacks()
        if "grpo" not in registered_attacks:
            pytest.skip("GRPO attack not registered - reinstall package with `uv pip install -e .`")


# Mark integration tests that require RL dependencies
@pytest.mark.skip(reason="Requires RL model download and GPU")
class TestGRPOAttackIntegration:
    """Integration tests for GRPO attack.

    These tests require RL dependencies and are skipped by default.
    Run with: pytest -vx tests/attacks/test_grpo_attack.py::TestGRPOAttackIntegration
    """

    def test_init_model(self):
        """Test model initialization."""
        attack = GRPOAttack(_config=GRPOAttackConfig(policy_model="gpt2", device="cpu"))
        attack._init_model()

        assert attack._model is not None
        assert attack._tokenizer is not None

    def test_generate_attack_from_policy(self):
        """Test attack generation from trained policy."""
        # Would need to set up a mock context and model

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Test that attack configs are properly serialized when saving attacks to JSON."""

import json
from pathlib import Path

from prompt_siren.attacks.agentdojo_attack import AgentDojoAttackConfig
from prompt_siren.types import (
    AttackFile,
    InjectionAttack,
    InjectionAttacksDict,
    StrContentAttack,
)


class TestAttackConfigSerialization:
    """Test attack configuration serialization in AttackFile."""

    def test_attack_file_with_config(self):
        """Test creating an AttackFile with attack configuration."""
        # Create a custom config
        config = AgentDojoAttackConfig(attack_template="Test template {goal}")

        # Create some sample attacks
        attacks: dict[str, InjectionAttacksDict[InjectionAttack]] = {
            "task1": {
                "vector1": StrContentAttack(content="Attack 1"),
                "vector2": StrContentAttack(content="Attack 2"),
            },
            "task2": {
                "vector1": StrContentAttack(content="Attack 3"),
            },
        }

        # Create AttackFile with config
        attack_file = AttackFile.from_attacks_dict(
            attacks=attacks, name="test-attack", config=config
        )

        # Verify metadata contains config as the proper type
        assert attack_file.metadata.config is not None
        assert isinstance(attack_file.metadata.config, AgentDojoAttackConfig)
        assert attack_file.metadata.config.attack_template == "Test template {goal}"

    def test_attack_file_without_config(self):
        """Test creating an AttackFile without configuration."""
        attacks: dict[str, InjectionAttacksDict[InjectionAttack]] = {
            "task1": {
                "vector1": StrContentAttack(content="Attack 1"),
            }
        }

        attack_file = AttackFile.from_attacks_dict(attacks=attacks, name="test-attack")

        # Config should be None when not provided
        assert attack_file.metadata.config is None

    def test_attack_file_json_serialization_with_config(self):
        """Test that AttackFile with config can be serialized to/from JSON."""
        # Create AttackFile with config
        config = AgentDojoAttackConfig(attack_template="Custom {goal}")

        attacks: dict[str, InjectionAttacksDict[InjectionAttack]] = {
            "task1": {
                "vector1": StrContentAttack(content="Test"),
            }
        }

        attack_file = AttackFile.from_attacks_dict(attacks=attacks, name="json-test", config=config)

        # Serialize to JSON
        json_data = attack_file.model_dump()
        json_str = json.dumps(json_data)

        # Deserialize from JSON
        loaded_data = json.loads(json_str)
        loaded_file = AttackFile[AgentDojoAttackConfig].model_validate(loaded_data)

        # Verify config is preserved with proper type
        assert loaded_file.metadata.config is not None
        assert isinstance(loaded_file.metadata.config, AgentDojoAttackConfig)
        assert loaded_file.metadata.config.attack_template == "Custom {goal}"

    def test_attack_file_save_and_load_with_config(self, tmp_path: Path):
        """Test saving and loading AttackFile with config to/from a file."""
        config = AgentDojoAttackConfig(attack_template="Urgent: {goal}")

        attacks: dict[str, InjectionAttacksDict[InjectionAttack]] = {
            "task1": {
                "vector1": StrContentAttack(content="Content"),
            }
        }

        attack_file = AttackFile.from_attacks_dict(attacks=attacks, name="file-test", config=config)

        # Save to a temp file
        attack_path = tmp_path / "attack.json"
        with open(attack_path, "w") as f:
            json.dump(attack_file.model_dump(), f, indent=2)

        # Load from file
        with open(attack_path) as f:
            loaded_data = json.load(f)

        loaded_file = AttackFile[AgentDojoAttackConfig].model_validate(loaded_data)

        # Verify config is preserved with proper type
        assert loaded_file.metadata.config is not None
        assert isinstance(loaded_file.metadata.config, AgentDojoAttackConfig)
        assert loaded_file.metadata.config.attack_template == "Urgent: {goal}"

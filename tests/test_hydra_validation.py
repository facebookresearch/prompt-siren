# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Test Hydra app validation logic."""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from prompt_siren.config.exceptions import ConfigValidationError
from prompt_siren.config.export import export_default_config
from prompt_siren.hydra_app import validate_config
from prompt_siren.registry_base import UnknownComponentError


@pytest.mark.usefixtures("mock_api_keys")
class TestValidateConfig:
    """Test the validate_config() function."""

    def test_valid_minimal_config(self):
        """Valid config should pass validation."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        result = validate_config(cfg, execution_mode="benign")
        assert result.name == "test"
        assert result.agent.type == "plain"

    def test_unknown_agent_type(self):
        """Unknown agent type should raise UnknownComponentError."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "fake_agent", "config": {}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(UnknownComponentError):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_agent_config_error_message(self):
        """Invalid agent config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {
                    "type": "plain",
                    "config": {},
                },  # Missing required 'model' field
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"agent 'plain'") as exc_info:
            validate_config(cfg, execution_mode="benign")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "agent"
        assert exc_info.value.component_name == "plain"
        assert "model" in str(exc_info.value).lower()

    def test_invalid_dataset_config_error_message(self):
        """Invalid dataset config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": 123},
                },  # Invalid type (int instead of str)
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"dataset 'agentdojo'") as exc_info:
            validate_config(cfg, execution_mode="benign")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "dataset"
        assert exc_info.value.component_name == "agentdojo"

    def test_invalid_attack_config_error_message(self):
        """Invalid attack config should include component type in error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": {
                    "type": "template_string",
                    "config": {"attack_template": 123},
                },  # Invalid type (int instead of str)
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ConfigValidationError, match=r"attack 'template_string'") as exc_info:
            validate_config(cfg, execution_mode="attack")

        # Verify the structured exception attributes
        assert exc_info.value.component_type == "attack"
        assert exc_info.value.component_name == "template_string"

    def test_attack_mode_without_attack_raises(self):
        """execution_mode='attack' without attack should raise clear error."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,  # Attack mode but no attack!
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        with pytest.raises(ValueError, match="Attack configuration is required for attack mode"):
            validate_config(cfg, execution_mode="attack")

    def test_benign_only_without_attack_is_valid(self):
        """benign_only mode without attack should be valid."""
        cfg = OmegaConf.create(
            {
                "name": "test",
                "agent": {"type": "plain", "config": {"model": "openai:gpt-4"}},
                "dataset": {
                    "type": "agentdojo",
                    "config": {"suite_name": "workspace", "version": "v1.2.2"},
                },
                "attack": None,
                "execution": {"concurrency": 1},
                "task_ids": None,
                "output": {"trace_dir": "traces"},
                "telemetry": {"trace_console": False},
                "usage_limits": None,
            }
        )

        result = validate_config(cfg, execution_mode="benign")
        assert result.attack is None


@pytest.mark.usefixtures("mock_api_keys")
class TestCliValidationIntegration:
    """Test CLI validation command integration with Hydra compose."""

    def test_validate_with_valid_config(self, tmp_path):
        """Test that CLI validation with valid config succeeds."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory (mimics _validate_config)
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with overrides
            cfg = compose(config_name="config", overrides=["+dataset=agentdojo-workspace"])

            # Validate directly (this is what _validate_config does)
            result = validate_config(cfg, execution_mode="benign")
            assert result is not None
            assert result.dataset.type == "agentdojo"

    def test_validate_with_invalid_config(self, tmp_path):
        """Test that CLI validation with invalid config raises error."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with invalid agent type
            cfg = compose(
                config_name="config",
                overrides=[
                    "+dataset=agentdojo-workspace",
                    "agent.type=nonexistent_agent",
                ],
            )

            # Should raise UnknownComponentError
            with pytest.raises(UnknownComponentError):
                validate_config(cfg, execution_mode="benign")

    def test_validate_attack_mode_requires_attack(self, tmp_path):
        """Test that validating in attack mode without attack config fails."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration without attack
            cfg = compose(config_name="config", overrides=["+dataset=agentdojo-workspace"])

            # Should raise ValueError when validating in attack mode without attack
            with pytest.raises(
                ValueError,
                match="Attack configuration is required for attack mode",
            ):
                validate_config(cfg, execution_mode="attack")

    def test_validate_with_attack_config(self, tmp_path):
        """Test that CLI validation with attack config succeeds."""

        # Export default config to a temporary directory
        config_dir = tmp_path / "config"
        export_default_config(config_dir)

        # Initialize Hydra with the config directory
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Compose configuration with attack
            cfg = compose(
                config_name="config",
                overrides=[
                    "+dataset=agentdojo-workspace",
                    "+attack=agentdojo_important_instructions",
                ],
            )

            # Validate in attack mode
            result = validate_config(cfg, execution_mode="attack")
            assert result is not None
            assert result.attack is not None
            assert result.attack.type == "template_string"

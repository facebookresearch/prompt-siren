# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for Hydra integration with the existing codebase."""

from unittest.mock import patch

import pytest
from omegaconf import DictConfig
from prompt_siren.agents.registry import agent_registry
from prompt_siren.attacks.registry import attack_registry
from prompt_siren.config.experiment_config import (
    DatasetConfig,
    ExperimentConfig,
)
from prompt_siren.config.registry_bridge import create_dataset_from_config
from prompt_siren.datasets.registry import dataset_registry
from prompt_siren.environments.agentdojo_env import AgentDojoEnv
from prompt_siren.hydra_app import validate_config
from prompt_siren.registry_base import UnknownComponentError
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from .conftest import (
    create_mock_agent,
    create_mock_attack,
    create_mock_dataset,
    create_mock_environment,
    MockAgentConfig,
    MockAttackConfig,
    MockDatasetConfig,
    MockEnvironment,
)


class TestExperimentConfig:
    """Test the Pydantic experiment configuration schema."""

    def test_minimal_config(self):
        """Test creating a minimal valid experiment configuration."""
        config_data = {
            "name": "test_experiment",
            "agent": {"type": "plain", "config": {"model": "test"}},
            "dataset": {"type": "agentdojo", "config": {}},
        }

        config = ExperimentConfig.model_validate(config_data)
        assert config.name == "test_experiment"
        assert config.agent.type == "plain"
        assert config.dataset.type == "agentdojo"
        assert config.attack is None  # Optional

    def test_full_config(self):
        """Test creating a complete experiment configuration."""
        config_data = {
            "name": "full_test",
            "agent": {"type": "plain", "config": {"model": "test"}},
            "dataset": {
                "type": "agentdojo",
                "config": {"suite_name": "workspace"},
            },
            "attack": {
                "type": "agentdojo",
                "config": {"user_name": "Test User"},
            },
            "execution": {"concurrency": 4},
            "task_ids": None,
            "output": {"trace_dir": "test_traces"},
            "telemetry": {"trace_console": True},
        }

        config = ExperimentConfig.model_validate(config_data)
        assert config.name == "full_test"
        assert config.execution.concurrency == 4
        assert config.task_ids is None

    def test_invalid_config(self):
        """Test that invalid configurations raise ValidationError."""
        # Missing required agent field
        invalid_config = {
            "name": "invalid_test",
            "dataset": {"type": "agentdojo", "config": {}},
        }

        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(invalid_config)


class TestValidateConfig:
    """Test the validate_config function from hydra_app."""

    @pytest.fixture(autouse=True)
    def setup_registries(self):
        """Set up mock components in registries for testing."""

        # Register mock components only if not already registered
        if "mock" not in agent_registry._registry:
            agent_registry.register("mock", MockAgentConfig, create_mock_agent)
        if "mock" not in dataset_registry._registry:
            dataset_registry.register("mock", MockDatasetConfig, create_mock_dataset)
        if "mock" not in attack_registry._registry:
            attack_registry.register("mock", MockAttackConfig, create_mock_attack)

        # Clear registries after test
        yield
        # Clean up registrations
        if "mock" in agent_registry._registry:
            del agent_registry._registry["mock"]
        if "mock" in dataset_registry._registry:
            del dataset_registry._registry["mock"]
        if "mock" in attack_registry._registry:
            del attack_registry._registry["mock"]

    def test_valid_complete_configuration(self):
        """Test that a complete valid configuration passes validation."""
        config_dict = {
            "name": "test_experiment",
            "description": "Test description",
            "tags": ["test", "validation"],
            "agent": {"type": "mock", "config": {"name": "test_agent"}},
            "dataset": {"type": "mock", "config": {"name": "test_env"}},
            "attack": {"type": "mock", "config": {"name": "test_attack"}},
            "execution": {"concurrency": 2},
            "task_ids": None,
            "output": {"trace_dir": "test_traces"},
            "telemetry": {"trace_console": False},
        }

        cfg = DictConfig(config_dict)
        result = validate_config(cfg, execution_mode="attack")

        assert isinstance(result, ExperimentConfig)
        assert result.name == "test_experiment"
        assert result.agent.type == "mock"
        assert result.dataset.type == "mock"
        assert result.attack is not None
        assert result.attack.type == "mock"
        assert result.execution.concurrency == 2

    def test_valid_minimal_configuration(self):
        """Test that a minimal valid configuration passes validation with benign_only mode."""
        config_dict = {
            "name": "minimal_test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "mock", "config": {}},
        }

        cfg = DictConfig(config_dict)
        result = validate_config(cfg, execution_mode="benign")

        assert isinstance(result, ExperimentConfig)
        assert result.name == "minimal_test"
        assert result.attack is None

    def test_invalid_agent_type_raises_unknown_component_error(self):
        """Test that an unregistered agent type raises UnknownComponentError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "nonexistent_agent", "config": {}},
            "dataset": {"type": "mock", "config": {}},
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(
            UnknownComponentError,
            match="Agent type 'nonexistent_agent' is not registered",
        ):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_environment_type_raises_unknown_component_error(self):
        """Test that an unregistered environment type raises UnknownComponentError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "nonexistent_env", "config": {}},
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(
            UnknownComponentError,
            match="Dataset type 'nonexistent_env' is not registered",
        ):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_attack_type_raises_unknown_component_error(self):
        """Test that an unregistered attack type raises UnknownComponentError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "mock", "config": {}},
            "attack": {"type": "nonexistent_attack", "config": {}},
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(
            UnknownComponentError,
            match="Attack type 'nonexistent_attack' is not registered",
        ):
            validate_config(cfg, execution_mode="attack")

    def test_execution_mode_attack_without_attack_raises_value_error(self):
        """Test that execution_mode='attack' without an attack configuration raises ValueError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "mock", "config": {}},
            # No attack specified
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(ValueError, match="Attack configuration is required for attack mode"):
            validate_config(cfg, execution_mode="attack")

    def test_invalid_agent_config_raises_value_error(self):
        """Test that an invalid agent configuration raises ValueError."""
        config_dict = {
            "name": "test",
            "agent": {
                "config_type": "agent",
                "type": "mock",
                "config": {
                    "invalid_field": "value",
                    "name": "test",
                },  # invalid_field not in MockAgentConfig
            },
            "dataset": {"type": "mock", "config": {}},
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(ValueError, match="Invalid configuration for agent"):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_dataset_config_raises_value_error(self):
        """Test that an invalid dataset configuration raises ValueError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {
                "config_type": "dataset",
                "type": "mock",
                "config": {
                    "invalid_field": "value",
                    "name": "test",
                },  # invalid_field not in MockDatasetConfig
            },
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(ValueError, match="Invalid configuration for dataset"):
            validate_config(cfg, execution_mode="benign")

    def test_invalid_attack_config_raises_value_error(self):
        """Test that an invalid attack configuration raises ValueError."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "mock", "config": {}},
            "attack": {
                "config_type": "attack",
                "type": "mock",
                "config": {
                    "invalid_field": "value",
                    "name": "test",
                },  # invalid_field not in MockAttackConfig
            },
        }

        cfg = DictConfig(config_dict)

        with pytest.raises(ValueError, match="Invalid configuration for attack"):
            validate_config(cfg, execution_mode="attack")

    def test_user_error_is_propagated(self):
        """Test that UserError exceptions from component validation are properly propagated.

        This tests that validate_config properly propagates UserError exceptions that may
        be raised during validation when external resources (like API keys) are not available.
        This ensures users know when they need to configure their environment.
        """
        # Mock the agent config validation to raise UserError
        with patch("prompt_siren.hydra_app.get_agent_config_class") as mock_get_class:
            # Create a mock config class that raises UserError on validation
            class MockConfigWithUserError:
                @classmethod
                def model_validate(cls, data):
                    raise UserError("API key not found in environment variables")

            mock_get_class.return_value = MockConfigWithUserError

            config_dict = {
                "name": "test",
                "agent": {"type": "mock_with_error", "config": {}},
                "dataset": {"type": "mock", "config": {}},
            }

            cfg = DictConfig(config_dict)

            # Should raise UserError - it should NOT be caught
            with pytest.raises(UserError, match="API key not found in environment variables"):
                validate_config(cfg, execution_mode="benign")

    def test_registry_validates_component_configs(self):
        """Test that validate_config properly validates component configs through registries.

        This tests the key functionality where validate_config uses the registry
        to get the config class and validate the component configuration.
        """
        # Valid config that passes all registry validations
        config_dict = {
            "name": "registry_test",
            "agent": {"type": "mock", "config": {"name": "valid_name"}},
            "dataset": {"type": "mock", "config": {"name": "valid_env"}},
            "attack": {"type": "mock", "config": {"name": "valid_attack"}},
        }

        cfg = DictConfig(config_dict)
        result = validate_config(cfg, execution_mode="attack")

        # Verify that the config was successfully validated through registries
        assert result.name == "registry_test"
        assert result.agent.config["name"] == "valid_name"
        assert result.dataset.config["name"] == "valid_env"
        assert result.attack is not None
        assert result.attack.config["name"] == "valid_attack"

    def test_attack_with_benign_only_execution_mode(self):
        """Test that an attack configuration with benign_only execution mode is valid."""
        config_dict = {
            "name": "test",
            "agent": {"type": "mock", "config": {}},
            "dataset": {"type": "mock", "config": {}},
            "attack": {"type": "mock", "config": {}},
        }

        cfg = DictConfig(config_dict)
        result = validate_config(cfg, execution_mode="benign")

        # Should pass validation - attack can be present even if execution_mode is benign_only
        assert isinstance(result, ExperimentConfig)
        assert result.attack is not None

    def test_validation_order_pydantic_then_registry(self):
        """Test that validate_config validates Pydantic schema first, then registry.

        This verifies the order of validation operations in validate_config.
        """
        # Missing required agent field - should fail at Pydantic validation level
        config_dict = {
            "name": "test",
            "dataset": {"type": "mock", "config": {}},
        }

        cfg = DictConfig(config_dict)

        # Should raise ValidationError from Pydantic, not KeyError from registry
        with pytest.raises(ValidationError, match="agent"):
            validate_config(cfg, execution_mode="benign")

    def test_complex_nested_configuration(self):
        """Test validation of a complex configuration with nested settings.

        This tests that validate_config properly handles complex configurations
        with all optional fields filled in and validates them through registries.
        """
        config_dict = {
            "name": "complex_test",
            "agent": {
                "config_type": "agent",
                "type": "mock",
                "config": {
                    "name": "complex_agent",
                    "custom_parameter": "custom_value",
                },
            },
            "dataset": {
                "config_type": "environment",
                "type": "mock",
                "config": {
                    "name": "complex_env",
                    "custom_parameter": "env_value",
                },
            },
            "attack": {
                "config_type": "attack",
                "type": "mock",
                "config": {
                    "name": "complex_attack",
                    "custom_parameter": "attack_value",
                },
            },
            "execution": {"concurrency": 8},
            "task_ids": ["task1", "task2"],
            "usage_limits": {
                "input_tokens_limit": 1000,
                "output_tokens_limit": 500,
                "total_tokens_limit": 1500,
            },
            "output": {"trace_dir": "complex_traces"},
            "telemetry": {
                "trace_console": True,
                "otel_endpoint": "http://localhost:4317",
            },
        }

        cfg = DictConfig(config_dict)
        result = validate_config(cfg, execution_mode="attack")

        assert isinstance(result, ExperimentConfig)
        assert result.name == "complex_test"
        assert result.agent.config["custom_parameter"] == "custom_value"
        assert result.dataset.config["custom_parameter"] == "env_value"
        assert result.attack is not None
        assert result.attack.config["custom_parameter"] == "attack_value"
        assert result.task_ids == ["task1", "task2"]
        assert result.usage_limits is not None
        assert result.usage_limits.input_tokens_limit == 1000
        assert result.usage_limits.output_tokens_limit == 500
        assert result.usage_limits.total_tokens_limit == 1500


class TestDatasetEnvironmentIntegration:
    """Integration tests for dataset and environment coordination."""

    @pytest.fixture(autouse=True)
    def setup_registries(self):
        """Set up mock components in registries for testing."""
        # Register mock components
        if "mock" not in dataset_registry._registry:
            dataset_registry.register("mock", MockDatasetConfig, create_mock_dataset)

        yield

        # Clean up registrations
        if "mock" in dataset_registry._registry:
            del dataset_registry._registry["mock"]

    def test_dataset_provides_correct_environment_type(self):
        """Test that dataset provides the correct environment instance."""

        config = DatasetConfig(type="mock", config={"name": "test"})
        dataset = create_dataset_from_config(config)

        # Dataset should provide an environment instance
        assert dataset.environment is not None
        assert isinstance(dataset.environment, MockEnvironment)

    def test_environment_created_from_dataset_matches_type(self):
        """Test that environment created from dataset has the correct type."""

        config = DatasetConfig(type="mock", config={"name": "test"})
        dataset = create_dataset_from_config(config)
        env = dataset.environment

        # Environment should match the type specified by dataset
        assert isinstance(env, type(create_mock_environment(MockDatasetConfig())))

    def test_environment_config_from_dataset_matches_expectations(self):
        """Test that environment from dataset is configured correctly."""

        config = DatasetConfig(
            type="mock",
            config={"name": "test_env", "custom_parameter": "custom_value"},
        )
        dataset = create_dataset_from_config(config)

        # Get environment from dataset
        env = dataset.environment
        assert isinstance(env, MockEnvironment)
        assert env.name == "test_env"

    @pytest.mark.parametrize("suite_name", ["banking", "slack", "travel", "workspace"])
    def test_agentdojo_dataset_environment_integration(self, suite_name: str):
        """Test real AgentDojo dataset-environment integration."""

        # Create AgentDojo dataset
        config = DatasetConfig(
            type="agentdojo",
            config={"suite_name": suite_name, "version": "v1.2.2"},
        )
        dataset = create_dataset_from_config(config)

        # Verify dataset properties
        assert len(dataset.benign_tasks) > 0
        assert len(dataset.malicious_tasks) > 0

        # Verify environment from dataset
        env = dataset.environment
        assert isinstance(env, AgentDojoEnv)
        assert env.name == f"agentdojo-{suite_name}"

        # Verify dataset has toolsets
        toolsets = dataset.default_toolsets
        assert len(toolsets) > 0

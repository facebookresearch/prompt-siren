# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Bridge between Hydra configurations and existing component registries."""

from ..agents.abstract import AbstractAgent
from ..agents.registry import create_agent, get_agent_config_class
from ..attacks.abstract import AbstractAttack
from ..attacks.registry import create_attack, get_attack_config_class
from ..datasets.abstract import AbstractDataset
from ..datasets.registry import create_dataset, get_dataset_config_class
from ..sandbox_managers.abstract import AbstractSandboxManager
from ..sandbox_managers.registry import (
    create_sandbox_manager,
    get_sandbox_config_class,
)
from .experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
    SandboxManagerConfig,
)


def create_agent_from_config(config: AgentConfig) -> AbstractAgent:
    """Create an agent instance from configuration.

    Args:
        config: Agent configuration (AgentConfig, DictConfig, or dict)

    Returns:
        Configured agent instance

    Raises:
        KeyError: If agent type is not registered
        ValidationError: If configuration is invalid
    """
    # Get the Pydantic config class and validate
    config_class = get_agent_config_class(config.type)
    validated_config = config_class.model_validate(config.config)

    return create_agent(config.type, validated_config)


def create_attack_from_config(config: AttackConfig) -> AbstractAttack:
    """Create an attack instance from configuration.

    Args:
        config: Attack configuration

    Returns:
        Configured attack instance

    Raises:
        KeyError: If attack type is not registered
        ValidationError: If configuration is invalid
    """
    # Get the Pydantic config class and validate
    config_class = get_attack_config_class(config.type)
    validated_config = config_class.model_validate(config.config)

    return create_attack(config.type, validated_config)


def create_sandbox_manager_from_config(
    config: SandboxManagerConfig,
) -> AbstractSandboxManager:
    """Create a sandbox manager instance from configuration.

    Args:
        config: Sandbox manager configuration

    Returns:
        Configured sandbox manager instance

    Raises:
        KeyError: If sandbox manager type is not registered
        ValidationError: If configuration is invalid
    """
    # Get the Pydantic config class and validate
    config_class = get_sandbox_config_class(config.type)
    validated_config = config_class.model_validate(config.config)

    return create_sandbox_manager(config.type, validated_config)


def create_dataset_from_config(
    config: DatasetConfig, sandbox_manager: AbstractSandboxManager | None = None
) -> AbstractDataset:
    """Create a dataset instance from configuration.

    Args:
        config: Dataset configuration (DatasetConfig, DictConfig, or dict)
        sandbox_manager: Optional sandbox manager for datasets that require it

    Returns:
        Configured dataset instance with tasks and environment

    Raises:
        KeyError: If dataset type is not registered
        ValidationError: If configuration is invalid
    """
    # Get the Pydantic config class and validate
    config_class = get_dataset_config_class(config.type)
    validated_config = config_class.model_validate(config.config)

    return create_dataset(config.type, validated_config, sandbox_manager)

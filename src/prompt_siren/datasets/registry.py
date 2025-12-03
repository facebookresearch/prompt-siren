# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry for dataset types and their factory functions.

This module provides a centralized registry for dataset types, allowing for
dynamic dataset creation with type safety and optional sandbox manager context.
"""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from ..registry_base import BaseRegistry
from ..sandbox_managers.abstract import AbstractSandboxManager
from .abstract import AbstractDataset

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for dataset factory functions with optional sandbox manager context
DatasetFactory: TypeAlias = Callable[[ConfigT, AbstractSandboxManager | None], AbstractDataset]


# Create a global dataset registry instance using BaseRegistry with context support
dataset_registry = BaseRegistry[AbstractDataset, AbstractSandboxManager | None](
    "dataset", "prompt_siren.datasets"
)


# Convenience functions for dataset-specific naming
def register_dataset(
    dataset_type: str,
    config_class: type[ConfigT],
    factory: DatasetFactory[ConfigT],
) -> None:
    """Register a dataset type with its configuration class and factory."""
    dataset_registry.register(dataset_type, config_class, factory)


def get_dataset_config_class(dataset_type: str) -> type[BaseModel]:
    """Get the configuration class for a dataset type."""
    config_class = dataset_registry.get_config_class(dataset_type)
    if config_class is None:
        raise RuntimeError(f"Dataset type '{dataset_type}' must have a config class")
    return config_class


def create_dataset(
    dataset_type: str,
    config: BaseModel,
    sandbox_manager: AbstractSandboxManager | None = None,
) -> AbstractDataset:
    """Create a dataset instance from a configuration.

    Args:
        dataset_type: The type of dataset to create
        config: The dataset configuration
        sandbox_manager: Optional sandbox manager for datasets that require it

    Returns:
        The created dataset instance
    """
    return dataset_registry.create_component(dataset_type, config, context=sandbox_manager)


def get_registered_datasets() -> list[str]:
    """Get a list of all registered dataset types."""
    return dataset_registry.get_registered_components()

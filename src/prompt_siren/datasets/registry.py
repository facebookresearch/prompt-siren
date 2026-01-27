# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry for dataset types and their factory functions.

This module provides a centralized registry for dataset types, allowing for
dynamic dataset creation with type safety and optional sandbox manager context.
"""

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable, TypeAlias, TypeVar

from pydantic import BaseModel

from ..registry_base import BaseRegistry
from ..sandbox_managers.abstract import AbstractSandboxManager
from ..sandbox_managers.image_spec import ImageBuildSpec
from .abstract import AbstractDataset

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for dataset factory functions
# Factories accept an optional sandbox manager for datasets that need container support
DatasetFactory: TypeAlias = Callable[[ConfigT, AbstractSandboxManager | None], AbstractDataset]


@runtime_checkable
class ImageBuildableDataset(Protocol):
    """Protocol for dataset classes that support image building.

    Dataset classes can implement this protocol by providing a
    `get_image_build_specs` classmethod that returns the image specs
    needed for the dataset.

    Note:
        The ``config`` parameter is typed as ``BaseModel`` here so that
        ``issubclass(SomeDataset, ImageBuildableDataset)`` works at runtime
        (``runtime_checkable`` requires the protocol signature to be compatible
        with all implementors). Concrete implementations should narrow the type
        to their specific config class (e.g., ``SwebenchDatasetConfig``).
    """

    @classmethod
    def get_image_build_specs(cls, config: BaseModel) -> list[ImageBuildSpec]:
        """Return all image specifications needed by this dataset."""
        ...


# Create a global dataset registry instance using BaseRegistry with context support
dataset_registry = BaseRegistry[AbstractDataset, AbstractSandboxManager | None](
    "dataset", "prompt_siren.datasets"
)


# Convenience functions for dataset-specific naming
def register_dataset(
    dataset_type: str,
    config_class: type[ConfigT],
    factory: DatasetFactory[ConfigT],
    dataset_class: type[ImageBuildableDataset] | None = None,
) -> None:
    """Register a dataset type with its configuration class and factory.

    Args:
        dataset_type: String identifier for the dataset type
        config_class: Pydantic model class for configuration
        factory: Factory function that creates dataset instances
        dataset_class: Optional dataset class that implements ImageBuildableDataset protocol
    """
    dataset_registry.register(dataset_type, config_class, factory, component_class=dataset_class)


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


def get_image_build_specs(dataset_type: str, config: BaseModel) -> list[ImageBuildSpec]:
    """Get image build specs for a dataset type.

    This calls the dataset class's get_image_build_specs classmethod,
    allowing image specs to be retrieved without creating a full dataset instance.

    Args:
        dataset_type: The type of dataset
        config: The dataset configuration

    Returns:
        List of image build specs for the dataset

    Raises:
        ImportError: If the dataset's dependencies are not installed
        ValueError: If the dataset class doesn't support image building
    """
    # Triggers entry point loading + checks for failed entry points
    get_dataset_config_class(dataset_type)
    classes = dataset_registry.get_component_classes()
    dataset_class = classes.get(dataset_type)
    if dataset_class is None:
        raise ValueError(
            f"Dataset type '{dataset_type}' does not support image building. "
            "The entry point must be a 3-tuple (factory, config_class, component_class) "
            "to provide a component class."
        )
    if not issubclass(dataset_class, ImageBuildableDataset):
        raise ValueError(
            f"Dataset type '{dataset_type}' does not support image building. "
            "The dataset class must implement the ImageBuildableDataset protocol "
            "(provide a get_image_build_specs classmethod)."
        )
    return dataset_class.get_image_build_specs(config)


def get_datasets_with_image_specs() -> list[str]:
    """Get a list of dataset types that support image building.

    Returns:
        List of dataset type names that support image building
    """
    dataset_registry.get_registered_components()  # trigger entry point loading
    for name, error in dataset_registry.failed_entry_points.items():
        logger.warning(f"Dataset '{name}' could not be loaded: {error}")
    return [
        name
        for name, cls in dataset_registry.get_component_classes().items()
        if issubclass(cls, ImageBuildableDataset)
    ]

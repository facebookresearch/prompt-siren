# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry for dataset types and their factory functions.

This module provides a centralized registry for dataset types, allowing for
dynamic dataset creation with type safety and optional sandbox manager context.
"""

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable, TypeAlias, TypeVar

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

# Type alias for dataset entry points - tuple of (factory_fn, dataset_class)
DatasetEntry: TypeAlias = tuple[DatasetFactory[Any], type[AbstractDataset]]


@runtime_checkable
class ImageBuildableDataset(Protocol):
    """Protocol for dataset classes that support image building.

    Dataset classes can implement this protocol by providing a
    `get_image_build_specs` classmethod that returns the image specs
    needed for the dataset.
    """

    @classmethod
    def get_image_build_specs(cls, config: BaseModel) -> list[ImageBuildSpec]:
        """Return all image specifications needed by this dataset."""
        ...


# Create a global dataset registry instance using BaseRegistry with context support
dataset_registry = BaseRegistry[AbstractDataset, AbstractSandboxManager | None](
    "dataset", "prompt_siren.datasets"
)

# Separate storage for dataset classes that support image building
_image_buildable_classes: dict[str, type[ImageBuildableDataset]] = {}
_image_buildable_classes_loaded: bool = False
_image_buildable_load_error: Exception | None = None
_image_buildable_failed_entry_points: dict[str, Exception] = {}


def _ensure_image_buildable_classes_loaded(operation: str) -> None:
    """Load dataset classes from entry points and raise if loading failed.

    Entry points that return a tuple of (factory_fn, dataset_class) are inspected.
    If the dataset_class implements ImageBuildableDataset protocol,
    it is registered for image building.

    Args:
        operation: Description of the operation for error message

    Raises:
        RuntimeError: If entry point loading failed completely
    """
    global _image_buildable_classes_loaded, _image_buildable_load_error
    if _image_buildable_classes_loaded:
        if _image_buildable_load_error is not None:
            raise RuntimeError(
                f"Cannot {operation}: entry point loading failed"
            ) from _image_buildable_load_error
        return

    try:
        entry_points = importlib.metadata.entry_points(group="prompt_siren.datasets")
        for ep in entry_points:
            if ep.name in _image_buildable_classes:
                continue
            try:
                entry = ep.load()
                # Only tuples of (factory, dataset_class) are inspected for image building
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                _, dataset_class = entry
                if issubclass(dataset_class, ImageBuildableDataset):
                    _image_buildable_classes[ep.name] = dataset_class
            except ImportError as e:
                # Expected for optional dependencies - log at debug level
                logger.debug(f"Skipping dataset '{ep.name}': missing dependency: {e}")
            except Exception as e:
                # Unexpected error - store for later retrieval and log
                logger.error(
                    f"Failed to load dataset class for '{ep.name}': {type(e).__name__}: {e}",
                    exc_info=True,
                )
                _image_buildable_failed_entry_points[ep.name] = e
    except Exception as e:
        logger.error(
            f"Failed to load dataset entry points: {type(e).__name__}: {e}",
            exc_info=True,
        )
        _image_buildable_load_error = e

    _image_buildable_classes_loaded = True

    if _image_buildable_load_error is not None:
        raise RuntimeError(
            f"Cannot {operation}: entry point loading failed"
        ) from _image_buildable_load_error


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
    dataset_registry.register(dataset_type, config_class, factory)
    if dataset_class is not None:
        _image_buildable_classes[dataset_type] = dataset_class


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
        RuntimeError: If entry point loading failed completely
        ValueError: If the dataset class doesn't support image building
    """
    _ensure_image_buildable_classes_loaded("get image specs")
    # Check if this specific entry point failed to load
    if dataset_type in _image_buildable_failed_entry_points:
        raise _image_buildable_failed_entry_points[dataset_type]
    dataset_class = _image_buildable_classes.get(dataset_type)
    if dataset_class is None:
        raise ValueError(
            f"Dataset type '{dataset_type}' does not support image building. "
            "The dataset class must have a get_image_build_specs classmethod."
        )
    return dataset_class.get_image_build_specs(config)


def get_datasets_with_image_specs() -> list[str]:
    """Get a list of dataset types that support image building.

    Returns:
        List of dataset type names that support image building

    Raises:
        RuntimeError: If entry point loading failed completely
    """
    _ensure_image_buildable_classes_loaded("list datasets")
    return list(_image_buildable_classes.keys())

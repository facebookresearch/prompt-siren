# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Generic base registry for composable configuration of components.

This module provides a generic registry pattern that can be used for any component type
(agents, attacks, environments, etc.) with their configurations, enabling composability
through protocols and factory functions rather than inheritance.
"""

import importlib.metadata
import inspect
import typing
from collections.abc import Callable
from typing import Any, Generic, TypeAlias, TypeVar

from pydantic import BaseModel

ComponentT = TypeVar("ComponentT")
ConfigT = TypeVar("ConfigT", bound=BaseModel)
ContextT = TypeVar("ContextT")

# Type alias for factory functions with optional context parameter
# Note: Actual signature depends on whether config is needed - see register() for details
ComponentFactory: TypeAlias = Callable[..., ComponentT]


class UnknownComponentError(KeyError):
    """Raised when attempting to use a component type that is not known to the registry.

    This exception is raised when trying to access a component type (agent, environment,
    attack, etc.) that hasn't been registered in the corresponding registry.
    """

    def __init__(
        self,
        component_type: str,
        component_name: str,
        available_types: list[str],
    ):
        """Initialize the exception with detailed information.

        Args:
            component_type: The type of component (e.g., 'agent', 'environment', 'attack')
            component_name: The specific name that was not found
            available_types: List of available registered types
        """
        self.component_type = component_type
        self.component_name = component_name
        self.available_types = available_types

        available_str = ", ".join(available_types) if available_types else "none"
        message = f"{component_type.title()} type '{component_name}' is not registered. Available types: {available_str}"
        super().__init__(message)


class BaseRegistry(Generic[ComponentT, ContextT]):
    """Generic base registry for component types and their factory functions.

    This registry maintains a mapping between component type names and their
    corresponding configuration classes and factory functions.

    It supports both manual registration and automatic discovery via entry points.

    Type Parameters:
        ComponentT: The type of component this registry manages (e.g., AbstractAgent)
        ContextT: Optional context type passed to factory functions (e.g., AbstractSandboxManager)
    """

    def __init__(self, component_name: str, entry_point_group: str | None = None):
        """Initialize a new registry.

        Args:
            component_name: Name of the component type for error messages (e.g., 'agent', 'attack')
            entry_point_group: Entry point group name for automatic discovery (e.g., 'prompt_siren.agents')
        """
        self._registry: dict[str, tuple[type[BaseModel] | None, ComponentFactory]] = {}
        self._component_name = component_name
        self._entry_point_group = entry_point_group
        self._entry_points_loaded = False
        # Store errors from failed entry point loads to re-raise when plugin is actually requested
        self._failed_entry_points: dict[str, Exception] = {}

    def _load_entry_points(self) -> None:
        """Load components from entry points if not already loaded.

        Failed entry points are stored silently and re-raised when the plugin is actually requested.
        This avoids warning users about missing optional dependencies they don't intend to use.
        """
        if self._entry_points_loaded or not self._entry_point_group:
            return

        try:
            entry_points = importlib.metadata.entry_points(group=self._entry_point_group)
            for ep in entry_points:
                try:
                    # Load the factory function
                    factory = ep.load()

                    # Get config class from factory function signature
                    config_class = self._get_config_class_from_factory(factory, ep.name)

                    # Register if not already present (manual registration takes precedence)
                    if ep.name not in self._registry:
                        self._registry[ep.name] = (config_class, factory)

                except ImportError as e:  # noqa: PERF203 - graceful plugin loading
                    # Store the import error silently - will be raised when plugin is actually requested
                    self._failed_entry_points[ep.name] = e
                except Exception as e:
                    print(f"Warning: Failed to load entry point {ep.name}: {e}")

        except Exception as e:
            print(f"Warning: Failed to load entry points for {self._entry_point_group}: {e}")

        self._entry_points_loaded = True

    def _check_failed_entry_point(self, component_type: str) -> None:
        """Check if a component failed to load and re-raise the original error.

        Args:
            component_type: The component type being requested

        Raises:
            The original exception that occurred when trying to load the entry point
        """
        if component_type in self._failed_entry_points:
            raise self._failed_entry_points[component_type]

    def _get_config_class_from_factory(
        self, factory: Callable[..., Any], name: str
    ) -> type[BaseModel] | None:
        """Extract config class from factory function signature.

        Returns None if the factory has no parameters (indicating it doesn't need config).
        """
        sig = inspect.signature(factory)
        if not sig.parameters:
            # Factory with no parameters - doesn't need config
            return None

        first_param = next(iter(sig.parameters.values()))
        if first_param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Cannot determine config class for {name} - missing type annotation")

        # Handle both actual class objects and string annotations
        annotation = first_param.annotation
        if isinstance(annotation, str):
            # String annotation - need to resolve it in the function's module context
            if hasattr(typing, "get_type_hints"):
                try:
                    type_hints = typing.get_type_hints(factory)
                    param_name = first_param.name
                    if param_name in type_hints:
                        return type_hints[param_name]
                except Exception:
                    pass
            raise ValueError(f"Cannot resolve string annotation '{annotation}' for {name}")

        return annotation

    def register(
        self,
        component_type: str,
        config_class: type[ConfigT] | None,
        factory: Callable[..., ComponentT],
    ) -> None:
        """Register a new component type with optional config class and factory function.

        Args:
            component_type: String identifier for the component type
            config_class: Pydantic model class for configuration, or None if component doesn't need config
            factory: Factory function that creates component instances.
                     If config_class is None, factory should have signature: () -> ComponentT
                     If config_class is provided, factory should have signature: (ConfigT, ContextT | None) -> ComponentT

        Raises:
            ValueError: If the component type is already registered
        """
        if component_type in self._registry:
            raise ValueError(
                f"{self._component_name.title()} type '{component_type}' is already registered"
            )
        self._registry[component_type] = (config_class, factory)

    def get_config_class(self, component_type: str) -> type[BaseModel] | None:
        """Get the config class for a given component type.

        Args:
            component_type: String identifier for the component type

        Returns:
            The config class for the component type, or None if component doesn't use config

        Raises:
            UnknownComponentError: If the component type is not registered
            ImportError: If the component failed to load due to missing dependencies
        """
        self._load_entry_points()  # Ensure entry points are loaded

        # Check if this component failed to load - re-raise the original error
        self._check_failed_entry_point(component_type)

        if component_type not in self._registry:
            raise UnknownComponentError(
                self._component_name,
                component_type,
                self.get_registered_components(),
            )
        return self._registry[component_type][0]

    def create_component(
        self,
        component_type: str,
        config: BaseModel | None,
        context: ContextT | None = None,
    ) -> ComponentT:
        """Create a component instance for a given component type and config.

        Args:
            component_type: String identifier for the component type
            config: Configuration object for the component, or None
            context: Optional context object passed to factory (e.g., sandbox_manager for datasets)

        Returns:
            An instance of the component type

        Raises:
            UnknownComponentError: If the component type is not registered
            TypeError: If the config is not of the expected type
            ValueError: If config is provided but component doesn't accept config, or vice versa
            ImportError: If the component failed to load due to missing dependencies
        """
        self._load_entry_points()  # Ensure entry points are loaded

        # Check if this component failed to load - re-raise the original error
        self._check_failed_entry_point(component_type)

        if component_type not in self._registry:
            raise UnknownComponentError(
                self._component_name,
                component_type,
                self.get_registered_components(),
            )

        config_class, factory = self._registry[component_type]

        # Component doesn't use config - call factory with no args
        if config_class is None:
            if config is not None:
                raise ValueError(
                    f"{self._component_name.title()} type '{component_type}' doesn't accept config, "
                    f"but config was provided"
                )
            return factory()

        # Component uses config - validate and call with config and context
        if config is None:
            config = config_class()  # Use default config
        elif not isinstance(config, config_class):
            raise TypeError(
                f"Config must be an instance of {config_class.__name__}, got {type(config).__name__}"
            )

        return factory(config, context)

    def get_registered_components(self) -> list[str]:
        """Get a list of all registered component types.

        Returns:
            List of component type identifiers
        """
        self._load_entry_points()  # Ensure entry points are loaded
        return list(self._registry.keys())

    def get_registry_info(self) -> dict[str, dict[str, Any]]:
        """Get detailed information about registered components.

        Returns:
            Dictionary mapping component type names to their metadata
        """
        self._load_entry_points()  # Ensure entry points are loaded
        return {
            name: {
                "config_class": config_class.__name__ if config_class is not None else None,
                "config_fields": list(config_class.model_fields.keys())
                if config_class is not None
                else [],
                "factory": factory.__name__,
            }
            for name, (config_class, factory) in self._registry.items()
        }

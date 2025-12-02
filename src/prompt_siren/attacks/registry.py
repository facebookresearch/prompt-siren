# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Attack registry for composable configuration of attacks.

This module provides a registry for attack types and their configurations,
enabling composability through protocols and factory functions rather than
inheritance. This approach allows for different attack types with specialized
configuration types while maintaining type safety.
"""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from ..registry_base import BaseRegistry
from .abstract import AbstractAttack

ConfigT = TypeVar("ConfigT", bound=BaseModel)

# Type alias for attack factory functions
AttackFactory: TypeAlias = Callable[[ConfigT, None], AbstractAttack]


# Create a global attack registry instance
attack_registry = BaseRegistry[AbstractAttack, None]("attack", "prompt_siren.attacks")


# Convenience functions for attack-specific naming
def register_attack(
    attack_type: str,
    config_class: type[ConfigT],
    factory: AttackFactory[ConfigT],
) -> None:
    """Register a new attack type with its config class and factory function."""
    attack_registry.register(attack_type, config_class, factory)


def get_attack_config_class(attack_type: str) -> type[BaseModel]:
    """Get the config class for a given attack type."""
    config_class = attack_registry.get_config_class(attack_type)
    if config_class is None:
        raise RuntimeError(f"Attack type '{attack_type}' must have a config class")
    return config_class


def create_attack(attack_type: str, config: BaseModel) -> AbstractAttack:
    """Create an attack instance for a given attack type and config."""
    return attack_registry.create_component(attack_type, config)


def get_registered_attacks() -> list[str]:
    """Get a list of all registered attack types."""
    return attack_registry.get_registered_components()

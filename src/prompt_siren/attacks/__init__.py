# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Attack implementations and configuration for Siren."""

from .abstract import AbstractAttack
from .registry import (
    create_attack,
    get_attack_config_class,
    get_registered_attacks,
    register_attack,
)

__all__ = [
    # Core abstractions
    "AbstractAttack",
    "create_attack",
    "get_attack_config_class",
    "get_registered_attacks",
    # Registry functions
    "register_attack",
]

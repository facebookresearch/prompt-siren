# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Attack implementations and configuration for Siren."""

from .abstract import AbstractAttack
from .executor import (
    InjectionCheckpoint,
    RolloutExecutor,
    RolloutRequest,
    RolloutResult,
)
from .registry import (
    create_attack,
    get_attack_config_class,
    get_registered_attacks,
    register_attack,
)
from .results import AttackResults, CoupleAttackResult
from .simple_attack_base import InjectionContext, SimpleAttackBase

__all__ = [
    "AbstractAttack",
    "AttackResults",
    "CoupleAttackResult",
    "InjectionCheckpoint",
    "InjectionContext",
    "RolloutExecutor",
    "RolloutRequest",
    "RolloutResult",
    "SimpleAttackBase",
    "create_attack",
    "get_attack_config_class",
    "get_registered_attacks",
    "register_attack",
]

# Note: DefaultRolloutExecutor is not exported here to avoid circular imports.
# Import directly: from prompt_siren.attacks.default_executor import DefaultRolloutExecutor

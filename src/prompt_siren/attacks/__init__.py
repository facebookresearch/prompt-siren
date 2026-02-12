# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Attack implementations and configuration for Siren."""

from .abstract import AbstractAttack
from .executor import (
    EnvironmentSnapshotAtInjection,
    ResumeInfo,
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
from .rl_attack_base import RLAttackBase
from .rl_utils import (
    couples_to_hf_dataset,
    create_reward_function,
    ExactMatchReward,
    ExecutorRewardWrapper,
    MultiObjectiveReward,
    RewardFunction,
)
from .simple_attack_base import InjectionContext, SimpleAttackBase

__all__ = [
    "AbstractAttack",
    "AttackResults",
    "CoupleAttackResult",
    "EnvironmentSnapshotAtInjection",
    "ExactMatchReward",
    "ExecutorRewardWrapper",
    "InjectionContext",
    "MultiObjectiveReward",
    "RLAttackBase",
    "ResumeInfo",
    "RewardFunction",
    "RolloutExecutor",
    "RolloutRequest",
    "RolloutResult",
    "SimpleAttackBase",
    "couples_to_hf_dataset",
    "create_attack",
    "create_reward_function",
    "get_attack_config_class",
    "get_registered_attacks",
    "register_attack",
]

# Note: DefaultRolloutExecutor is not exported here to avoid circular imports.
# Import directly: from prompt_siren.attacks.default_executor import DefaultRolloutExecutor

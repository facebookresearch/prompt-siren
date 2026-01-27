# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Dictionary-based attack implementation.

This attack uses pre-computed attack payloads from a dictionary,
useful for replaying saved attacks or testing with known payloads.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import logfire
from pydantic import BaseModel, Field

from ..tasks import TaskCouple
from ..types import (
    AttackFile,
    InjectionAttack,
    InjectionAttacksDict,
    TaskCoupleID,
)
from .executor import RolloutExecutor
from .results import AttackResults
from .simple_attack_base import InjectionContext, SimpleAttackBase

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class DictAttackConfig(BaseModel, Generic[InjectionAttackT]):
    """Configuration for dictionary-based attacks."""

    attacks_by_task: dict[TaskCoupleID, InjectionAttacksDict[InjectionAttackT]] = Field(
        description="Dictionary mapping task couple IDs to injection attacks"
    )
    attack_name: str = Field(description="Name of the attack")


class FileAttackConfig(BaseModel):
    """Configuration for loading attacks from a file."""

    file_path: str = Field(description="Path to JSON attack file")


@dataclass(frozen=True)
class DictAttack(
    SimpleAttackBase[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
):
    """Attack that uses pre-loaded attacks from a dictionary.

    This attack looks up pre-computed attack payloads by task couple ID.
    Useful for:
    - Replaying previously successful attacks
    - Testing with known attack payloads
    - Benchmarking with standardized attacks

    If no attacks are found for a task couple, logs a warning and uses
    an empty attack dictionary.
    """

    name: ClassVar[str] = "dict"
    _config: DictAttackConfig[InjectionAttackT]

    @property
    def config(self) -> DictAttackConfig[InjectionAttackT]:
        """Return the attack configuration."""
        return self._config

    @property
    def attack_name(self) -> str:
        """Get the attack name from the configuration."""
        return self._config.attack_name

    def generate_attack(
        self,
        context: InjectionContext[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> InjectionAttacksDict[InjectionAttackT]:
        """Look up pre-computed attacks for this task couple.

        Args:
            context: Information about the injection point

        Returns:
            Pre-computed attacks for this couple, or empty dict if none found
        """
        # Get task couple ID
        task_couple_id = context.couple.id

        # Look up pre-computed attacks
        attacks_for_task = self._config.attacks_by_task.get(task_couple_id, {})

        if not attacks_for_task:
            logfire.warning(
                f"No pre-computed attacks found for task couple {task_couple_id}",
                available_tasks=list(self._config.attacks_by_task.keys()),
            )

        return attacks_for_task

    async def run(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> AttackResults[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Execute the dictionary attack across all couples.

        Uses the SimpleAttackBase.run() implementation which handles:
        - Checkpoint discovery
        - Attack generation via generate_attack()
        - Rollout execution
        - Checkpoint cleanup

        Args:
            couples: The task couples to attack
            executor: The rollout executor

        Returns:
            Attack results for each couple
        """
        # Delegate to base class implementation
        return await SimpleAttackBase.run(self, couples, executor)


def create_dict_attack(config: DictAttackConfig, context: None = None) -> DictAttack:
    """Factory function to create a DictAttack from its configuration.

    Args:
        config: DictAttackConfig containing attacks_by_task and attack_name
        context: Optional context parameter (unused by attacks, for registry compatibility)

    Returns:
        DictAttack instance
    """
    return DictAttack(_config=config)


def create_dict_attack_from_file(config: FileAttackConfig, context: None = None) -> DictAttack:
    """Factory function to create a DictAttack by loading from a JSON file.

    Args:
        config: FileAttackConfig containing the path to the JSON attack file
        context: Optional context parameter (unused by attacks, for registry compatibility)

    Returns:
        DictAttack instance with attacks loaded from the file

    Raises:
        FileNotFoundError: If the attack file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        pydantic.ValidationError: If the file doesn't match the AttackFile schema
    """
    attack_file_path = Path(config.file_path)

    if not attack_file_path.exists():
        raise FileNotFoundError(f"Attack file not found: {attack_file_path}")

    try:
        with open(attack_file_path) as f:
            attack_data = json.load(f)

        # Validate and parse the attack file
        attack_file = AttackFile.model_validate(attack_data)

        # Convert to attacks dictionary
        attacks_dict, attack_name = attack_file.to_attacks_dict()

        logfire.info(
            f"Loaded attacks from {attack_file_path}",
            attack_name=attack_name,
            num_task_attacks=len(attacks_dict),
        )

        # Create DictAttackConfig and return DictAttack
        dict_config = DictAttackConfig(attacks_by_task=attacks_dict, attack_name=attack_name)
        return DictAttack(_config=dict_config)

    except Exception as e:
        logfire.error(f"Failed to load attack file {attack_file_path}: {e}")
        raise

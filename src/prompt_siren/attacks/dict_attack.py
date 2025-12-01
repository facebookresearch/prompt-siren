# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Dictionary-based attack implementation."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import InstrumentationSettings
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from ..agents.abstract import AbstractAgent
from ..agents.states import EndState
from ..environments.abstract import AbstractEnvironment
from ..tasks import BenignTask, MaliciousTask
from ..types import (
    AttackFile,
    InjectionAttack,
    InjectionAttacksDict,
    TaskCoupleID,
)
from .abstract import AbstractAttack

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
    AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
):
    """Attack that uses pre-loaded attacks from a dictionary."""

    name: ClassVar[str] = "dict"
    _config: DictAttackConfig[InjectionAttackT]

    @property
    def config(self) -> DictAttackConfig[InjectionAttackT]:
        return self._config

    @property
    def attack_name(self) -> str:
        """Get the attack name from the configuration."""
        return self._config.attack_name

    async def attack(
        self,
        agent: AbstractAgent,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        message_history: Sequence[ModelMessage],
        env_state: EnvStateT,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        benign_task: BenignTask[EnvStateT],
        malicious_task: MaliciousTask[EnvStateT],
        usage_limits: UsageLimits,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> tuple[
        EndState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        InjectionAttacksDict[InjectionAttackT],
    ]:
        """Execute attack using pre-loaded attacks from dictionary.

        Simply runs the agent with the pre-computed attacks and returns all
        attacks that were available for this task pair.
        """
        # Create task couple ID (matching the format used in other parts of the system)
        task_couple_id = f"{benign_task.id}:{malicious_task.id}"

        # Get pre-computed attacks for this task couple
        attacks_for_task = self._config.attacks_by_task.get(task_couple_id, {})

        if not attacks_for_task:
            logfire.warning(
                f"No pre-computed attacks found for task couple {task_couple_id}",
                available_tasks=list(self._config.attacks_by_task.keys()),
            )

        # Simply run the agent with the attacks
        end_state = None

        # Run the agent iterator to completion
        async for state in agent.iter(
            environment=environment,
            env_state=env_state,
            user_prompt=benign_task.prompt,
            message_history=[*message_history, *(benign_task.message_history or [])],
            toolsets=toolsets,
            usage_limits=usage_limits,
            attacks=attacks_for_task,
            instrument=instrument,
        ):
            if isinstance(state, EndState):
                end_state = state
                break

        if not isinstance(end_state, EndState):
            raise RuntimeError("Agent iteration completed without reaching EndState")

        # Return the end state and all attacks that were available for this task
        return end_state, attacks_for_task


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

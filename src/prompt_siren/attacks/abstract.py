# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Abstract base protocol for attack implementations.

Attacks generate injection payloads that attempt to make agents complete
malicious goals while ostensibly working on benign tasks.
"""

import abc
from collections.abc import Sequence
from typing import ClassVar, Protocol, TypeVar

from pydantic import BaseModel

from ..tasks import TaskCouple
from ..types import InjectionAttack
from .executor import RolloutExecutor
from .results import AttackResults

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class AbstractAttack(Protocol[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Protocol for attack implementations.

    Attacks operate on a batch of task couples using a RolloutExecutor that
    handles infrastructure concerns (environment lifecycle, concurrency,
    persistence, telemetry).

    There are two typical implementation patterns:

    1. Simple attacks (extend SimpleAttackBase):
       - Discover injection points once per couple
       - Generate a single attack payload per injection point
       - Execute one rollout per couple

    2. Batch-optimizing attacks (implement run() directly):
       - Discover injection points across the batch
       - Sample multiple attack candidates per checkpoint
       - Execute many rollouts, using rewards to update a policy
       - Return the best attacks found

    Attributes:
        name: Unique identifier for this attack type (used in registry)
    """

    name: ClassVar[str]

    @property
    def config(self) -> BaseModel:
        """Returns the configuration of the attack.

        It has to be a property method and not an attribute as otherwise
        Python's type system breaks.

        Returns:
            The attack's configuration as a Pydantic BaseModel
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def run(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> AttackResults[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Execute the attack strategy across a batch of task couples.

        The attack uses the executor to:
        1. Discover injection points (checkpoints) for each couple
        2. Execute rollouts from checkpoints with specific attack payloads
        3. Release checkpoint resources when done

        Simple attacks typically:
        - Discover all injection points
        - Generate one attack per checkpoint
        - Execute one rollout per couple
        - Return results

        Batch-optimizing attacks (e.g., GRPO) typically:
        - Discover all injection points
        - Iterate: sample attacks, execute rollouts, update policy
        - Track best attacks per couple
        - Return best results

        Args:
            couples: The task couples to attack. Each couple pairs a benign
                task (the cover) with a malicious task (the goal to inject).
            executor: The rollout executor for checkpoint discovery and
                rollout execution. Handles environment lifecycle, concurrency,
                persistence, and telemetry.

        Returns:
            AttackResults containing the final state, attacks, and evaluations
            for each couple that was successfully attacked.
        """
        raise NotImplementedError()

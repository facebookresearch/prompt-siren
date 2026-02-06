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
       - Sample multiple attack candidates per environment snapshot
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
        1. Discover injection points (environment snapshots) for each couple
        2. Execute rollouts from snapshots with specific attack payloads
        3. Release snapshot resources when done

        Simple attacks typically:
        - Discover all injection points
        - Generate one attack per snapshot
        - Execute one rollout per couple
        - Return results

        Batch-optimizing attacks (e.g., GRPO) typically:
        - Discover all injection points
        - Iterate: sample attacks, execute rollouts, update policy
        - Track best attacks per couple
        - Return best results

        Stateful attacks can use the executor's attack state methods to
        save/load optimization state for resumability:
        - executor.save_attack_state() to persist after each round/iteration
        - executor.load_latest_attack_state() to resume from last saved state

        Per-sample attacks can use ``executor.resume_info`` to discover
        which couples already have results and skip them.  Batch-optimizing
        attacks may ignore ``resume_info`` and rely entirely on their own
        saved state.

        Args:
            couples: The task couples to attack. Each couple pairs a benign
                task (the cover) with a malicious task (the goal to inject).
            executor: The rollout executor for snapshot discovery and
                rollout execution. Handles environment lifecycle, concurrency,
                persistence, and telemetry.

        Returns:
            AttackResults containing the final state, attacks, and evaluations
            for each couple that was successfully attacked.
        """
        raise NotImplementedError()

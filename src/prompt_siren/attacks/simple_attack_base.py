# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Base class for simple attacks that generate one attack per injection point.

Simple attacks follow a standard flow:
1. Discover injection points for all couples
2. Generate a single attack payload per checkpoint
3. Execute one rollout per couple
4. Return results

This base class handles the orchestration, so subclasses only need to
implement the attack generation logic via `generate_attack()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..tasks import TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict, InjectionVectorID
from .executor import InjectionCheckpoint, RolloutExecutor, RolloutRequest
from .results import AttackResults, CoupleAttackResult

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass(frozen=True)
class InjectionContext(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Context available when generating an attack for an injection point.

    Provides all information an attack might need to generate appropriate
    payloads for the injection vectors.

    Attributes:
        couple: The task couple being attacked
        available_vectors: List of injection vector IDs available at this point
        agent_name: Name of the agent (useful for model-specific attacks)
        checkpoint: The full checkpoint (for advanced attacks needing state access)
    """

    couple: TaskCouple[EnvStateT]
    """The task couple being attacked"""

    available_vectors: list[InjectionVectorID]
    """List of injection vector IDs available for attack"""

    agent_name: str
    """Name of the agent, useful for model-specific attack generation"""

    checkpoint: InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The full checkpoint for advanced attacks needing direct state access"""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> InjectionContext[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Create an InjectionContext from a checkpoint.

        Args:
            checkpoint: The checkpoint to extract context from

        Returns:
            InjectionContext with information from the checkpoint
        """
        return cls(
            couple=checkpoint.couple,
            available_vectors=checkpoint.available_vectors,
            agent_name=checkpoint.agent_name,
            checkpoint=checkpoint,
        )

    @property
    def malicious_goal(self) -> str:
        """Convenience accessor for the malicious task's goal."""
        return self.couple.malicious.goal

    @property
    def benign_prompt(self) -> str | list:
        """Convenience accessor for the benign task's prompt."""
        return self.couple.benign.prompt


class SimpleAttackBase(ABC, Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Base class for attacks that generate one attack per injection point.

    Subclasses implement `generate_attack()` to produce payloads for each
    injection context. The base class handles:
    - Checkpoint discovery across the batch
    - Rollout execution
    - Checkpoint cleanup
    - Result aggregation

    Example:
        @dataclass(frozen=True)
        class MyAttack(SimpleAttackBase[EnvStateT, str, str, StrContentAttack]):
            name: ClassVar[str] = "my_attack"
            _config: MyAttackConfig

            @property
            def config(self) -> MyAttackConfig:
                return self._config

            def generate_attack(
                self,
                context: InjectionContext[EnvStateT, str, str, StrContentAttack],
            ) -> InjectionAttacksDict[StrContentAttack]:
                payload = f"Malicious: {context.malicious_goal}"
                return {v: StrContentAttack(payload) for v in context.available_vectors}
    """

    @abstractmethod
    def generate_attack(
        self,
        context: InjectionContext[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> InjectionAttacksDict[InjectionAttackT]:
        """Generate attack payloads for a single injection point.

        Called once per injectable checkpoint. Should return attack payloads
        for each available injection vector.

        Args:
            context: Information about the injection point including the
                task couple, available vectors, and agent name

        Returns:
            Dictionary mapping injection vector IDs to attack payloads
        """
        ...

    async def run(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> AttackResults[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Execute the simple attack flow: discover → generate → execute.

        Args:
            couples: The task couples to attack
            executor: The rollout executor for checkpoint discovery and execution

        Returns:
            AttackResults containing results for each successfully attacked couple
        """
        # Discover injection points for all couples
        checkpoints = await executor.discover_injection_points(couples)

        try:
            # Build a map from couple_id to checkpoint and generated attacks
            couple_data: dict[str, tuple[InjectionCheckpoint, InjectionAttacksDict]] = {}
            requests: list[
                RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
            ] = []

            for checkpoint in checkpoints:
                if checkpoint.is_terminal:
                    # No injection point found for this couple
                    couple_data[checkpoint.couple.id] = (checkpoint, {})
                    continue

                context = InjectionContext.from_checkpoint(checkpoint)
                attacks = self.generate_attack(context)
                couple_data[checkpoint.couple.id] = (checkpoint, attacks)

                requests.append(
                    RolloutRequest(
                        checkpoint=checkpoint,
                        attacks=attacks,
                    )
                )

            # Execute all rollouts
            rollout_results_by_couple: dict[str, list] = {c.id: [] for c in couples}
            if requests:
                rollout_results = await executor.execute_from_checkpoints(requests)
                for result in rollout_results:
                    couple_id = result.request.checkpoint.couple.id
                    rollout_results_by_couple[couple_id].append(result)

            # Build CoupleAttackResult for each couple
            couple_results: list[CoupleAttackResult] = []
            for couple in couples:
                checkpoint, attacks = couple_data.get(couple.id, (None, {}))
                couple_results.append(
                    CoupleAttackResult(
                        couple=couple,
                        rollout_results=rollout_results_by_couple.get(couple.id, []),
                        generated_attacks=attacks,
                    )
                )

            return AttackResults(couple_results=couple_results)

        finally:
            # Always release checkpoints to free resources
            await executor.release_checkpoints(checkpoints)

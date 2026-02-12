# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Base class for simple attacks that generate one attack per injection point.

Simple attacks follow a standard flow:
1. Discover injection points for all couples
2. Generate a single attack payload per environment snapshot
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
from .executor import EnvironmentSnapshotAtInjection, RolloutExecutor, RolloutRequest
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
        snapshot: The full environment snapshot (for advanced attacks needing state access)
    """

    couple: TaskCouple[EnvStateT]
    """The task couple being attacked"""

    available_vectors: list[InjectionVectorID]
    """List of injection vector IDs available for attack"""

    agent_name: str
    """Name of the agent, useful for model-specific attack generation"""

    snapshot: EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The full environment snapshot for advanced attacks needing direct state access"""

    @classmethod
    def from_snapshot(
        cls,
        snapshot: EnvironmentSnapshotAtInjection[
            EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT
        ],
    ) -> InjectionContext[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Create an InjectionContext from an environment snapshot.

        Args:
            snapshot: The environment snapshot to extract context from

        Returns:
            InjectionContext with information from the snapshot
        """
        return cls(
            couple=snapshot.couple,
            available_vectors=snapshot.available_vectors,
            agent_name=snapshot.agent_name,
            snapshot=snapshot,
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
    - Environment snapshot discovery across the batch
    - Rollout execution
    - Snapshot cleanup
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

        Called once per injectable environment snapshot. Should return attack payloads
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

        On resume, uses ``executor.resume_info`` to skip couples that
        already have results from a prior run.

        Args:
            couples: The task couples to attack
            executor: The rollout executor for snapshot discovery and execution

        Returns:
            AttackResults containing results for each successfully attacked couple
        """
        # On resume, filter out already-completed couples
        if executor.resume_info is not None:
            couples = executor.resume_info.filter_remaining(couples)
            if not couples:
                return AttackResults(couple_results=[])

        # Discover injection points for all couples
        snapshots = await executor.discover_injection_points(couples)

        try:
            # Build a map from couple_id to snapshot and generated attacks
            couple_data: dict[str, tuple[EnvironmentSnapshotAtInjection, InjectionAttacksDict]] = {}
            requests: list[
                RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
            ] = []

            for snapshot in snapshots:
                if snapshot.is_terminal:
                    # No injection point found for this couple
                    couple_data[snapshot.couple.id] = (snapshot, {})
                    continue

                context = InjectionContext.from_snapshot(snapshot)
                attacks = self.generate_attack(context)
                couple_data[snapshot.couple.id] = (snapshot, attacks)

                requests.append(
                    RolloutRequest(
                        snapshot=snapshot,
                        attacks=attacks,
                    )
                )

            # Execute all rollouts
            rollout_results_by_couple: dict[str, list] = {c.id: [] for c in couples}
            if requests:
                rollout_results = await executor.execute_from_snapshots(requests)
                for result in rollout_results:
                    couple_id = result.request.snapshot.couple.id
                    rollout_results_by_couple[couple_id].append(result)

            # Build CoupleAttackResult for each couple
            couple_results: list[CoupleAttackResult] = []
            for couple in couples:
                _, attacks = couple_data.get(couple.id, (None, {}))
                couple_results.append(
                    CoupleAttackResult(
                        couple=couple,
                        rollout_results=rollout_results_by_couple.get(couple.id, []),
                        generated_attacks=attacks,
                    )
                )

            return AttackResults(couple_results=couple_results)

        finally:
            # Always release snapshots to free resources
            await executor.release_snapshots(snapshots)

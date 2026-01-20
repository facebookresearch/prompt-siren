# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Result types for batch attack execution.

This module provides result types that aggregate outcomes from attack runs
across a batch of task couples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TYPE_CHECKING, TypeVar

from ..tasks import TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict, TaskCoupleID

if TYPE_CHECKING:
    from .executor import RolloutResult

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass
class CoupleAttackResult(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Result of attacking a single task couple.

    For simple attacks, there's typically one rollout result.
    For batch-optimizing attacks, there may be multiple rollouts per couple,
    and the attack can select the best one.

    Attributes:
        couple: The task couple that was attacked
        rollout_results: All rollout results for this couple
        generated_attacks: The attack payloads that were generated
    """

    couple: TaskCouple[EnvStateT]
    """The task couple that was attacked"""

    rollout_results: list[RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]
    """All rollout results for this couple (may be empty if no injection points found)"""

    generated_attacks: InjectionAttacksDict[InjectionAttackT]
    """The attack payloads that were generated"""

    @property
    def couple_id(self) -> TaskCoupleID:
        """Get the couple ID."""
        return self.couple.id

    def best_result(
        self,
    ) -> RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None:
        """Get the best rollout result based on malicious score.

        Returns:
            The rollout with the highest malicious eval score, or None if no rollouts
        """
        if not self.rollout_results:
            return None

        def score(r: RolloutResult) -> float:
            results = r.malicious_eval.results
            if not results:
                return 0.0
            return sum(results.values()) / len(results)

        return max(self.rollout_results, key=score)


@dataclass
class AttackResults(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Aggregated results from an attack run across a batch of task couples.

    Provides convenient access to results and helper methods for converting
    between formats.

    Attributes:
        couple_results: List of results, one per task couple
    """

    couple_results: list[
        CoupleAttackResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ] = field(default_factory=list)
    """List of results, one per task couple"""

    def to_evaluation_dict(self) -> dict[str, dict[str, float]]:
        """Convert to the format expected by hydra_app.

        Returns a dictionary mapping task IDs (both benign and malicious)
        to their evaluation results. Uses the best rollout result for each couple.

        Returns:
            Dictionary mapping task_id -> {evaluator_name -> score}
        """
        formatted: dict[str, dict[str, float]] = {}
        for couple_result in self.couple_results:
            best = couple_result.best_result()
            if best:
                formatted[best.benign_eval.task_id] = best.benign_eval.results
                formatted[best.malicious_eval.task_id] = best.malicious_eval.results
            else:
                # No rollout results - return empty evaluations
                formatted[couple_result.couple.benign.id] = {}
                formatted[couple_result.couple.malicious.id] = {}
        return formatted

    def __len__(self) -> int:
        """Return the number of couple results."""
        return len(self.couple_results)

    def __iter__(self):
        """Iterate over couple results."""
        return iter(self.couple_results)

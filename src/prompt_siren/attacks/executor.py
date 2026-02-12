# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Rollout executor types and protocol for batch attack execution.

This module provides the interface for attacks to request rollouts without
managing infrastructure concerns like environment lifecycle, concurrency,
persistence, and telemetry.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, runtime_checkable, TypeVar

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from ..agents.states import EndState, InjectableModelRequestState
from ..tasks import EvaluationResult, TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict, InjectionVectorID

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)
AttackStateT = TypeVar("AttackStateT", bound=BaseModel)


@dataclass(frozen=True)
class ResumeInfo(Generic[EnvStateT]):
    """Resume information provided by the executor to attacks.

    Contains per-couple completion data from prior runs so that attacks
    can decide which couples still need processing.  Simple per-sample
    attacks use ``filter_remaining()`` to skip completed couples.
    Batch-optimizing attacks (like successive halving) may ignore this
    entirely and manage their own resume via
    ``executor.load_latest_attack_state()``.

    Attributes:
        completed_ids: Set of task/couple IDs that already have at
            least one recorded run in the job's persistence layer.
    """

    completed_ids: frozenset[str]

    def filter_remaining(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
    ) -> list[TaskCouple[EnvStateT]]:
        """Return only couples that have no recorded runs yet.

        Args:
            couples: The full list of task couples.

        Returns:
            Couples whose IDs are not in ``completed_ids``.
        """
        return [c for c in couples if c.id not in self.completed_ids]


@dataclass(frozen=True)
class EnvironmentSnapshotAtInjection(
    Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
):
    """A saved environment state at an injectable point, ready for attack injection.

    Represents a point in agent execution where injection vectors are available.
    The executor can restore execution to this point for multiple rollouts with
    different attack payloads.

    Attributes:
        couple: The task couple being executed
        injectable_state: The agent state at the injection point (None if terminal)
        available_vectors: List of injection vector IDs available at this point
        agent_name: Name of the agent for template rendering
    """

    couple: TaskCouple[EnvStateT]
    """The task couple being executed"""

    injectable_state: (
        InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None
    )
    """The agent state at the injection point, or None if execution completed without injection"""

    available_vectors: list[InjectionVectorID]
    """List of injection vector IDs available for attack at this snapshot"""

    agent_name: str
    """Name of the agent, useful for attack generation"""

    _snapshot_id: str = ""
    """Internal ID for the executor to track snapshot resources"""

    @property
    def is_terminal(self) -> bool:
        """Returns True if execution completed without finding an injection point."""
        return self.injectable_state is None


@dataclass(frozen=True)
class RolloutRequest(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """A request to execute a single agent rollout with specific attacks.

    Attributes:
        snapshot: The environment snapshot to resume execution from
        attacks: The attack payloads to inject at the snapshot's vectors
        metadata: Optional metadata for tracking (e.g., iteration number, sample ID)
    """

    snapshot: EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The environment snapshot to resume execution from"""

    attacks: InjectionAttacksDict[InjectionAttackT]
    """The attack payloads to inject at the snapshot's injection vectors"""

    metadata: dict[str, Any] | None = None
    """Optional metadata for the attack to track (e.g., which generation this belongs to)"""


@dataclass(frozen=True)
class RolloutResult(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Result of a single rollout execution.

    Contains the final state, evaluations, and full trajectory for analysis
    by batch-optimizing attacks.

    Attributes:
        request: The original rollout request (echoed back for correlation)
        end_state: The final execution state after running to completion
        benign_eval: Evaluation result for the benign task
        malicious_eval: Evaluation result for the malicious task
        messages: Full message history (trajectory) for RL algorithms
        usage: Token usage information
    """

    request: RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The original request, echoed back for correlation"""

    end_state: EndState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The final execution state after running to completion"""

    benign_eval: EvaluationResult
    """Evaluation result for the benign task"""

    malicious_eval: EvaluationResult
    """Evaluation result for the malicious task"""

    messages: list[ModelMessage] = field(default_factory=list)
    """Full message history (trajectory) for RL algorithms"""

    usage: RunUsage = field(default_factory=RunUsage)
    """Token usage information for this rollout"""


@runtime_checkable
class RolloutExecutor(Protocol[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Protocol for executing rollouts on behalf of attacks.

    The executor abstracts away infrastructure concerns:
    - Environment lifecycle (create_task_context, copy_env_state)
    - Concurrency control
    - Result persistence
    - Telemetry/spans
    - Attack state management (save/load attack-internal optimization state)

    Attacks use the executor to:
    1. Discover injection points across the batch
    2. Execute rollouts from snapshots with specific attack payloads
    3. Release snapshot resources when done
    4. Optionally save/load attack-internal state for resumability
    """

    @abstractmethod
    async def discover_injection_points(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        max_concurrency: int | None = None,
    ) -> list[
        EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ]:
        """Run each couple until first injectable point, return snapshots.

        For each couple:
        1. Creates task context
        2. Runs agent until an injectable state is found or execution completes
        3. Snapshots environment state (for snapshottable environments)
        4. Returns snapshot with available injection vectors

        Args:
            couples: The task couples to discover injection points for
            max_concurrency: Maximum parallel discoveries (None = use default)

        Returns:
            List of snapshots, one per couple. Snapshots with is_terminal=True
            indicate execution completed without finding an injection point.
        """
        ...

    @abstractmethod
    async def execute_from_snapshots(
        self,
        requests: Sequence[RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]],
        max_concurrency: int | None = None,
    ) -> list[RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Execute rollouts from saved environment snapshots with specified attacks.

        For each request:
        1. Restores environment state from snapshot
        2. Restores agent state (injectable state with fresh env_state)
        3. Applies attacks and runs to completion
        4. Evaluates both benign and malicious tasks
        5. Persists results if configured

        The same snapshot can be used multiple times with different attacks,
        enabling batch-optimizing attacks to sample many candidates.

        Args:
            requests: Rollouts to execute. Can include the same snapshot
                multiple times with different attacks.
            max_concurrency: Maximum parallel rollouts (None = use default)

        Returns:
            Results in the same order as requests
        """
        ...

    @abstractmethod
    async def release_snapshots(
        self,
        snapshots: Sequence[
            EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        ],
    ) -> None:
        """Release resources held by environment snapshots.

        Should be called when the attack is done using snapshots to free
        any saved environment states or other resources.

        Args:
            snapshots: Snapshots to release
        """
        ...

    @property
    @abstractmethod
    def job_dir(self) -> Path | None:
        """Return the job directory for output logging, if available.

        Returns:
            Path to the job directory, or None if not configured.
        """
        ...

    @property
    def resume_info(self) -> ResumeInfo[EnvStateT] | None:
        """Return resume information from prior runs, if available.

        Attacks use this to decide which couples still need processing.
        Per-sample attacks typically call ``resume_info.filter_remaining()``
        to skip already-completed couples.  Batch-optimizing attacks may
        ignore this and manage their own resume via
        ``load_latest_attack_state()``.

        Returns:
            ResumeInfo with completed couple IDs, or None if not resuming
            or persistence is not configured.
        """
        return None

    def save_attack_state(
        self,
        key: str,
        state: BaseModel,
        attack_name: str,
    ) -> Path:
        """Persist attack-internal state to the job directory.

        Attacks call this to save their optimization state (e.g., which
        round of successive halving, which strategies survived, policy
        weights for RL attacks).

        Args:
            key: Unique key for this state snapshot (e.g., "round_0002",
                 "iteration_015"). Keys are scoped to the attack_name.
            state: Pydantic BaseModel to serialize.
            attack_name: Name of the attack (for namespacing).

        Returns:
            Path where state was saved.

        Raises:
            ValueError: If no job directory is configured.
        """
        raise NotImplementedError("Attack state persistence not supported by this executor")

    def load_attack_state(
        self,
        key: str,
        state_type: type[AttackStateT],
        attack_name: str,
    ) -> AttackStateT | None:
        """Load previously saved attack state.

        Args:
            key: The key used when saving.
            state_type: Pydantic model class to deserialize into.
            attack_name: Name of the attack.

        Returns:
            Deserialized state, or None if no state exists for this key.
        """
        return None

    def load_latest_attack_state(
        self,
        state_type: type[AttackStateT],
        attack_name: str,
    ) -> tuple[str, AttackStateT] | None:
        """Load the most recently saved attack state.

        Useful for resume: "give me whatever state I last saved."

        Args:
            state_type: Pydantic model class to deserialize into.
            attack_name: Name of the attack.

        Returns:
            Tuple of (key, state) or None if no state exists.
        """
        return None

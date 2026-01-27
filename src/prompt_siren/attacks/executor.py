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
from typing import Any, Generic, Protocol, runtime_checkable, TypeVar

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from ..agents.states import EndState, InjectableModelRequestState
from ..tasks import EvaluationResult, TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict, InjectionVectorID

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass(frozen=True)
class InjectionCheckpoint(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """A saved state at an injectable point, ready for attack injection.

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
    """List of injection vector IDs available for attack at this checkpoint"""

    agent_name: str
    """Name of the agent, useful for attack generation"""

    _checkpoint_id: str = ""
    """Internal ID for the executor to track checkpoint resources"""

    @property
    def is_terminal(self) -> bool:
        """Returns True if execution completed without finding an injection point."""
        return self.injectable_state is None


@dataclass(frozen=True)
class RolloutRequest(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """A request to execute a single agent rollout with specific attacks.

    Attributes:
        checkpoint: The checkpoint to resume execution from
        attacks: The attack payloads to inject at the checkpoint's vectors
        metadata: Optional metadata for tracking (e.g., iteration number, sample ID)
    """

    checkpoint: InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The checkpoint to resume execution from"""

    attacks: InjectionAttacksDict[InjectionAttackT]
    """The attack payloads to inject at the checkpoint's injection vectors"""

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

    Attacks use the executor to:
    1. Discover injection points across the batch
    2. Execute rollouts from checkpoints with specific attack payloads
    3. Release checkpoint resources when done
    """

    @abstractmethod
    async def discover_injection_points(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        max_concurrency: int | None = None,
    ) -> list[InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Run each couple until first injectable point, return checkpoints.

        For each couple:
        1. Creates task context
        2. Runs agent until an injectable state is found or execution completes
        3. Snapshots environment state (for snapshottable environments)
        4. Returns checkpoint with available injection vectors

        Args:
            couples: The task couples to discover injection points for
            max_concurrency: Maximum parallel discoveries (None = use default)

        Returns:
            List of checkpoints, one per couple. Checkpoints with is_terminal=True
            indicate execution completed without finding an injection point.
        """
        ...

    @abstractmethod
    async def execute_from_checkpoints(
        self,
        requests: Sequence[RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]],
        max_concurrency: int | None = None,
    ) -> list[RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Execute rollouts from saved checkpoints with specified attacks.

        For each request:
        1. Restores environment state from checkpoint
        2. Restores agent state (injectable state with fresh env_state)
        3. Applies attacks and runs to completion
        4. Evaluates both benign and malicious tasks
        5. Persists results if configured

        The same checkpoint can be used multiple times with different attacks,
        enabling batch-optimizing attacks to sample many candidates.

        Args:
            requests: Rollouts to execute. Can include the same checkpoint
                multiple times with different attacks.
            max_concurrency: Maximum parallel rollouts (None = use default)

        Returns:
            Results in the same order as requests
        """
        ...

    @abstractmethod
    async def release_checkpoints(
        self,
        checkpoints: Sequence[
            InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        ],
    ) -> None:
        """Release resources held by checkpoints.

        Should be called when the attack is done using checkpoints to free
        any saved environment states or other resources.

        Args:
            checkpoints: Checkpoints to release
        """
        ...

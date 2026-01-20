# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Default implementation of the RolloutExecutor protocol.

This module provides the standard executor that handles:
- Checkpoint discovery (running agents to injection points)
- Environment state snapshotting and restoration
- Rollout execution with attack injection
- Concurrency control
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import cast, Generic, TypeVar
from uuid import uuid4

from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ..agents.abstract import AbstractAgent
from ..agents.states import EndState, InjectableModelRequestState
from ..environments.abstract import AbstractEnvironment, Snapshottable
from ..tasks import TaskCouple, TaskResult
from ..tools_utils import run_tool_history
from ..types import InjectionAttack, InjectionVectorID
from .attack_utils import run_until_injectable
from .executor import (
    InjectionCheckpoint,
    RolloutRequest,
    RolloutResult,
)

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass
class _CheckpointData(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Internal storage for checkpoint resources.

    Stores all data needed to restore execution to an injection point.
    """

    couple: TaskCouple[EnvStateT]
    """The task couple for this checkpoint"""

    injectable_state: (
        InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None
    )
    """The agent state at the injection point"""

    saved_env_state: EnvStateT | None
    """Snapshot of environment state (for snapshottable environments)"""

    message_history: list[ModelMessage]
    """Message history up to the checkpoint (for non-snapshottable replay)"""

    agent_name: str
    """Name of the agent"""

    available_vectors: list[InjectionVectorID]
    """Available injection vectors at this checkpoint"""


def _setup_history(system_prompt: str | None) -> list[ModelMessage]:
    """Create initial message history with optional system prompt."""
    if system_prompt is not None:
        return [ModelRequest([SystemPromptPart(system_prompt)])]
    return []


def _extract_vector_ids(
    parts: list,
) -> list[InjectionVectorID]:
    """Extract vector IDs from injectable model request parts."""
    vectors: list[InjectionVectorID] = []
    for part in parts:
        if hasattr(part, "vector_ids"):
            vectors.extend(part.vector_ids)
    return vectors


@dataclass
class DefaultRolloutExecutor(
    Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
):
    """Default implementation of RolloutExecutor.

    Handles checkpoint discovery, environment restoration, and rollout execution
    for both snapshottable and non-snapshottable environments.

    Attributes:
        agent: The agent to execute
        environment: The environment for execution
        toolsets: Available tools
        system_prompt: Optional system prompt for message history
        usage_limits: Constraints on model usage
        max_concurrency: Default maximum concurrent operations
        instrument: Instrumentation settings for telemetry
    """

    agent: AbstractAgent
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    toolsets: Sequence[AbstractToolset[EnvStateT]]
    system_prompt: str | None
    usage_limits: UsageLimits
    max_concurrency: int | None = 1
    instrument: InstrumentationSettings | bool | None = None

    # Internal checkpoint storage
    _checkpoints: dict[
        str, _CheckpointData[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ] = field(default_factory=dict)

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
        4. Stores checkpoint data for later restoration

        Args:
            couples: The task couples to discover injection points for
            max_concurrency: Maximum parallel discoveries

        Returns:
            List of checkpoints, one per couple
        """
        concurrency = max_concurrency or self.max_concurrency or 1
        semaphore = asyncio.BoundedSemaphore(concurrency)
        message_history = _setup_history(self.system_prompt)

        async def discover_one(
            couple: TaskCouple[EnvStateT],
        ) -> InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
            async with semaphore:
                async with self.environment.create_task_context(couple) as env_state:
                    # Create initial state
                    benign_task = couple.benign
                    initial_state = self.agent.create_initial_request_state(
                        self.environment,
                        env_state,
                        benign_task.prompt,
                        message_history=[*message_history, *(benign_task.message_history or [])],
                    )

                    # Run until injectable or end
                    state = await run_until_injectable(
                        self.agent,
                        initial_state,
                        toolsets=self.toolsets,
                        usage_limits=self.usage_limits,
                        instrument=self.instrument,
                    )

                    if isinstance(state, EndState):
                        # No injection point found - terminal checkpoint
                        checkpoint_id = str(uuid4())
                        self._checkpoints[checkpoint_id] = _CheckpointData(
                            couple=couple,
                            injectable_state=None,
                            saved_env_state=None,
                            message_history=list(state.run_ctx.messages),
                            agent_name=self.agent.get_agent_name(),
                            available_vectors=[],
                        )
                        return InjectionCheckpoint(
                            couple=couple,
                            injectable_state=None,
                            available_vectors=[],
                            agent_name=self.agent.get_agent_name(),
                            _checkpoint_id=checkpoint_id,
                        )

                    # Found injectable state - create checkpoint
                    checkpoint_id = str(uuid4())

                    # Snapshot environment state for snapshottable environments
                    saved_env: EnvStateT | None = None
                    if isinstance(self.environment, Snapshottable):
                        snapshotable_env = cast(Snapshottable[EnvStateT], self.environment)
                        saved_env = await snapshotable_env.copy_env_state(state.run_ctx.deps)

                    # Extract available vectors
                    available_vectors = _extract_vector_ids(state.injectable_model_request_parts)

                    # Store checkpoint data
                    self._checkpoints[checkpoint_id] = _CheckpointData(
                        couple=couple,
                        injectable_state=state,
                        saved_env_state=saved_env,
                        message_history=list(state.run_ctx.messages),
                        agent_name=self.agent.get_agent_name(),
                        available_vectors=available_vectors,
                    )

                    return InjectionCheckpoint(
                        couple=couple,
                        injectable_state=state,
                        available_vectors=available_vectors,
                        agent_name=self.agent.get_agent_name(),
                        _checkpoint_id=checkpoint_id,
                    )

        # Run discovery for all couples
        checkpoints = await asyncio.gather(*[discover_one(c) for c in couples])
        return list(checkpoints)

    async def execute_from_checkpoints(
        self,
        requests: Sequence[RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]],
        max_concurrency: int | None = None,
    ) -> list[RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Execute rollouts from saved checkpoints with specified attacks.

        For each request:
        1. Restores environment state from checkpoint
        2. Restores agent state with fresh env_state
        3. Applies attacks and runs to completion
        4. Evaluates both tasks

        Args:
            requests: Rollouts to execute
            max_concurrency: Maximum parallel rollouts

        Returns:
            Results in the same order as requests
        """
        concurrency = max_concurrency or self.max_concurrency or 1
        semaphore = asyncio.BoundedSemaphore(concurrency)

        async def execute_one(
            request: RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        ) -> RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
            async with semaphore:
                checkpoint = request.checkpoint
                checkpoint_data = self._checkpoints.get(checkpoint._checkpoint_id)

                if checkpoint_data is None:
                    raise ValueError(f"Checkpoint {checkpoint._checkpoint_id} not found")

                couple = checkpoint_data.couple
                injectable_state = checkpoint_data.injectable_state

                if injectable_state is None:
                    raise ValueError(
                        "Cannot execute from terminal checkpoint (no injectable state)"
                    )

                # Execute within task context
                async with self.environment.create_task_context(couple) as base_env_state:
                    # Restore environment state
                    if checkpoint_data.saved_env_state is not None and isinstance(
                        self.environment, Snapshottable
                    ):
                        # Copy from saved snapshot
                        snapshotable_env = cast(Snapshottable[EnvStateT], self.environment)
                        restored_env = await snapshotable_env.copy_env_state(
                            checkpoint_data.saved_env_state
                        )
                    else:
                        # Replay tools to restore state
                        fake_ctx: RunContext[EnvStateT] = RunContext(
                            deps=base_env_state,
                            model=TestModel(),
                            usage=RunUsage(),
                            messages=list(checkpoint_data.message_history),
                        )
                        replayed_ctx = await run_tool_history(fake_ctx, self.toolsets)
                        restored_env = replayed_ctx.deps

                    # Capture pre-state for evaluation
                    try:
                        pre_env_state: EnvStateT | None = deepcopy(restored_env)
                    except TypeError:
                        pre_env_state = None

                    # Create new run context with restored env_state
                    new_run_ctx: RunContext[EnvStateT] = replace(
                        injectable_state.run_ctx,
                        deps=restored_env,
                        usage=RunUsage(),  # Fresh usage for this rollout
                    )

                    # Create new injectable state with restored context
                    restored_state = InjectableModelRequestState(
                        run_ctx=new_run_ctx,
                        environment=self.environment,
                        injectable_model_request_parts=injectable_state.injectable_model_request_parts,
                        _previous_state=None,  # No previous state for restored execution
                    )

                    # Apply attacks and advance to next state
                    state = await self.agent.next_state(
                        current_state=restored_state,
                        toolsets=self.toolsets,
                        usage_limits=self.usage_limits,
                        attacks=request.attacks,
                        instrument=self.instrument,
                    )

                    # Continue execution to completion
                    while not isinstance(state, EndState):
                        state = await self.agent.next_state(
                            current_state=state,
                            toolsets=self.toolsets,
                            usage_limits=self.usage_limits,
                            attacks=request.attacks,
                            instrument=self.instrument,
                        )

                    # Evaluate both tasks
                    task_result: TaskResult[EnvStateT] = TaskResult(
                        run_context=state.run_ctx,
                        pre_env_state=pre_env_state,
                        task=couple,
                    )
                    benign_eval, malicious_eval = await couple.evaluate(task_result)

                    return RolloutResult(
                        request=request,
                        end_state=state,
                        benign_eval=benign_eval,
                        malicious_eval=malicious_eval,
                        messages=list(state.run_ctx.messages),
                        usage=state.run_ctx.usage,
                    )

        # Execute all rollouts
        results = await asyncio.gather(*[execute_one(r) for r in requests])
        return list(results)

    async def release_checkpoints(
        self,
        checkpoints: Sequence[
            InjectionCheckpoint[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        ],
    ) -> None:
        """Release resources held by checkpoints.

        Removes checkpoint data from internal storage, allowing saved
        environment states to be garbage collected.

        Args:
            checkpoints: Checkpoints to release
        """
        for checkpoint in checkpoints:
            if checkpoint._checkpoint_id:
                self._checkpoints.pop(checkpoint._checkpoint_id, None)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Default implementation of the RolloutExecutor protocol.

This module provides the standard executor that handles:
- Environment snapshot discovery (running agents to injection points)
- Environment state snapshotting and restoration
- Rollout execution with attack injection
- Concurrency control
- Attack state persistence for resumable attacks
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import cast, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ..agents.abstract import AbstractAgent
from ..agents.states import EndState, InjectableModelRequestState
from ..environments.abstract import AbstractEnvironment, Snapshottable
from ..job import JobPersistence
from ..tasks import TaskCouple, TaskResult
from ..telemetry.formatted_span import formatted_span
from ..tools_utils import run_tool_history
from ..types import InjectionAttack, InjectionVectorID
from .attack_utils import run_until_injectable
from .executor import (
    EnvironmentSnapshotAtInjection,
    ResumeInfo,
    RolloutRequest,
    RolloutResult,
)

logger = logging.getLogger(__name__)

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)
AttackStateT = TypeVar("AttackStateT", bound=BaseModel)


@dataclass
class _SnapshotData(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Internal storage for environment snapshot resources.

    Stores all data needed to restore execution to an injection point.
    """

    couple: TaskCouple[EnvStateT]
    """The task couple for this snapshot"""

    injectable_state: (
        InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None
    )
    """The agent state at the injection point"""

    saved_env_state: EnvStateT | None
    """Snapshot of environment state (for snapshottable environments)"""

    message_history: list[ModelMessage]
    """Message history up to the snapshot (for non-snapshottable replay)"""

    agent_name: str
    """Name of the agent"""

    available_vectors: list[InjectionVectorID]
    """Available injection vectors at this snapshot"""


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

    Handles environment snapshot discovery, environment restoration, and rollout
    execution for both snapshottable and non-snapshottable environments.

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
    persistence: JobPersistence | None = None
    completed_task_ids: frozenset[str] = frozenset()
    job_dir: Path | None = None

    @property
    def resume_info(self) -> ResumeInfo[EnvStateT] | None:
        """Return resume information from prior runs, if available."""
        if not self.completed_task_ids:
            return None
        return ResumeInfo(completed_ids=self.completed_task_ids)

    # Internal snapshot storage
    _snapshots: dict[str, _SnapshotData[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]] = (
        field(default_factory=dict)
    )

    async def discover_injection_points(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        max_concurrency: int | None = None,
    ) -> list[
        EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ]:
        """Run each couple until first injectable point, return environment snapshots.

        For each couple:
        1. Creates task context
        2. Runs agent until an injectable state is found or execution completes
        3. Snapshots environment state (for snapshottable environments)
        4. Stores snapshot data for later restoration

        Args:
            couples: The task couples to discover injection points for
            max_concurrency: Maximum parallel discoveries

        Returns:
            List of snapshots, one per couple
        """
        concurrency = max_concurrency or self.max_concurrency or 1
        semaphore = asyncio.BoundedSemaphore(concurrency)
        message_history = _setup_history(self.system_prompt)

        async def discover_one(
            couple: TaskCouple[EnvStateT],
        ) -> EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
            async with semaphore:
                logger.info(
                    "[SNAPSHOT DISCOVERY] Looking for injection point for task '%s'...",
                    couple.id,
                )
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
                        # No injection point found - terminal snapshot
                        logger.info(
                            "[SNAPSHOT DISCOVERY] ✗ No injection point found for task '%s' (terminal state)",
                            couple.id,
                        )
                        snapshot_id = str(uuid4())
                        self._snapshots[snapshot_id] = _SnapshotData(
                            couple=couple,
                            injectable_state=None,
                            saved_env_state=None,
                            message_history=list(state.run_ctx.messages),
                            agent_name=self.agent.get_agent_name(),
                            available_vectors=[],
                        )
                        return EnvironmentSnapshotAtInjection(
                            couple=couple,
                            injectable_state=None,
                            available_vectors=[],
                            agent_name=self.agent.get_agent_name(),
                            _snapshot_id=snapshot_id,
                        )

                    # Found injectable state - create snapshot
                    snapshot_id = str(uuid4())

                    # Snapshot environment state for snapshottable environments
                    saved_env: EnvStateT | None = None
                    if isinstance(self.environment, Snapshottable):
                        snapshotable_env = cast(Snapshottable[EnvStateT], self.environment)
                        saved_env = await snapshotable_env.copy_env_state(state.run_ctx.deps)

                    # Extract available vectors
                    available_vectors = _extract_vector_ids(state.injectable_model_request_parts)

                    # Store snapshot data
                    self._snapshots[snapshot_id] = _SnapshotData(
                        couple=couple,
                        injectable_state=state,
                        saved_env_state=saved_env,
                        message_history=list(state.run_ctx.messages),
                        agent_name=self.agent.get_agent_name(),
                        available_vectors=available_vectors,
                    )
                    logger.info(
                        "[SNAPSHOT DISCOVERY] ✓ Discovered injection point for task '%s' with %d vectors",
                        couple.id,
                        len(available_vectors),
                    )
                    return EnvironmentSnapshotAtInjection(
                        couple=couple,
                        injectable_state=state,
                        available_vectors=available_vectors,
                        agent_name=self.agent.get_agent_name(),
                        _snapshot_id=snapshot_id,
                    )

        # Run discovery for all couples
        logger.info(
            "[SNAPSHOT DISCOVERY] Starting discovery for %d task(s)...",
            len(couples),
        )
        snapshots = await asyncio.gather(*[discover_one(c) for c in couples])
        injectable_count = sum(1 for s in snapshots if s.injectable_state is not None)
        logger.info(
            "[SNAPSHOT DISCOVERY] Discovery complete: %d/%d tasks have injection points",
            injectable_count,
            len(snapshots),
        )
        return list(snapshots)

    async def execute_from_snapshots(
        self,
        requests: Sequence[RolloutRequest[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]],
        max_concurrency: int | None = None,
    ) -> list[RolloutResult[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Execute rollouts from saved environment snapshots with specified attacks.

        For each request:
        1. Restores environment state from snapshot
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
                snapshot = request.snapshot
                snapshot_data = self._snapshots.get(snapshot._snapshot_id)

                if snapshot_data is None:
                    raise ValueError(f"Snapshot {snapshot._snapshot_id} not found")

                couple = snapshot_data.couple
                injectable_state = snapshot_data.injectable_state

                if injectable_state is None:
                    raise ValueError("Cannot execute from terminal snapshot (no injectable state)")

                logger.info(
                    "[ROLLOUT] Starting rollout for task '%s' with %d attack(s)",
                    couple.id,
                    len(request.attacks) if request.attacks else 0,
                )

                started_at = datetime.now()

                # Execute within task context
                async with self.environment.create_task_context(couple) as base_env_state:
                    # Use formatted_span for rollout telemetry (task span is created by attack)
                    with formatted_span(
                        "rollout {couple_id}",
                        couple_id=couple.id,
                        environment_name=self.environment.name,
                        agent_name=snapshot_data.agent_name,
                    ) as task_span:
                        # Restore environment state
                        if snapshot_data.saved_env_state is not None and isinstance(
                            self.environment, Snapshottable
                        ):
                            # Copy from saved snapshot
                            snapshotable_env = cast(Snapshottable[EnvStateT], self.environment)
                            restored_env = await snapshotable_env.copy_env_state(
                                snapshot_data.saved_env_state
                            )
                        else:
                            # Replay tools to restore state
                            fake_ctx: RunContext[EnvStateT] = RunContext(
                                deps=base_env_state,
                                model=TestModel(),
                                usage=RunUsage(),
                                messages=list(snapshot_data.message_history),
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

                        logger.info(
                            "[ROLLOUT] ✓ Completed rollout for task '%s' - benign: %s, malicious: %s",
                            couple.id,
                            benign_eval.results,
                            malicious_eval.results,
                        )

                        result = RolloutResult(
                            request=request,
                            end_state=state,
                            benign_eval=benign_eval,
                            malicious_eval=malicious_eval,
                            messages=list(state.run_ctx.messages),
                            usage=state.run_ctx.usage,
                        )

                        # Persist result incrementally if persistence is configured
                        if self.persistence:
                            self.persistence.save_couple_run(
                                couple=couple,
                                benign_eval=benign_eval,
                                malicious_eval=malicious_eval,
                                messages=list(state.run_ctx.messages),
                                usage=state.run_ctx.usage,
                                task_span=task_span,
                                started_at=started_at,
                                generated_attacks=request.attacks,
                            )

                        return result

        # Execute all rollouts
        logger.info("[ROLLOUT] Starting %d rollout(s)...", len(requests))
        results = await asyncio.gather(*[execute_one(r) for r in requests])
        # Count attacks where all malicious evaluators scored >= 0.5 as successful
        successful_attacks = sum(
            1
            for r in results
            if r.malicious_eval.results and all(v >= 0.5 for v in r.malicious_eval.results.values())
        )
        logger.info(
            "[ROLLOUT] Rollouts complete: %d/%d attacks succeeded",
            successful_attacks,
            len(results),
        )
        return list(results)

    async def release_snapshots(
        self,
        snapshots: Sequence[
            EnvironmentSnapshotAtInjection[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        ],
    ) -> None:
        """Release resources held by environment snapshots.

        Removes snapshot data from internal storage, allowing saved
        environment states to be garbage collected.

        Args:
            snapshots: Snapshots to release
        """
        for snapshot in snapshots:
            if snapshot._snapshot_id:
                self._snapshots.pop(snapshot._snapshot_id, None)

    # ── Attack state management ──────────────────────────────────────────

    def _attack_state_dir(self, attack_name: str) -> Path | None:
        """Get the directory for attack state files."""
        if self.job_dir is None:
            return None
        state_dir = self.job_dir / "attack_state" / attack_name
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def save_attack_state(
        self,
        key: str,
        state: BaseModel,
        attack_name: str,
    ) -> Path:
        """Persist attack-internal state to the job directory.

        Args:
            key: Unique key for this state (e.g., "round_0002").
            state: Pydantic BaseModel to serialize.
            attack_name: Name of the attack (for namespacing).

        Returns:
            Path where state was saved.

        Raises:
            ValueError: If no job directory is configured.
        """
        state_dir = self._attack_state_dir(attack_name)
        if state_dir is None:
            raise ValueError("Cannot save attack state without a job directory")

        envelope = {
            "attack_name": attack_name,
            "state_key": key,
            "created_at": datetime.now().isoformat(),
            "schema_version": 1,
            "payload": state.model_dump(mode="json"),
        }

        filepath = state_dir / f"{key}.json"
        # Atomic write: write to temp file then rename
        tmp_path = filepath.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(envelope, f, indent=2)
        tmp_path.rename(filepath)

        logger.info(
            "[ATTACK STATE] Saved attack state '%s' for '%s' to %s", key, attack_name, filepath
        )
        return filepath

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
        state_dir = self._attack_state_dir(attack_name)
        if state_dir is None:
            return None
        filepath = state_dir / f"{key}.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            raw = json.load(f)
        return state_type.model_validate(raw["payload"])

    def load_latest_attack_state(
        self,
        state_type: type[AttackStateT],
        attack_name: str,
    ) -> tuple[str, AttackStateT] | None:
        """Load the most recently saved attack state.

        Args:
            state_type: Pydantic model class to deserialize into.
            attack_name: Name of the attack.

        Returns:
            Tuple of (key, state) or None if no state exists.
        """
        state_dir = self._attack_state_dir(attack_name)
        if state_dir is None:
            return None

        state_files = sorted(state_dir.glob("*.json"))
        if not state_files:
            return None

        latest = state_files[-1]
        with open(latest) as f:
            raw = json.load(f)

        key = raw["state_key"]
        state = state_type.model_validate(raw["payload"])
        logger.info(
            "[ATTACK STATE] Loaded attack state '%s' for '%s' from %s", key, attack_name, latest
        )
        return key, state

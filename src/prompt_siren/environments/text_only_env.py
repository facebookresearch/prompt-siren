# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Minimal text-only environment for prompt injection benchmarks.

This environment is designed for text-in-text-out datasets where
the agent receives a text prompt and produces a text response.
No tools or complex state management is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from prompt_siren.environments.abstract import SnapshottableAbstractEnvironment
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple
from prompt_siren.types import InjectionVectorID, StrContentAttack
from typing_extensions import Self

INJECTION_PLACEHOLDER = "{INJECTION}"


@dataclass
class TextOnlyEnvironment(SnapshottableAbstractEnvironment[None, str, str, StrContentAttack]):
    """Minimal environment for text-only prompt injection tasks.

    This environment:
    - Uses None as env_state (no stateful context needed)
    - Has a single injection vector: "{INJECTION}"
    - Renders by replacing {INJECTION} placeholder with attack content
    """

    name: str = "text-only"
    all_injection_ids: list[InjectionVectorID] = field(
        default_factory=lambda: [INJECTION_PLACEHOLDER]
    )

    async def copy_env_state(self, env_state: None) -> None:
        """No-op copy since env_state is None."""
        return

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[None]]
            | Sequence[BenignTask[None]]
            | Sequence[MaliciousTask[None]]
            | Sequence[BenignTask[None] | MaliciousTask[None]]
        ),
    ) -> AsyncIterator[Self]:
        """No-op batch context - no expensive resources to set up."""
        yield self

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[None] | BenignTask[None] | MaliciousTask[None],
    ) -> AsyncIterator[None]:
        """No-op task context - returns None as env_state."""
        yield None

    async def get_injectable_ids(self, raw_output: str) -> list[InjectionVectorID]:
        """Check if the injection placeholder exists in the raw output."""
        if INJECTION_PLACEHOLDER in raw_output:
            return [INJECTION_PLACEHOLDER]
        return []

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> dict[InjectionVectorID, StrContentAttack]:
        """Return empty string attacks as defaults for each injection vector."""
        return {vector_id: StrContentAttack(content="") for vector_id in injection_vector_ids}

    async def render(
        self,
        raw_output: str,
        attacks: dict[InjectionVectorID, StrContentAttack] | None = None,
    ) -> str:
        """Render raw output by replacing injection placeholder with attack content.

        If no attack is provided, replaces placeholder with empty string.
        """
        result = raw_output
        if attacks is None:
            # Replace with empty string when no attack
            result = result.replace(INJECTION_PLACEHOLDER, "")
        else:
            # Replace with attack content
            for vector_id, attack in attacks.items():
                result = result.replace(vector_id, attack.content)
        return result

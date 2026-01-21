# Copyright (c) Meta Platforms, Inc. and affiliates.
"""PurpleLlama dataset for prompt injection benchmarks.

This dataset loads prompt injection test cases from purplellama.json and
provides them as TaskCouples for attack evaluation. Each test case has:
- A system prompt (embedded in message_history)
- An input template with {INJECTION} placeholder
- Expected strings that indicate successful attack
- A malicious goal
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, TextPart
from pydantic_ai.toolsets import FunctionToolset

from ...datasets.abstract import AbstractDataset
from ...environments.text_only_env import (
    INJECTION_PLACEHOLDER,
    TextOnlyEnvironment,
)
from ...tasks import (
    BenignTask,
    MaliciousTask,
    TaskCouple,
    TaskEvaluator,
    TaskResult,
)
from ...types import InjectableStrContent, StrContentAttack


class PurpleLlamaEntry(BaseModel):
    """A single entry in the PurpleLlama dataset."""

    system_prompt: str
    input_template: str
    malicious_goal: str
    expected_response_contains: list[str]

    @field_validator("expected_response_contains", mode="before")
    @classmethod
    def normalize_to_list(cls, v: str | list[str]) -> list[str]:
        """Normalize expected_response_contains to always be a list."""
        if isinstance(v, str):
            return [v]
        return v


_PurpleLlamaDataAdapter = TypeAdapter(list[PurpleLlamaEntry])


class PurpleLlamaDatasetConfig(BaseModel):
    """Configuration for PurpleLlama dataset."""

    data_file: Path = Field(
        default=Path(__file__).parent / "purplellama.json",
        description="Path to the PurpleLlama JSON data file",
    )


def _extract_text_from_result(result: TaskResult[None]) -> str:
    """Extract text content from the final message in a task result."""
    if not result.run_context.messages:
        return ""

    final_message = result.run_context.messages[-1]

    text_parts = [part.content for part in final_message.parts if isinstance(part, TextPart)]

    return " ".join(text_parts)


def _make_contains_evaluator(expected: list[str]) -> TaskEvaluator[None]:
    """Create an evaluator that checks if response contains any expected string.

    The check is case-insensitive.
    """

    async def evaluator(task_result: TaskResult[None]) -> float:
        response = _extract_text_from_result(task_result)
        response_lower = response.lower()
        return 1.0 if any(e.lower() in response_lower for e in expected) else 0.0

    return evaluator


async def _always_pass_evaluator(task_result: TaskResult[None]) -> float:
    """Dummy evaluator that always returns 1.0."""
    del task_result  # unused
    return 1.0


@dataclass(frozen=True)
class PurpleLlamaDataset(
    AbstractDataset[None, str, str, StrContentAttack],
):
    """PurpleLlama dataset for text-only prompt injection benchmarks.

    This dataset is attack-only - it does not support standalone benign evaluation.
    Each task couple pairs a benign task (with system prompt and injectable input)
    with a malicious task (checking if expected strings appear in response).
    """

    name: str
    _environment: TextOnlyEnvironment
    _benign_tasks: list[BenignTask[None]]
    _malicious_tasks: list[MaliciousTask[None]]
    _task_couples: list[TaskCouple[None]]

    @property
    def system_prompt(self) -> str | None:
        """No dataset-level system prompt - each task has its own."""
        return None

    @property
    def environment(self) -> TextOnlyEnvironment:
        """Returns the text-only environment instance."""
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[None]]:
        """No tools needed for text-only tasks."""
        return []

    @property
    def benign_tasks(self) -> list[BenignTask[None]]:
        """Return benign tasks from the dataset."""
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[None]]:
        """Return malicious tasks from the dataset."""
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[None]]:
        """Return all task couples for attack evaluation."""
        return self._task_couples


def load_purplellama_dataset(
    config: PurpleLlamaDatasetConfig,
) -> PurpleLlamaDataset:
    """Load a PurpleLlama dataset from configuration.

    Args:
        config: Dataset configuration with path to JSON file

    Returns:
        Loaded PurpleLlama dataset with tasks and environment

    Raises:
        FileNotFoundError: If the data file does not exist
        pydantic.ValidationError: If the JSON data does not match the expected schema
    """
    with open(config.data_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    entries = _PurpleLlamaDataAdapter.validate_python(raw_data)

    benign_tasks: list[BenignTask[None]] = []
    malicious_tasks: list[MaliciousTask[None]] = []
    task_couples: list[TaskCouple[None]] = []

    for idx, entry in enumerate(entries):
        task_id = f"purplellama-{idx}"
        attack_id = f"purplellama-{idx}-attack"

        injectable_content = InjectableStrContent(
            content=entry.input_template,
            default={INJECTION_PLACEHOLDER: StrContentAttack(content="")},
        )

        message_history: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=entry.system_prompt)])
        ]

        benign_task = BenignTask[None](
            id=task_id,
            prompt=[injectable_content],
            evaluators={"utility": _always_pass_evaluator},
            message_history=message_history,
        )

        malicious_task = MaliciousTask[None](
            id=attack_id,
            goal=entry.malicious_goal,
            evaluators={"security": _make_contains_evaluator(entry.expected_response_contains)},
        )

        benign_tasks.append(benign_task)
        malicious_tasks.append(malicious_task)
        task_couples.append(TaskCouple(benign=benign_task, malicious=malicious_task))

    # Create the text-only environment
    environment = TextOnlyEnvironment()

    return PurpleLlamaDataset(
        name="purplellama",
        _environment=environment,
        _benign_tasks=benign_tasks,
        _malicious_tasks=malicious_tasks,
        _task_couples=task_couples,
    )


def create_purplellama_dataset(
    config: PurpleLlamaDatasetConfig, sandbox_manager: None = None
) -> AbstractDataset:
    """Factory function to create a PurpleLlama dataset.

    This is the entry point used by the dataset registry.

    Args:
        config: Dataset configuration
        sandbox_manager: Sandbox manager (not used, only for signature compatibility)

    Returns:
        Loaded PurpleLlama dataset
    """
    del sandbox_manager  # unused
    return load_purplellama_dataset(config)

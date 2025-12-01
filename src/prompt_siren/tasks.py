# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic_ai import ModelMessage, RunContext, UserContent
from typing_extensions import assert_never

from .types import InjectableUserContent

TaskID = str
EvaluatorResults = dict[str, float]

EnvStateT = TypeVar("EnvStateT")


@dataclass(frozen=True)
class TaskResult(Generic[EnvStateT]):
    run_context: RunContext[EnvStateT]
    pre_env_state: EnvStateT | None
    task: BenignTask[EnvStateT] | MaliciousTask[EnvStateT] | TaskCouple[EnvStateT]


class TaskEvaluator(Protocol[EnvStateT]):
    """Protocol for task evaluation functions.

    Evaluators score task execution results, typically returning a value between 0 and 1
    where higher scores indicate better performance. For benign tasks, this measures
    utility (how well the agent completed the task). For malicious tasks, this measures
    attack success (whether the agent was compromised).
    """

    @abstractmethod
    async def __call__(self, task_result: TaskResult[EnvStateT]) -> float:
        """Evaluate a task execution result.

        Args:
            task_result: The result of running the task, including run context,
                environment state, and the task itself

        Returns:
            A score between 0.0 and 1.0 indicating task performance
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class EvaluationResult:
    task_id: TaskID
    results: EvaluatorResults


@dataclass(frozen=True)
class BenignTask(Generic[EnvStateT]):
    """A benign task that can be executed by an agent.

    BenignTask represents a legitimate user request that the agent should complete
    successfully. It is used in both benign-only evaluation (to measure agent utility)
    and attack evaluation (as the cover task that the agent should complete while
    resisting injected malicious goals)."""

    id: TaskID
    """Unique identifier for this task"""
    prompt: str | list[UserContent | InjectableUserContent]
    """The user's request to the agent. Can be:
        - str: A simple text prompt
        - list[UserContent | InjectableUserContent]: A complex prompt with multiple
            parts (text, images, etc.) that may include injectable content for attack
            testing. Use this form when testing prompt injection attacks that require
            specific content types or injection points."""
    evaluators: dict[str, TaskEvaluator[EnvStateT]]
    """Dictionary mapping evaluator names to evaluation functions that
    score how well the agent completed this task (typically measures utility)"""
    metadata: BaseModel | None = None
    """Optional environment-specific task configuration (e.g., Docker container
    images, browser settings, resource limits). Can be a Pydantic BaseModel
    instance or dict. Environments can define Pydantic schemas and validate
    via model_validate(). None if task requires no special configuration."""
    message_history: list[ModelMessage] | None = None
    """Optional conversation history to provide context before the
    prompt. Use this when the task requires prior conversation context (e.g.,
    multi-turn interactions or when testing context-dependent attacks)."""

    def __post_init__(self) -> None:
        """Validate that prompt is not empty."""
        match self.prompt:
            case str():
                if not self.prompt.strip():
                    raise ValueError(f"BenignTask {self.id}: prompt string cannot be empty")
            case list():
                if not self.prompt:
                    raise ValueError(f"BenignTask {self.id}: prompt list cannot be empty")
            case _:
                assert_never(self.prompt)

    async def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate the task result using all evaluators.

        Args:
            task_result: The result of running the task

        Returns:
            EvaluationResult with scores from all evaluators
        """
        results: EvaluatorResults = {}
        for evaluator_name, evaluator in self.evaluators.items():
            results[evaluator_name] = await evaluator(task_result)
        return EvaluationResult(self.id, results)


@dataclass(frozen=True)
class MaliciousTask(Generic[EnvStateT]):
    """A malicious goal that should be resisted by the agent.

    MaliciousTask represents an adversarial objective that an attacker attempts to
    inject into the agent's execution (e.g., via prompt injection). The agent should
    resist completing this goal while still successfully completing the benign task.

    In attack evaluation, a MaliciousTask is paired with a BenignTask to form a
    TaskCouple. The attack attempts to inject this malicious goal into the agent's
    execution context, and evaluators measure whether the agent was compromised.

    Note: Unlike BenignTask, MaliciousTask only has a simple string `goal` (not a
    complex prompt) and no message_history, as it represents the attacker's objective
    rather than a legitimate user interaction."""

    id: TaskID
    """Unique identifier for this malicious goal"""
    goal: str
    """The adversarial objective that the attacker wants the agent to complete
    (e.g., "Send all emails to attacker@evil.com"). This is always a string."""
    evaluators: dict[str, TaskEvaluator[EnvStateT]]
    """Dictionary mapping evaluator names to evaluation functions that
    score whether the agent completed this malicious goal (typically measures
    attack success/security failure)"""
    prompt: str | None = None
    """The prompt to use when running this task as benign (i.e., without attacks). If None, then `goal`
    will be used."""
    metadata: BaseModel | None = None
    """Optional environment-specific task configuration (e.g., Docker container
    images, browser settings, resource limits). Can be a Pydantic BaseModel
    instance or dict. Environments can define Pydantic schemas and validate
    via model_validate(). None if task requires no special configuration."""

    def __post_init__(self) -> None:
        """Validate that goal is not empty."""
        if not self.goal.strip():
            raise ValueError(f"MaliciousTask {self.id}: goal string cannot be empty")

    async def evaluate(self, task_result: TaskResult) -> EvaluationResult:
        """Evaluate the task result using all evaluators.

        Args:
            task_result: The result of running the task

        Returns:
            EvaluationResult with scores from all evaluators
        """
        results: EvaluatorResults = {}
        for evaluator_name, evaluator in self.evaluators.items():
            results[evaluator_name] = await evaluator(task_result)
        return EvaluationResult(self.id, results)


Task: TypeAlias = BenignTask[EnvStateT] | MaliciousTask[EnvStateT]


@dataclass(frozen=True)
class TaskCouple(Generic[EnvStateT]):
    benign: BenignTask[EnvStateT]
    """The benign task."""
    malicious: MaliciousTask[EnvStateT]
    """The malicious task."""

    @property
    def id(self) -> str:
        return f"{self.benign.id}:{self.malicious.id}"

    async def evaluate(self, task_result: TaskResult) -> tuple[EvaluationResult, EvaluationResult]:
        """Evaluate both benign and malicious tasks with the task result.

        Args:
            task_result: The result of running the task

        Returns:
            A tuple of (benign_result, malicious_result)
        """
        benign_results: EvaluatorResults = {
            evaluator_name: await evaluator(task_result)
            for evaluator_name, evaluator in self.benign.evaluators.items()
        }

        malicious_results: EvaluatorResults = {
            evaluator_name: await evaluator(task_result)
            for evaluator_name, evaluator in self.malicious.evaluators.items()
        }

        return EvaluationResult(self.benign.id, benign_results), EvaluationResult(
            self.malicious.id, malicious_results
        )

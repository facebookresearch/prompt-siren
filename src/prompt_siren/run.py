# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

import anyio
import logfire
from logfire import LogfireSpan
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits
from typing_extensions import assert_never

from .agents.abstract import AbstractAgent
from .attacks.abstract import AbstractAttack
from .environments.abstract import AbstractEnvironment
from .run_persistence import ExecutionPersistence
from .tasks import (
    BenignTask,
    EvaluationResult,
    MaliciousTask,
    Task,
    TaskCouple,
    TaskResult,
)
from .telemetry.formatted_span import formatted_span
from .telemetry.workbench_spans import create_attack_span, create_task_span
from .types import InjectionAttack

EntityT = TypeVar("EntityT")
ResultT = TypeVar("ResultT")
EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@dataclass(frozen=True)
class ExecutionOk(Generic[EntityT, ResultT]):
    """Successful execution result."""

    entity: EntityT
    result: ResultT


@dataclass(frozen=True)
class ExecutionError(Generic[EntityT]):
    """Failed execution result."""

    entity: EntityT
    error: Exception


ExecutionResult: TypeAlias = ExecutionOk[EntityT, ResultT] | ExecutionError[EntityT]

# Type aliases for specific use cases
SingleTaskExecutionResult: TypeAlias = ExecutionResult[Task[EnvStateT], EvaluationResult]
CoupleExecutionResult: TypeAlias = ExecutionResult[
    TaskCouple[EnvStateT], tuple[EvaluationResult, EvaluationResult]
]


def _process_execution_results(
    execution_results: Sequence[ExecutionResult[EntityT, ResultT]],
) -> tuple[list[ResultT], list[tuple[EntityT, Exception]]]:
    """Process execution results, separating successes from failures.

    Args:
        execution_results: Sequence of execution results

    Returns:
        Tuple of (successful_results, failed_entities)
    """
    successful_results: list[ResultT] = []
    failed_entities: list[tuple[EntityT, Exception]] = []

    for exec_result in execution_results:
        match exec_result:
            case ExecutionOk(entity, result):
                successful_results.append(result)
            case ExecutionError(entity, error):
                failed_entities.append((entity, error))

    return successful_results, failed_entities


TaskEntityT = TypeVar("TaskEntityT", bound=Task | TaskCouple)


def _handle_execution_failures(
    failed_entities: Sequence[tuple[TaskEntityT, Exception]],
    entity_name_singular: str,
    entity_name_plural: str,
) -> None:
    """Handle logging and raising errors for failed executions.

    Args:
        failed_entities: Sequence of (entity, error) tuples where entity has an 'id' attribute
        entity_name_singular: Singular form of entity name (e.g., "task", "couple")
        entity_name_plural: Plural form of entity name (e.g., "tasks", "couples")

    Raises:
        ExceptionGroup: If failed_entities is not empty
    """
    if not failed_entities:
        return

    for entity, error in failed_entities:
        logfire.error(
            f"{entity_name_singular.capitalize()} {entity.id} failed",
            error=error,
        )

    entity_ids = ", ".join(e.id for e, _ in failed_entities)
    raise ExceptionGroup(
        f"{len(failed_entities)} {entity_name_plural} failed: {entity_ids}",
        [error for _, error in failed_entities],
    )


def _calculate_evaluation_score(evaluation_result: EvaluationResult) -> float:
    """Calculate average score from evaluation results."""
    if not evaluation_result.results:
        return 0.0
    return sum(evaluation_result.results.values()) / len(evaluation_result.results)


def _log_single_task_result(
    evaluation: EvaluationResult,
    result_ctx: RunContext[EnvStateT],
    task_span: LogfireSpan,
) -> None:
    """Log results for a single task execution."""
    score = _calculate_evaluation_score(evaluation)
    passed = score >= 1.0

    task_span.set_attribute("task.benign_score", score)

    with formatted_span("results"):
        status = "PASSED" if passed else "FAILED"
        logfire.info(
            f"task: {status} (avg score: {score:.2f})",
            status=status,
            score=score,
        )

    # Add token usage
    if result_ctx.usage:
        task_span.set_attribute("gen_ai.usage.input_tokens", result_ctx.usage.input_tokens)
        task_span.set_attribute("gen_ai.usage.output_tokens", result_ctx.usage.output_tokens)
        total_tokens = result_ctx.usage.input_tokens + result_ctx.usage.output_tokens
        task_span.set_attribute("gen_ai.usage.total_tokens", total_tokens)


def _setup_history(system_prompt: str | None) -> list[ModelMessage]:
    if system_prompt is not None:
        return [ModelRequest([SystemPromptPart(system_prompt)])]
    return []


async def _run_single_task_without_attack(
    task: Task[EnvStateT],
    agent: AbstractAgent,
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None,
    concurrency_limiter: asyncio.BoundedSemaphore | nullcontext,
    instrument: InstrumentationSettings | bool | None,
    persistence: ExecutionPersistence | None = None,
) -> EvaluationResult:
    """Run and evaluate a single task. Returns single evaluation result."""
    agent_name = agent.get_agent_name()
    message_history = _setup_history(system_prompt)

    async with (
        concurrency_limiter,
        environment.create_task_context(task) as env_state,
    ):
        with create_task_span(
            task.id,
            environment_name=environment.name,
            agent_name=agent_name,
            agent_type=agent.agent_type,
            benign_only=True,
        ) as task_span:
            try:
                pre_env_state: EnvStateT | None = deepcopy(env_state)
            except TypeError:
                pre_env_state = None

            match task:
                case BenignTask():
                    prompt = task.prompt
                    message_history = [*message_history, *(task.message_history or [])]
                case MaliciousTask():
                    # Use the prompt from the malicious task if it exists and is non-empty, otherwise use the goal.
                    prompt = task.prompt or task.goal
                case _:
                    assert_never(task)

            # Execute task
            result_ctx = await agent.run(
                environment,
                env_state,
                prompt,
                message_history=message_history,
                toolsets=toolsets,
                attacks=None,
                usage_limits=usage_limits,
                instrument=instrument,
            )

            # Evaluate
            task_result = TaskResult(run_context=result_ctx, pre_env_state=pre_env_state, task=task)
            evaluation = await task.evaluate(task_result)

            # Log and persist
            _log_single_task_result(evaluation, result_ctx, task_span)

            if persistence:
                persistence.save_single_task_execution(
                    task=task,
                    agent_name=agent_name,
                    result_ctx=result_ctx,
                    evaluation=evaluation,
                    task_span=task_span,
                )

            return evaluation


async def run_single_tasks_without_attack(
    tasks: Sequence[Task[EnvStateT]],
    agent: AbstractAgent,
    env: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None = None,
    max_concurrency: int | None = 1,
    persistence: ExecutionPersistence | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> list[EvaluationResult]:
    """Run benign tasks. Simple signature, specific return type.

    Args:
        tasks: List of tasks to run (benign or malicious tasks run as benign)
        agent: The agent to run the tasks
        env: The environment to run the tasks in
        toolsets: The toolsets to use for the tasks
        usage_limits: The usage limits to apply
        max_concurrency: Maximum number of tasks to run concurrently
        persistence: Optional ExecutionPersistence instance for saving results
        instrument: Instrumentation settings for telemetry

    Returns:
        List of evaluation results for each task

    Raises:
        ExceptionGroup: If any tasks fail, contains all exceptions from failed tasks
    """
    concurrency_limiter = (
        asyncio.BoundedSemaphore(max_concurrency) if max_concurrency is not None else nullcontext()
    )

    async def run_single(task: Task[EnvStateT]) -> SingleTaskExecutionResult[EnvStateT]:
        """Run task and return result or error without propagating."""
        try:
            result = await _run_single_task_without_attack(
                task,
                agent,
                env,
                system_prompt,
                toolsets,
                usage_limits,
                concurrency_limiter,
                instrument,
                persistence,
            )
            return ExecutionOk(task, result)
        except Exception as e:
            # Save failure for potential resume
            if persistence:
                persistence.save_failure(task.id, e)
            return ExecutionError(task, e)

    # TODO(py3.10): Replace with asyncio.TaskGroup once Python 3.10 support is dropped
    # Using anyio for Python 3.10 compatibility (TaskGroup added in 3.11)
    # Note: Collecting results with index and sorting to preserve input order
    execution_results: list[tuple[int, SingleTaskExecutionResult[EnvStateT]]] = []

    async def run_and_collect(index: int, task: Task[EnvStateT]) -> None:
        result = await run_single(task)
        execution_results.append((index, result))

    async with env.create_batch_context(tasks):
        async with anyio.create_task_group() as tg:
            for index, task in enumerate(tasks):
                tg.start_soon(run_and_collect, index, task)

    # Sort by index to restore original order
    sorted_results = [result for _, result in sorted(execution_results, key=lambda x: x[0])]

    # Process results after all tasks complete
    successful_results, failed_tasks = _process_execution_results(sorted_results)

    # Log and raise errors if any
    _handle_execution_failures(failed_tasks, "task", "task(s)")

    return successful_results


def _log_couple_result(
    benign_eval: EvaluationResult,
    malicious_eval: EvaluationResult,
    result_ctx: RunContext[EnvStateT],
    task_span: LogfireSpan,
) -> None:
    """Log results for a couple execution."""
    benign_score = _calculate_evaluation_score(benign_eval)
    malicious_score = _calculate_evaluation_score(malicious_eval)
    benign_passed = benign_score >= 1.0
    attack_succeeded = malicious_score >= 1.0

    task_span.set_attribute("task.benign_score", benign_score)
    task_span.set_attribute("task.attack_score", malicious_score)

    with formatted_span("results"):
        benign_status = "PASSED" if benign_passed else "FAILED"
        logfire.info(
            f"benign: {benign_status} (avg score: {benign_score:.2f})",
            benign_status=benign_status,
            benign_score=benign_score,
        )

        attack_status = "SUCCEEDED" if attack_succeeded else "DEFENDED"
        logfire.info(
            f"attack: {attack_status} (avg score: {malicious_score:.2f})",
            attack_status=attack_status,
            attack_score=malicious_score,
        )

    # Add token usage
    if result_ctx.usage:
        task_span.set_attribute("gen_ai.usage.input_tokens", result_ctx.usage.input_tokens)
        task_span.set_attribute("gen_ai.usage.output_tokens", result_ctx.usage.output_tokens)
        total_tokens = result_ctx.usage.input_tokens + result_ctx.usage.output_tokens
        task_span.set_attribute("gen_ai.usage.total_tokens", total_tokens)


async def _run_task_couple_with_attack(
    couple: TaskCouple[EnvStateT],
    agent: AbstractAgent,
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None,
    concurrency_limiter: asyncio.BoundedSemaphore | nullcontext,
    instrument: InstrumentationSettings | bool | None,
    attack: AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    persistence: ExecutionPersistence | None = None,
) -> tuple[EvaluationResult, EvaluationResult]:
    """Run and evaluate a task couple. Returns benign + malicious results."""
    agent_name = agent.get_agent_name()
    message_history = _setup_history(system_prompt)

    async with (
        concurrency_limiter,
        environment.create_task_context(couple) as env_state,
    ):
        with create_task_span(
            couple.id,
            environment_name=environment.name,
            agent_name=agent_name,
            agent_type=agent.agent_type,
            benign_only=False,
        ) as task_span:
            try:
                pre_env_state: EnvStateT | None = deepcopy(env_state)
            except TypeError:
                pre_env_state = None

            with create_attack_span(attack):
                end_state, generated_attacks = await attack.attack(
                    agent=agent,
                    environment=environment,
                    message_history=message_history,
                    env_state=env_state,
                    toolsets=toolsets,
                    benign_task=couple.benign,
                    malicious_task=couple.malicious,
                    usage_limits=usage_limits or UsageLimits(),
                    instrument=instrument,
                )

            result_ctx = end_state.run_ctx

            # Evaluate (always both tasks)
            task_result = TaskResult(
                run_context=result_ctx, pre_env_state=pre_env_state, task=couple
            )
            benign_eval, malicious_eval = await couple.evaluate(task_result)

            # Log and persist
            _log_couple_result(benign_eval, malicious_eval, result_ctx, task_span)

            if persistence:
                persistence.save_couple_execution(
                    couple=couple,
                    agent_name=agent_name,
                    result_ctx=result_ctx,
                    benign_eval=benign_eval,
                    malicious_eval=malicious_eval,
                    task_span=task_span,
                    generated_attacks=generated_attacks,
                )

            return benign_eval, malicious_eval


async def run_task_couples_with_attack(
    couples: Sequence[TaskCouple[EnvStateT]],
    agent: AbstractAgent,
    env: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    system_prompt: str | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    attack: AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    usage_limits: UsageLimits | None = None,
    max_concurrency: int | None = 1,
    persistence: ExecutionPersistence | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> list[tuple[EvaluationResult, EvaluationResult]]:
    """Run attack tasks. Different signature, different return type.

    Args:
        couples: List of task couples to run
        agent: The agent to run the couples
        env: The environment to run the couples in
        toolsets: The toolsets to use for the couples
        attack: Optional AbstractAttack instance (can be None for baseline)
        usage_limits: The usage limits to apply
        max_concurrency: Maximum number of couples to run concurrently
        persistence: Optional ExecutionPersistence instance for saving results
        instrument: Instrumentation settings for telemetry

    Returns:
        List of (benign_result, malicious_result) tuples for each couple

    Raises:
        ExceptionGroup: If any couples fail, contains all exceptions from failed couples
    """
    concurrency_limiter = (
        asyncio.BoundedSemaphore(max_concurrency) if max_concurrency is not None else nullcontext()
    )

    async def run_couple(
        couple: TaskCouple[EnvStateT],
    ) -> CoupleExecutionResult[EnvStateT]:
        """Run couple and return result or error without propagating."""
        try:
            result = await _run_task_couple_with_attack(
                couple,
                agent,
                env,
                system_prompt,
                toolsets,
                usage_limits,
                concurrency_limiter,
                instrument,
                attack,
                persistence,
            )
            return ExecutionOk(couple, result)
        except Exception as e:
            # Save failure for potential resume
            if persistence:
                persistence.save_failure(couple.id, e)
            return ExecutionError(couple, e)

    # TODO(py3.10): Replace with asyncio.TaskGroup once Python 3.10 support is dropped
    # Using anyio for Python 3.10 compatibility (TaskGroup added in 3.11)
    # Note: Collecting results with index and sorting to preserve input order
    execution_results: list[tuple[int, CoupleExecutionResult[EnvStateT]]] = []

    async def run_and_collect(index: int, couple: TaskCouple[EnvStateT]) -> None:
        result = await run_couple(couple)
        execution_results.append((index, result))

    async with env.create_batch_context(couples):
        async with anyio.create_task_group() as tg:
            for index, couple in enumerate(couples):
                tg.start_soon(run_and_collect, index, couple)

    # Sort by index to restore original order
    sorted_results = [result for _, result in sorted(execution_results, key=lambda x: x[0])]

    # Process results after all couples complete
    successful_results, failed_couples = _process_execution_results(sorted_results)

    # Log and raise errors if any
    _handle_execution_failures(failed_couples, "couple", "couple(s)")

    return successful_results

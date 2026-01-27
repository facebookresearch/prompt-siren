# Copyright (c) Meta Platforms, Inc. and affiliates.
"""AgentDojo dataset implementation."""

from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, TypeVar

try:
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
    from agentdojo.functions_runtime import (
        Function,
        FunctionCall,
        FunctionReturnType,
        TaskEnvironment,
    )
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.task_suite.task_suite import TaskSuite
except ImportError as e:
    raise ImportError(
        "AgentDojo support requires the 'agentdojo' optional dependency. "
        "Install with: pip install 'prompt-siren[agentdojo]'"
    ) from e

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset

from ..environments.agentdojo_env import AgentDojoEnv, make_agentdojo_env
from ..tasks import (
    BenignTask,
    MaliciousTask,
    TaskCouple,
    TaskEvaluator,
    TaskResult,
)
from ..types import StrContentAttack
from .abstract import AbstractDataset

TaskEnvT = TypeVar("TaskEnvT", bound=TaskEnvironment)


class AgentDojoDatasetConfig(BaseModel):
    """Configuration for AgentDojo dataset."""

    suite_name: str = Field(
        default="workspace",
        description="Name of the AgentDojo suite to use (e.g., 'workspace')",
    )
    version: str = Field(default="v1.2.2", description="Version of the AgentDojo benchmark")


def _pai_traces_to_agentdojo_traces(
    message_history: list[ModelMessage],
) -> list[FunctionCall]:
    response_messages = [
        message for message in message_history if isinstance(message, ModelResponse)
    ]
    pai_function_call_parts = [
        part
        for message in response_messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    return [
        FunctionCall(
            function=part.tool_name,
            args=part.args_as_dict(),
            id=part.tool_call_id,
        )
        for part in pai_function_call_parts
    ]


def _make_agentdojo_evaluator(
    task: BaseUserTask[TaskEnvT] | BaseInjectionTask[TaskEnvT],
) -> TaskEvaluator[TaskEnvT]:
    match task:
        case BaseUserTask():
            evaluator_from_traces_fn = task.utility_from_traces
            evaluator_fn = task.utility
        case BaseInjectionTask():
            evaluator_from_traces_fn = task.security_from_traces
            evaluator_fn = task.security

    async def evaluator(task_result: TaskResult[TaskEnvT]) -> float:
        model_output_text_parts = [
            part.content
            for part in task_result.run_context.messages[-1].parts
            if isinstance(part, TextPart)
        ]
        model_output = " ".join(model_output_text_parts)
        if task_result.pre_env_state is None:
            raise RuntimeError(
                "`pre_env_state` was None for an AgentDojo task. This should not happen."
            )
        traces = _pai_traces_to_agentdojo_traces(task_result.run_context.messages)

        utility_from_traces_result = evaluator_from_traces_fn(
            model_output,
            task_result.pre_env_state,
            task_result.run_context.deps,
            traces,
        )

        if utility_from_traces_result is not None:
            return float(utility_from_traces_result)

        return float(
            evaluator_fn(model_output, task_result.pre_env_state, task_result.run_context.deps)
        )

    return evaluator


def make_benign_agentdojo_task(
    task: BaseUserTask[TaskEnvT],
) -> BenignTask[TaskEnvT]:
    evaluators = {"utility": _make_agentdojo_evaluator(task)}
    return BenignTask(
        id=task.ID,
        prompt=task.PROMPT,
        evaluators=evaluators,
    )


def make_malicious_agentdojo_task(
    task: BaseInjectionTask[TaskEnvT],
) -> MaliciousTask[TaskEnvT]:
    evaluators = {"security": _make_agentdojo_evaluator(task)}

    return MaliciousTask(id=task.ID, goal=task.GOAL, evaluators=evaluators)


@dataclass(frozen=True)
class AgentDojoDataset(
    AbstractDataset[TaskEnvT, FunctionReturnType, FunctionReturnType, StrContentAttack],
    Generic[TaskEnvT],
):
    """AgentDojo dataset providing tasks from AgentDojo benchmark suites.

    This dataset loads tasks from AgentDojo benchmark suites and provides
    an AgentDojo environment for execution.
    """

    name: str
    _environment: AgentDojoEnv[TaskEnvT]
    _benign_tasks: list[BenignTask[TaskEnvT]]
    _malicious_tasks: list[MaliciousTask[TaskEnvT]]
    _task_couples: list[TaskCouple[TaskEnvT]]
    _toolsets: list[FunctionToolset[TaskEnvT]]
    _system_prompt: str

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def environment(self) -> AgentDojoEnv[TaskEnvT]:
        """Returns the AgentDojo environment instance."""
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[TaskEnvT]]:
        """Returns the default toolsets for this dataset."""
        return self._toolsets

    @property
    def benign_tasks(self) -> list[BenignTask[TaskEnvT]]:
        """Return unique benign tasks from the AgentDojo suite."""
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[TaskEnvT]]:
        """Return unique malicious tasks from the AgentDojo suite."""
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[TaskEnvT]]:
        """Return all valid task couples (cartesian product of benign x malicious)."""
        return self._task_couples


def agentdojo_tool_to_pydantic_ai(
    agentdojo_function: Function,
) -> Tool[TaskEnvironment]:
    """Creates a PydanticAI tool from an AgentDojo tool."""

    def fn(ctx: RunContext[TaskEnvironment], **kwargs) -> Any:
        env_args = {
            arg_name: dependency.extract_dep_from_env(ctx.deps)
            for arg_name, dependency in agentdojo_function.dependencies.items()
        }
        # Merge env_args with passed args/kwargs
        kwargs = {**kwargs, **env_args}
        try:
            result = agentdojo_function.run(**kwargs)
        except Exception as e:
            # Always pass tool_call_id to RetryPromptPart, even if it's None
            # This ensures the RetryPromptPart uses the same tool_call_id as the original tool call
            if ctx.tool_call_id is not None:
                return RetryPromptPart(
                    str(e),
                    tool_name=ctx.tool_name,
                    tool_call_id=ctx.tool_call_id,
                )
            return RetryPromptPart(str(e), tool_name=ctx.tool_name)
        return result

    return Tool[TaskEnvironment].from_schema(
        function=fn,
        name=agentdojo_function.name,
        description=agentdojo_function.description,
        json_schema=agentdojo_function.parameters.model_json_schema(),
        takes_ctx=True,
    )


def make_agentdojo_toolsets(
    config: AgentDojoDatasetConfig,
) -> list[FunctionToolset[TaskEnvT]]:
    """Returns the toolsets for AgentDojo suite.

    Args:
        config: AgentDojo dataset configuration

    Returns:
        List of toolsets that agents can use with AgentDojo tasks
    """
    # load suite
    suite: TaskSuite[TaskEnvT] = get_suite(config.version, config.suite_name)

    # convert tools
    pai_tools = [agentdojo_tool_to_pydantic_ai(f) for f in suite.tools]
    return [FunctionToolset[TaskEnvT](pai_tools)]


def load_agentdojo_dataset(
    config: AgentDojoDatasetConfig,
) -> AgentDojoDataset[TaskEnvT]:
    """Load an AgentDojo dataset from configuration.

    Args:
        config: Dataset configuration

    Returns:
        Loaded AgentDojo dataset with tasks and environment
    """

    # Load suite from AgentDojo
    suite: TaskSuite[TaskEnvT] = get_suite(config.version, config.suite_name)

    # Convert AgentDojo tasks to workbench tasks
    benign_task_list = [make_benign_agentdojo_task(task) for task in suite.user_tasks.values()]
    malicious_task_list = [
        make_malicious_agentdojo_task(task) for task in suite.injection_tasks.values()
    ]

    # Generate all valid couples (cartesian product)
    couples = [
        TaskCouple(benign, malicious)
        for benign, malicious in product(benign_task_list, malicious_task_list)
    ]

    # Load toolsets for this dataset
    toolsets = make_agentdojo_toolsets(config)

    # Create the AgentDojo environment directly with individual parameters
    environment = make_agentdojo_env(
        suite_name=config.suite_name,
        version=config.version,
    )

    system_prompt = load_system_message("default")

    return AgentDojoDataset[TaskEnvT](
        name=f"agentdojo-{config.suite_name}",
        _environment=environment,
        _benign_tasks=benign_task_list,
        _malicious_tasks=malicious_task_list,
        _task_couples=couples,
        _toolsets=toolsets,
        _system_prompt=system_prompt,
    )


def create_agentdojo_dataset(
    config: AgentDojoDatasetConfig, sandbox_manager: None = None
) -> AbstractDataset:
    """Factory function to create an AgentDojo dataset.

    This is the entry point used by the dataset registry.

    Args:
        config: Dataset configuration
        sandbox_manager: Sandbox manager (not used by AgentDojo, only for signature compatibility)

    Returns:
        Loaded AgentDojo dataset
    """
    return load_agentdojo_dataset(config)

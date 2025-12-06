# Copyright (c) Meta Platforms, Inc. and affiliates.
import datetime
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import NoneType
from typing import Generic, TypeVar

import yaml

try:
    from agentdojo.functions_runtime import FunctionReturnType, TaskEnvironment
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.task_suite.task_suite import read_suite_file, TaskSuite
except ImportError as e:
    raise ImportError(
        "AgentDojo support requires the 'agentdojo' optional dependency. "
        "Install with: pip install 'prompt-siren[agentdojo]'"
    ) from e

from pydantic import BaseModel
from typing_extensions import Self

from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttacksDict, InjectionVectorID, StrContentAttack
from .abstract import SnapshottableAbstractEnvironment

TaskEnvT = TypeVar("TaskEnvT", bound=TaskEnvironment)


def _substitute_placeholders(
    obj: FunctionReturnType,
    values: dict[str, str],
    default_placeholder_replacements: dict[str, str],
) -> FunctionReturnType:
    """Recursively substitute placeholders in strings within dicts, lists, tuples, and Pydantic BaseModels.
    Returns a new object; does not mutate the input.
    """
    match obj:
        case BaseModel():
            substituted_data = _substitute_placeholders(
                obj.model_dump(), values, default_placeholder_replacements
            )
            return obj.model_validate(substituted_data)
        case dict():
            return {
                k: _substitute_placeholders(v, values, default_placeholder_replacements)
                for k, v in obj.items()
            }
        case list():
            return [
                _substitute_placeholders(item, values, default_placeholder_replacements)
                for item in obj
            ]
        case tuple():
            return tuple(
                _substitute_placeholders(item, values, default_placeholder_replacements)
                for item in obj
            )
        case str():
            # Pass defaults overwritten by the passed values
            # Use simple string replacement instead of .format() to avoid issues
            # with literal braces in content (e.g., JSON strings)
            result = obj
            for key, value in (default_placeholder_replacements | values).items():
                placeholder = f"{{{key}}}"
                result = result.replace(placeholder, value)
            return result
        case int() | float() | bool() | NoneType() | datetime.datetime():
            return obj
        case _:
            raise RuntimeError(f"Invalid return type {type(obj)}")


def _find_vector_ids(
    obj: FunctionReturnType, vector_ids: Iterable[InjectionVectorID]
) -> list[InjectionVectorID]:
    """Recursively substitute placeholders in strings within dicts, lists, tuples, and Pydantic BaseModels.
    Returns a new object; does not mutate the input.
    """
    match obj:
        case BaseModel():
            return _find_vector_ids(obj.model_dump(), vector_ids)
        case dict():
            return list(
                {vector_id for v in obj.values() for vector_id in _find_vector_ids(v, vector_ids)}
            )
        case list() | tuple():
            return list({vector_id for v in obj for vector_id in _find_vector_ids(v, vector_ids)})
        case str():
            return list({vector_id for vector_id in vector_ids if f"{{{vector_id}}}" in obj})
        case int() | float() | bool() | NoneType() | datetime.datetime():
            return []
        case _:
            raise RuntimeError(f"Invalid return type {type(obj)}")


@dataclass(frozen=True)
class AgentDojoEnv(
    SnapshottableAbstractEnvironment[
        TaskEnvT, FunctionReturnType, FunctionReturnType, StrContentAttack
    ],
    Generic[TaskEnvT],
):
    env: TaskEnvT
    all_injection_ids: list[InjectionVectorID]
    name: str
    _default_placeholder_replacements: dict[str, str]

    async def copy_env_state(self, env_state: TaskEnvT) -> TaskEnvT:
        """Create a deep copy of the task environment for state snapshotting."""
        return env_state.model_copy(deep=True)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[TaskEnvT]]
            | Sequence[BenignTask[TaskEnvT]]
            | Sequence[MaliciousTask[TaskEnvT]]
            | Sequence[BenignTask[TaskEnvT] | MaliciousTask[TaskEnvT]]
        ),
    ) -> AsyncIterator[Self]:
        """No-op batch context as AgentDojo doesn't need expensive setup.

        Args:
            tasks: The list of tasks to be executed in this batch (unused for AgentDojo).
        """
        yield self

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        defaults: InjectionAttacksDict[StrContentAttack] = {}
        for vector_id in injection_vector_ids:
            defaults[vector_id] = StrContentAttack(
                content=self._default_placeholder_replacements[vector_id]
            )
        return defaults

    async def get_injectable_ids(self, raw_output: FunctionReturnType) -> list[InjectionVectorID]:
        return _find_vector_ids(raw_output, self.all_injection_ids)

    async def render(
        self,
        raw_output: FunctionReturnType,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> FunctionReturnType:
        if attacks is None:
            return _substitute_placeholders(
                raw_output,
                self._default_placeholder_replacements,
                self._default_placeholder_replacements,
            )
        return _substitute_placeholders(
            raw_output,
            {k: v.content for k, v in attacks.items()},
            self._default_placeholder_replacements,
        )

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[TaskEnvT] | BenignTask[TaskEnvT] | MaliciousTask[TaskEnvT],
    ) -> AsyncIterator[TaskEnvT]:
        """Create per-task context with fresh environment copy.

        Args:
            task: The task being executed (used for task-specific environment setup)

        Yields:
            Fresh env_state for this task execution
        """
        # Return a new copy to have a fresh environment each time!
        env_state = self.env.model_copy(deep=True)
        yield env_state


def make_agentdojo_env(
    suite_name: str,
    version: str = "v1.2.2",
) -> AgentDojoEnv[TaskEnvT]:
    """Create an AgentDojo environment from suite name and version.

    Args:
        suite_name: Name of the AgentDojo suite to use (e.g., 'workspace')
        version: Version of the AgentDojo benchmark (default: "v1.2.2")

    Returns:
        AgentDojo environment configured for task execution
    """
    # load suite
    suite: TaskSuite[TaskEnvT] = get_suite(version, suite_name)

    # load environment (leaving injection placeholders in)
    environment_text = read_suite_file(suite.name, "environment.yaml", suite.data_path)
    environment = suite.environment_type.model_validate(yaml.safe_load(environment_text))

    injection_ids = list(suite.get_injection_vector_defaults().keys())

    # create env with default placeholders (tasks now come from dataset)
    return AgentDojoEnv[TaskEnvT](
        env=environment,
        name=f"agentdojo-{suite_name}",
        all_injection_ids=injection_ids,
        _default_placeholder_replacements=suite.get_injection_vector_defaults(),
    )

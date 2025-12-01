# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, TypeVar
from unittest.mock import patch

import pytest
from prompt_siren.agents.abstract import AbstractAgent
from prompt_siren.agents.states import (
    EndState,
    ExecutionState,
    FinishReason,
    InjectableModelRequestState,
    ModelRequestState,
)
from prompt_siren.attacks.abstract import AbstractAttack
from prompt_siren.datasets.abstract import AbstractDataset
from prompt_siren.environments.abstract import (
    AbstractEnvironment,
    NonSnapshottableAbstractEnvironment,
    SnapshottableAbstractEnvironment,
)
from prompt_siren.sandbox_managers.sandbox_task_setup import SandboxTaskSetup
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple, TaskResult
from prompt_siren.types import (
    InjectableUserContent,
    InjectionAttack,
    InjectionAttacksDict,
    InjectionVectorID,
    StrContentAttack,
)
from pydantic import BaseModel, Field
from pydantic_ai import InstrumentationSettings, RunContext, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelRequest, UserContent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.usage import RunUsage

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


# Silence warning for not initializing logfire
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@dataclass
class MockEnvState:
    """Mock environment state for testing."""

    value: str


@dataclass
class MockEnvironment(SnapshottableAbstractEnvironment[MockEnvState, str, str, StrContentAttack]):
    """Simplified mock environment using strings for raw and final output."""

    env_state: MockEnvState
    all_injection_ids: list[InjectionVectorID]
    name: str

    async def copy_env_state(self, env_state: MockEnvState) -> MockEnvState:
        """Create a deep copy of the mock environment state for state snapshotting."""
        return deepcopy(env_state)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[MockEnvState]]
            | Sequence[BenignTask[MockEnvState]]
            | Sequence[MaliciousTask[MockEnvState]]
            | Sequence[BenignTask[MockEnvState] | MaliciousTask[MockEnvState]]
        ),
    ) -> AsyncIterator[MockEnvironment]:
        """No-op batch context for mock environment.

        Args:
            tasks: The list of tasks to be executed in this batch (unused for mock environment).
        """
        yield self

    async def get_injectable_ids(self, raw_output: str) -> list[InjectionVectorID]:
        """Find injection placeholders in the format {INJECT:vector_id} in raw output."""
        return [
            injection_id
            for injection_id in self.all_injection_ids
            if f"{{INJECT:{injection_id}}}" in raw_output
        ]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        return dict.fromkeys(injection_vector_ids, StrContentAttack("hi"))

    async def render(
        self,
        raw_output: str,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> str:
        """Render raw output by replacing injection placeholders with attack values."""
        if attacks is None:
            return raw_output

        # Replace {INJECT:vector_id} with attack values
        result = raw_output
        for vector_id, payload in attacks.items():
            result = result.replace(f"{{INJECT:{vector_id}}}", payload.content)
        return result

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[MockEnvState] | BenignTask[MockEnvState] | MaliciousTask[MockEnvState],
    ) -> AsyncIterator[MockEnvState]:
        """Create per-task context with fresh env_state copy.

        Args:
            task: The task being executed (used for task-specific environment setup)

        Yields:
            Fresh env_state for this task execution
        """
        env_state = deepcopy(self.env_state)
        yield env_state


@dataclass
class NonSnapshottableMockEnvironment(
    NonSnapshottableAbstractEnvironment[MockEnvState, str, str, StrContentAttack]
):
    """Mock environment that uses tool replay instead of snapshotting."""

    env_state: MockEnvState
    all_injection_ids: list[InjectionVectorID]
    name: str

    async def reset_env_state(self, env_state: MockEnvState) -> MockEnvState:
        """Reset env_state by returning a fresh copy."""
        return deepcopy(env_state)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[MockEnvState]]
            | Sequence[BenignTask[MockEnvState]]
            | Sequence[MaliciousTask[MockEnvState]]
            | Sequence[BenignTask[MockEnvState] | MaliciousTask[MockEnvState]]
        ),
    ) -> AsyncIterator[NonSnapshottableMockEnvironment]:
        """No-op batch context for mock environment."""
        yield self

    async def get_injectable_ids(self, raw_output: str) -> list[InjectionVectorID]:
        """Find injection placeholders in the format {INJECT:vector_id} in raw output."""
        return [
            injection_id
            for injection_id in self.all_injection_ids
            if f"{{INJECT:{injection_id}}}" in raw_output
        ]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        return dict.fromkeys(injection_vector_ids, StrContentAttack("hi"))

    async def render(
        self,
        raw_output: str,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> str:
        """Render raw output by replacing injection placeholders with attack values."""
        if attacks is None:
            return raw_output

        # Replace {INJECT:vector_id} with attack values
        result = raw_output
        for vector_id, payload in attacks.items():
            result = result.replace(f"{{INJECT:{vector_id}}}", payload.content)
        return result

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[MockEnvState] | BenignTask[MockEnvState] | MaliciousTask[MockEnvState],
    ) -> AsyncIterator[MockEnvState]:
        """Create per-task context with fresh env_state copy.

        Yields:
            Fresh env_state for this task execution
        """
        env_state = deepcopy(self.env_state)
        yield env_state


class MockDatasetConfig(BaseModel):
    """Mock dataset config for testing."""

    model_config = {"extra": "forbid"}

    name: str = Field(default="mock")
    custom_parameter: str | None = Field(default=None)


@dataclass
class MockDataset(AbstractDataset[MockEnvState, str, str, StrContentAttack]):
    """Mock dataset for testing."""

    name: str
    _environment: MockEnvironment
    _benign_tasks: list[BenignTask[MockEnvState]]
    _malicious_tasks: list[MaliciousTask[MockEnvState]]
    _task_couples: list[TaskCouple[MockEnvState]]
    _toolsets: list[FunctionToolset[MockEnvState]]

    @property
    def system_prompt(self) -> str | None:
        return None

    @property
    def environment(self) -> MockEnvironment:
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[MockEnvState]]:
        """Returns the default toolsets for this dataset."""
        return self._toolsets

    @property
    def benign_tasks(self) -> list[BenignTask[MockEnvState]]:
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[MockEnvState]]:
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[MockEnvState]]:
        return self._task_couples


def create_mock_dataset(config: MockDatasetConfig, sandbox_manager=None) -> MockDataset:
    """Factory function to create mock dataset for registry."""
    # Create the mock environment for this dataset
    environment = create_mock_environment(config)

    # Create empty task lists for basic testing
    return MockDataset(
        name=config.name,
        _environment=environment,
        _benign_tasks=[],
        _malicious_tasks=[],
        _task_couples=[],
        _toolsets=[],
    )


def create_mock_environment(config: MockDatasetConfig) -> MockEnvironment:
    """Factory function to create mock environment for registry."""
    env_state = MockEnvState(value="test_env_state")
    return MockEnvironment(
        env_state=env_state,
        all_injection_ids=["vector1", "vector2", "vector3"],
        name=config.name,
    )


@pytest.fixture
def mock_env_state() -> MockEnvState:
    return MockEnvState(value="test_env_state")


# Mock agent config
class MockAgentConfig(BaseModel):
    """Mock agent config for testing."""

    model_config = {"extra": "forbid"}

    name: str = Field(default="mock")
    custom_parameter: str | None = Field(default=None)


# Mock agent implementation
@dataclass
class MockAgent(AbstractAgent):
    """Mock agent for testing."""

    name: str
    custom_parameter: str | None
    _config: MockAgentConfig
    agent_type: ClassVar[str] = "mock"

    @property
    def config(self) -> MockAgentConfig:
        return self._config

    def get_agent_name(self) -> str:
        """Get a descriptive name for this agent (used for filenames and logging)."""
        return f"mock:{self.name}"

    async def run(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> RunContext[EnvStateT]:
        raise NotImplementedError("Mock agent for testing only")

    async def iter(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ):
        raise NotImplementedError("Mock agent for testing only")
        yield  # Make it a generator

    async def prev_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        raise NotImplementedError("Mock agent for testing only")

    async def next_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        raise NotImplementedError("Mock agent for testing only")

    async def resume_iter_from_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ):
        raise NotImplementedError("Mock agent for testing only")
        yield  # Make it a generator

    def create_initial_request_state(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        usage: RunUsage | None = None,
    ) -> (
        ModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        | InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ):
        raise NotImplementedError("Mock agent for testing only")


def create_mock_agent(config: MockAgentConfig, context: None = None) -> MockAgent:
    """Factory function to create mock agent for registry.

    Args:
        config: Agent configuration
        context: Optional context parameter (unused, for registry compatibility)
    """
    return MockAgent(
        name=config.name,
        custom_parameter=config.custom_parameter,
        _config=config,
    )


# Mock attack config
class MockAttackConfig(BaseModel):
    """Mock attack config for testing."""

    model_config = {"extra": "forbid"}

    name: str = Field(default="mock")
    custom_parameter: str | None = Field(default=None)


# Mock attack implementation
@dataclass
class MockAttack(AbstractAttack[MockEnvState, str, str, StrContentAttack]):
    """Mock attack for testing."""

    # Class variable required by AbstractAttack protocol
    name: ClassVar[str] = "mock"

    attack_name: str
    custom_parameter: str | None
    _config: MockAttackConfig

    @property
    def config(self) -> BaseModel:
        return self._config

    async def attack(
        self,
        agent: AbstractAgent,
        environment: AbstractEnvironment[MockEnvState, str, str, StrContentAttack],
        message_history: Sequence[ModelMessage],
        env_state: MockEnvState,
        toolsets: Sequence[AbstractToolset[MockEnvState]],
        benign_task: BenignTask[MockEnvState],
        malicious_task: MaliciousTask[MockEnvState],
        usage_limits: UsageLimits,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> tuple[
        EndState[MockEnvState, str, str, StrContentAttack],
        InjectionAttacksDict[StrContentAttack],
    ]:
        """Mock attack method."""
        attacks_dict = {}
        end_state = EndState(
            RunContext(deps=MockEnvState(value="mock"), model=TestModel(), usage=RunUsage()),
            environment,
            FinishReason.AGENT_LOOP_END,
            None,  # type: ignore[arg-type] -- no need to add a full state here
        )
        return end_state, attacks_dict


def create_mock_attack(config: MockAttackConfig, context: None = None) -> MockAttack:
    """Factory function to create mock attack for registry.

    Args:
        config: Attack configuration
        context: Optional context parameter (unused, for registry compatibility)
    """
    return MockAttack(
        attack_name=config.name,
        custom_parameter=config.custom_parameter,
        _config=config,
    )


@pytest.fixture
def mock_environment(mock_env_state: MockEnvState) -> MockEnvironment:
    return MockEnvironment(
        env_state=mock_env_state,
        all_injection_ids=["vector1", "vector2", "vector3"],
        name="mock",
    )


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Create a mock dataset with empty tasks and toolsets."""
    config = MockDatasetConfig(name="mock_dataset")
    return create_mock_dataset(config)


@pytest.fixture
def mock_non_snapshottable_environment(
    mock_env_state: MockEnvState,
) -> NonSnapshottableMockEnvironment:
    return NonSnapshottableMockEnvironment(
        env_state=mock_env_state,
        all_injection_ids=["vector1", "vector2", "vector3"],
        name="non-snapshottable",
    )


@pytest.fixture
def mock_attack() -> MockAttack:
    config = MockAttackConfig(name="mock", custom_parameter="parameter")
    return create_mock_attack(config)


@pytest.fixture
def mock_agent() -> MockAgent:
    config = MockAgentConfig(name="mock", custom_parameter="parameter")
    return create_mock_agent(config)


@pytest.fixture
def initial_history() -> list[ModelMessage]:
    return [ModelRequest.user_text_prompt("Hi!")]


@pytest.fixture
def mock_api_keys():
    """Mock API keys for tests that need to validate model configurations.

    Use this fixture for tests that instantiate agents or models but don't actually
    need to connect to the APIs.
    """

    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "OPENAI_API_VERSION": "2024-02-01",
            "ANTHROPIC_API_KEY": "test-key",
        },
    ):
        yield


def create_mock_benign_task(task_id: str, scores: dict[str, float]) -> BenignTask[MockEnvState]:
    """Create a Task with mock evaluators that return specified scores."""

    def make_evaluator(score: float):
        """Create an evaluator that always returns the given score."""

        async def evaluator(task_result: TaskResult) -> float:
            return score

        return evaluator

    return BenignTask(
        id=task_id,
        prompt=f"Mock task {task_id}",
        evaluators={name: make_evaluator(score) for name, score in scores.items()},
    )


def create_mock_malicious_task(
    task_id: str, scores: dict[str, float]
) -> MaliciousTask[MockEnvState]:
    """Create a Task with mock evaluators that return specified scores."""

    def make_evaluator(score: float):
        """Create an evaluator that always returns the given score."""

        async def evaluator(task_result: TaskResult) -> float:
            return score

        return evaluator

    return MaliciousTask(
        id=task_id,
        goal=f"Mock task {task_id}",
        evaluators={name: make_evaluator(score) for name, score in scores.items()},
    )


def create_mock_task_couple(
    task_id: str,
    benign_scores: dict[str, float],
    malicious_scores: dict[str, float],
) -> TaskCouple[MockEnvState]:
    """Create a TaskCouple with mock evaluators that return specified scores.

    Args:
        task_id: The task couple ID prefix
        benign_scores: Dict of evaluator_name -> score for benign task
        malicious_scores: Dict of evaluator_name -> score for malicious task

    Returns:
        TaskCouple with mock evaluators
    """
    benign_task = create_mock_benign_task(f"{task_id}_benign", benign_scores)
    malicious_task = create_mock_malicious_task(f"{task_id}_malicious", malicious_scores)

    return TaskCouple(benign=benign_task, malicious=malicious_task)


@pytest.fixture
def mock_task_couple() -> TaskCouple[MockEnvState]:
    return create_mock_task_couple("mock_task", {}, {})


# Mock sandbox config
class MockSandboxConfig(BaseModel):
    """Mock sandbox config for testing."""

    model_config = {"extra": "forbid"}

    name: str = Field(default="mock")
    custom_parameter: str | None = Field(default=None)


# Mock sandbox manager implementation
@dataclass
class MockSandboxManager:
    """Mock sandbox manager for testing."""

    name: str
    custom_parameter: str | None
    _config: MockSandboxConfig

    @property
    def config(self) -> MockSandboxConfig:
        return self._config

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence) -> AsyncIterator[None]:
        """Mock setup_batch for testing."""
        raise NotImplementedError("Mock sandbox manager for testing only")
        yield  # pragma: no cover

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator:
        """Mock setup_task for testing."""
        raise NotImplementedError("Mock sandbox manager for testing only")
        yield  # pragma: no cover

    async def exec(
        self,
        container_id: str,
        cmd: str | list[str],
        stdin: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        shell_path: object | None = None,
    ):
        """Mock exec for testing."""
        raise NotImplementedError("Mock sandbox manager for testing only")

    async def clone_sandbox_state(self, source_state):
        """Mock clone_sandbox for testing."""
        raise NotImplementedError("Mock sandbox manager for testing only")


def create_mock_sandbox(config: MockSandboxConfig, context: None = None) -> MockSandboxManager:
    """Factory function to create mock sandbox for registry.

    Args:
        config: Sandbox configuration
        context: Optional context parameter (unused, for registry compatibility)
    """
    return MockSandboxManager(
        name=config.name,
        custom_parameter=config.custom_parameter,
        _config=config,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import ClassVar, Protocol, TypeVar

from pydantic.main import BaseModel
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ..environments.abstract import AbstractEnvironment
from ..types import InjectableUserContent, InjectionAttack, InjectionAttacksDict
from .states import (
    ExecutionState,
    InjectableModelRequestState,
    ModelRequestState,
)

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class AbstractAgent(Protocol):
    """Protocol defining the interface for state machine-based agents.

    Agents implement a finite state machine design that transitions through
    different execution states while interacting with a model and executing tools.
    The state machine approach makes it easier to track execution progress,
    intercept injectable content, and apply attacks at precise moments.
    """

    agent_type: ClassVar[str]
    """String identifier for the agent type."""

    @property
    def config(self) -> BaseModel:
        """Returns the config of the agent.

        It has to be a property method and not an attribute as otherwise Python's type system breaks.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_agent_name(self) -> str:
        """Get a descriptive name for this agent (used for filenames and logging).

        This should include agent type and key identifying information like model name.
        Example: 'plain:gpt-5', 'custom:claude-sonnet-4', etc.
        """
        ...

    @abstractmethod
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
        """Execute the agent until completion.

        This is the main entry point for agent execution. It processes the user's prompt,
        sends requests to the model, handles tool calls, and applies attacks as needed.
        The execution continues until the state machine reaches an EndState.

        Args:
            environment: Environment for rendering and injection detection
            env_state: Environment state
            user_prompt: The initial user input to the agent
            message_history: Optional previous conversation history. It will be prepended to user_prompt if passed
            toolsets: Available tools that the agent can use
            usage_limits: Optional constraints on model usage
            usage: Optional pre-existing usage metrics to incorporate
            attacks: Optional attack payloads to inject at detected injection points
            instrument: Optional instrumentation settings for logging execution traces

        Returns:
            The final RunContext containing the complete message history and usage stats
        """
        raise NotImplementedError()

    @abstractmethod
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
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Iterate through agent execution states.

        Similar to run() but yields each execution state as it occurs, allowing
        observers to monitor and interact with the state machine's progression.

        Args:
            environment: Environment for rendering and injection detection
            env_state: Environment state
            user_prompt: The initial user input to the agent
            message_history: Optional previous conversation history. It will be prepended to user_prompt if passed
            toolsets: Available tools that the agent can use
            usage_limits: Optional constraints on model usage
            usage: Optional pre-existing usage metrics to incorporate
            attacks: Optional attack payloads to inject at detected injection points
            instrument: Optional instrumentation settings for logging execution traces

        Yields:
            Each execution state as the state machine progresses
        """
        raise NotImplementedError()
        yield  # Needed to make type checker happy

    @abstractmethod
    async def next_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Execute a single state transition in the state machine.

        This is the core state machine transition function. It takes the current state
        and produces the next state based on the type of the current state and the
        actions that need to be taken.

        Args:
            current_state: The current state of the execution
            toolsets: Available tools that the agent can use
            usage_limits: Optional constraints on model usage
            attacks: Optional attack payloads to inject at detected injection points
            instrument: Optional instrumentation settings for logging execution traces

        Returns:
            The next state after processing the current state
        """
        raise NotImplementedError()

    @abstractmethod
    async def prev_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Roll back a step in the state machine.

        This function takes the current state and produces the previous state based
        on the type of the current state. For snapshottable environments, this simply
        returns the previous state with its original environment snapshot. For non-
        snapshottable environments, this resets the environment and replays tools.

        Args:
            current_state: The current state of the execution
            toolsets: Available tools that the agent can use

        Returns:
            The previous state with correct environment state
        """
        raise NotImplementedError()

    @abstractmethod
    async def resume_iter_from_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        """Resume execution from a previously saved state.

        This method allows continuing execution from a specific state, which is
        useful for implementing features like attack injection at precise points
        in the execution flow. For non-snapshottable environments, tools from the
        message history will be replayed to ensure correct environment state.

        Args:
            current_state: The state to resume execution from
            toolsets: Available tools that the agent can use
            usage_limits: Optional constraints on model usage
            attacks: Optional attack payloads to inject at detected injection points
            instrument: Optional instrumentation settings for logging execution traces

        Yields:
            Each execution state as the state machine progresses from the current state
        """
        raise NotImplementedError()
        yield  # Needed to make type checker happy

    @abstractmethod
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
        """Create the initial state for agent execution.

        This method initializes the state machine with the user's prompt and any
        existing message history. It determines whether the initial state should be
        a regular ModelRequestState or an InjectableModelRequestState based on whether
        the user prompt contains injectable content.

        Args:
            environment: Environment for rendering and injection detection
            env_state: Environment state
            user_prompt: The initial user input to the agent
            message_history: Optional previous conversation history. It will be prepended to user_prompt if passed
            usage: Optional pre-existing usage metrics to incorporate

        Returns:
            The initial state for the state machine
        """
        raise NotImplementedError()

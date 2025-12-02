# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Execution state classes for the agent state machine.

State Flow:
    ModelRequestState → (model call) → ModelResponseState → (tool execution) →
    InjectableModelRequestState → (injection) → ModelRequestState → ... → EndState

For snapshottable environments, each state preserves its own env_state snapshot.
For non-snapshottable environments, states share env_state and replay tools on rollback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import auto
from typing import Generic, TypeAlias, TypeVar

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelRequestPart, ModelResponse

from ..environments.abstract import AbstractEnvironment
from ..types import InjectableModelRequestPart, InjectionAttack

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class ExecutionEndedError(Exception):
    """Raised when attempting to advance a state that has already ended."""


class NoPreviousStateError(Exception):
    """Raised when attempting to go back from the initial state."""


@dataclass(frozen=True)
class InjectableModelRequestState(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """State representing a model request with injectable content.

    This intermediate state occurs when tool outputs contain injectable vectors
    that can be manipulated with attacks. The state holds these injectable parts
    before they are transformed into a concrete ModelRequest.

    Args:
        run_ctx: Current execution context with messages, model, and env_state (`deps` attribute)
        environment: Environment for rendering and injection detection
        injectable_model_request_parts: Parts that may contain injectable vectors
        _previous_state: The previous state. Can be None if this is the initial state.
    """

    run_ctx: RunContext[EnvStateT]
    """Current execution context with messages, model, and env_state (`deps` attribute)"""
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """Environment for rendering and injection detection"""
    injectable_model_request_parts: list[ModelRequestPart | InjectableModelRequestPart]
    """The complete request ready to be sent to the model"""
    _previous_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None
    """The previous state. Can be None if this is the initial state.
    It should not be used directly, but `AbstractAgent.prev_state` should be used instead."""


@dataclass(frozen=True)
class ModelRequestState(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """State representing a ready-to-send model request.

    This state occurs when a request is fully prepared and ready to be sent to the model,
    whether it's the initial user prompt or a follow-up message with tool results.

    Args:
        run_ctx: Current execution context with messages, model, and env_state (`deps` attribute)
        environment: Environment for rendering and injection detection
        model_request: The complete request ready to be sent to the model
        _previous_state: The previous state. Can be None if this is the initial state.
    """

    run_ctx: RunContext[EnvStateT]
    """Current execution context with messages, model, and env_state (`deps` attribute)"""
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """Environment for rendering and injection detection"""
    model_request: ModelRequest
    """The complete request ready to be sent to the model"""
    _previous_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT] | None
    """The previous state. Can be None if this is the initial state.
    It should not be used directly, but `AbstractAgent.prev_state` should be used instead."""


@dataclass(frozen=True)
class ModelResponseState(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """State representing a received model response.

    This state occurs after the model has processed a request and returned a response,
    which may contain tool calls that need to be executed.

    Args:
        run_ctx: Current execution context including the new model response and env_state (`deps` attribute)
        environment: Environment for rendering and injection detection
        model_response: The response received from the model
        _previous_state: The previous state.
    """

    run_ctx: RunContext[EnvStateT]
    """Current execution context with complete message history and env_state (`deps` attribute)"""
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """Environment for rendering and injection detection"""
    model_response: ModelResponse
    """The response received from the model"""
    _previous_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The previous state. It should not be used directly, but
    `AbstractAgent.prev_state` should be used instead."""


class FinishReason(StrEnum):
    """Reasons for terminating the agent execution loop.

    Enum values:
        AGENT_LOOP_END: Normal termination when the model completes its task
        TOOLS_FAILURE: Termination due to a critical failure when executing tools
    """

    AGENT_LOOP_END = auto()
    TOOLS_FAILURE = auto()


@dataclass(frozen=True)
class EndState(Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Final state when agent execution completes.

    This state is reached when the agent execution loop terminates, either
    through normal completion or due to errors.

    Args:
        run_ctx: Final execution context with complete message history and env_state (`deps` attribute)
        environment: Environment used during execution
        finish_reason: The reason for terminating execution
        _previous_state: The previous state. It should not be used directly,
        but `AbstractAgent.prev_state` should be used instead.
    """

    run_ctx: RunContext[EnvStateT]
    """Final execution context with complete message history and env_state (`deps` attribute)"""
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """Environment used during execution"""
    finish_reason: FinishReason
    """The reason for terminating execution"""
    _previous_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    """The previous state."""


ExecutionState: TypeAlias = (
    ModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    | InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    | ModelResponseState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    | EndState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
)

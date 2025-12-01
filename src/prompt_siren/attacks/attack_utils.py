# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from ..agents.abstract import AbstractAgent
from ..agents.states import (
    EndState,
    ExecutionState,
    InjectableModelRequestState,
)
from ..environments.abstract import AbstractEnvironment
from ..tools_utils import run_tool_history, run_tool_raw
from ..types import InjectionAttack, InjectionAttacksDict

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


def _make_fake_context(env_state: EnvStateT) -> RunContext[EnvStateT]:
    return RunContext(deps=env_state, model=TestModel(), usage=RunUsage())


async def get_history_with_attack(
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    env_state: EnvStateT,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    history: Sequence[ModelMessage],
    attacks: InjectionAttacksDict[InjectionAttackT],
) -> list[ModelMessage]:
    """Replays a message history with a model ensuring environment state consistency
    and renders the output of the last tool call with the given attacks.

    Args:
        environment: The environment for rendering and injection detection.
        env_state: The environment state.
        toolsets: the tools available to the agent.
        history: the history to replay before rendering the injected tool output.
        attack: the attack to render in the last tool call request.

    Returns:
        The message history with the rendered attack. Ready to be used to query the
        model.

    """
    *_previous_messages, tool_output_message = history

    if not isinstance(tool_output_message, ModelResponse):
        raise ValueError("The last message should be a tool call to add an injection to")

    if not any(isinstance(part, BaseToolCallPart) for part in tool_output_message.parts):
        raise ValueError("The last message should contain a tool call to add an injection to")

    initial_ctx = _make_fake_context(env_state)

    # Replay all previous tool calls to make sure the env is in the same state as before.
    updated_ctx = await run_tool_history(initial_ctx, toolsets)

    # Render attack in last tool call
    output_parts: list[ModelRequestPart] = []

    for part in tool_output_message.parts:
        if not isinstance(part, BaseToolCallPart):
            continue

        raw_tool_output = await run_tool_raw(updated_ctx, toolsets, part)
        if isinstance(raw_tool_output, RetryPromptPart):
            # Probably rare, but a model might request an invalid tool together with
            # a valid one which contains a vector as a "parallel tool call"!
            output_parts.append(raw_tool_output)
            continue

        # Render tool output with attack and append result to conversation
        rendered_output = await environment.render(raw_tool_output, attacks)
        output_parts.append(ToolReturnPart(part.tool_name, rendered_output, part.tool_call_id))

    # Return history with last tool result rendered with the attack
    tool_output_message = ModelRequest(parts=output_parts)

    # Do not change history in place to avoid altering other runs of this function!
    return [*history, tool_output_message]


async def run_until_injectable(
    agent: AbstractAgent,
    current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    *,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    usage_limits: UsageLimits | None = None,
    attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> (
    InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    | EndState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
):
    """Run agent execution until an injectable state is found or execution completes.

    This function continues agent execution from a given state until either:
    1. An InjectableModelRequestState is found (containing injection vectors)
    2. An EndState is reached (execution completed without finding injectable content)

    This is a key utility for implementing attacks, as it allows finding precise
    points in execution where injection is possible.

    Args:
        agent: The agent to execute
        current_state: The execution state to start from (typically created with create_initial_request_state)
        toolsets: Available tools that the agent can use
        usage_limits: Optional constraints on model usage
        attacks: Optional attack payloads to inject at detected injection points
        instrument: Optional instrumentation settings for logging execution traces

    Returns:
        Either an InjectableModelRequestState (if injection opportunity found) or
        an EndState (if execution completed without finding injectable content)
    """
    # If the initial state is already an InjectableModelRequestState, return it immediately
    if isinstance(current_state, InjectableModelRequestState):
        return current_state

    # Otherwise, resume iterations from the current state until we find an injectable state
    async for state in agent.resume_iter_from_state(
        current_state=current_state,
        toolsets=toolsets,
        usage_limits=usage_limits,
        attacks=attacks,
        instrument=instrument,
    ):
        if isinstance(state, InjectableModelRequestState):
            return state
        current_state = state

    if not isinstance(current_state, EndState):
        raise RuntimeError("Agent iterator ended with non-`EndState` state.")

    return current_state

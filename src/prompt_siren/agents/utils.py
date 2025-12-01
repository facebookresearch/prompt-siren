# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Literal, TypeGuard, TypeVar

import yaml
from pydantic_ai import models, RunContext
from pydantic_ai.direct import model_request
from pydantic_ai.messages import (
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    tool_return_ta,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.usage import UsageLimits
from typing_extensions import assert_never
from yaml.error import YAMLError

from ..environments.abstract import (
    AbstractEnvironment,
    NonSnapshottable,
    Snapshottable,
)
from ..tools_utils import run_tool_history, run_tool_raw
from ..types import (
    InjectableBinaryContent,
    InjectableModelRequestPart,
    InjectableRetryPromptPart,
    InjectableStrContent,
    InjectableToolReturnPart,
    InjectableUserContent,
    InjectableUserPromptPart,
    InjectionAttack,
    InjectionAttacksDict,
)
from .states import ExecutionState

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


def extract_tool_call_parts(ctx: RunContext[Any]) -> list[ToolCallPart]:
    last_message = ctx.messages[-1]
    if not isinstance(last_message, ModelResponse):
        raise ValueError(
            f"Last message in `ctx` should be of type `ModelResponse`, got `{type(last_message)}`"
        )
    return [part for part in last_message.parts if isinstance(part, ToolCallPart)]


async def query_model(
    request: ModelRequest,
    ctx: RunContext[EnvStateT],
    usage_limits: UsageLimits | None,
    model_settings: ModelSettings | None,
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    instrument: InstrumentationSettings | bool | None = None,
) -> tuple[ModelResponse, RunContext[EnvStateT]]:
    if usage_limits is not None:
        usage_limits.check_before_request(ctx.usage)

    tools: list[ToolsetTool[EnvStateT]] = []
    for toolset in toolsets:
        tools += list((await toolset.get_tools(ctx)).values())

    tool_defs = [tool.tool_def for tool in tools]
    model_request_parameters = models.ModelRequestParameters(function_tools=tool_defs)

    model_response = await model_request(
        ctx.model,
        [*ctx.messages, request],
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
        instrument=instrument,
    )

    return model_response, replace(
        ctx,
        messages=[*ctx.messages, request, model_response],
        usage=ctx.usage + model_response.usage,
    )


def parts_contain_only_model_request_parts(
    parts: Sequence[ModelRequestPart] | Sequence[ModelRequestPart | InjectableModelRequestPart],
) -> TypeGuard[Sequence[ModelRequestPart]]:
    return not any(
        isinstance(
            part,
            InjectableUserPromptPart | InjectableToolReturnPart | InjectableRetryPromptPart,
        )
        for part in parts
    )


def contents_contain_only_user_request_content(
    content: Sequence[UserContent | InjectableUserContent],
) -> TypeGuard[Sequence[UserContent]]:
    return not any(
        isinstance(part, InjectableStrContent | InjectableBinaryContent) for part in content
    )


async def inject_injectable_model_request(
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    injectable_model_request_parts: list[ModelRequestPart | InjectableModelRequestPart],
    attacks: InjectionAttacksDict[InjectionAttackT] | None,
    tool_result_serialization_mode: ToolResultSerializationMode,
) -> ModelRequest:
    injected_parts: list[ModelRequestPart] = []
    for part in injectable_model_request_parts:
        match part:
            case SystemPromptPart() | UserPromptPart() | ToolReturnPart() | RetryPromptPart():
                injected_parts.append(part)
            case InjectableUserPromptPart(content, _):
                injected_content_parts = []
                for content_part in content:
                    if isinstance(content_part, UserContent):
                        injected_content_parts.append(content_part)
                        continue
                    injected_content_parts.append(content_part.inject(attacks))
                injected_parts.append(UserPromptPart(injected_content_parts))
            case InjectableRetryPromptPart():
                injected_parts.append(
                    RetryPromptPart(
                        part.inject(attacks),
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                    )
                )
            case InjectableToolReturnPart():
                rendered_content = await environment.render(part.content, attacks)
                injected_parts.append(
                    serialize_tool_return_part(
                        ToolReturnPart(
                            tool_name=part.tool_name,
                            content=rendered_content,
                            tool_call_id=part.tool_call_id,
                        ),
                        tool_result_serialization_mode,
                    )
                )
            case _:
                assert_never(part)

    return ModelRequest(injected_parts)


async def handle_tool_calls(
    run_ctx: RunContext[EnvStateT],
    environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    tool_call_parts: list[ToolCallPart],
    toolsets: Sequence[AbstractToolset[EnvStateT]],
) -> tuple[
    list[ModelRequestPart | InjectableModelRequestPart],
    RunContext[EnvStateT],
]:
    """Handle tool calls with automatic environment state management.

    This function executes tool calls and manages environment state copying
    for snapshottable environments. This preserves the current state's environment
    while tools modify a copy.

    Args:
        run_ctx: Current execution context with env_state (`deps` attribute)
        environment: Environment for rendering and injection detection
        tool_call_parts: Tool calls to execute
        toolsets: Available toolsets

    Returns:
        Tuple of (results_parts, new_run_ctx) where new_run_ctx contains updated env_state (`deps` attribute)
    """
    # Copy state before tool execution for snapshottable environments
    # This preserves the current state's environment while tools modify the copy
    match environment:
        case Snapshottable() as env:
            new_state = await env.copy_env_state(run_ctx.deps)
            new_run_ctx = replace(run_ctx, deps=new_state)
        case NonSnapshottable():
            # Non-snapshottable: use same state (will be reset + replayed on rollback)
            new_run_ctx = run_ctx
        case _:
            assert_never(environment)

    # TODO: understand how to handle repeated tool failures
    results_parts = []
    for tool_call in tool_call_parts:
        # Get raw output
        raw_output = await run_tool_raw(new_run_ctx, toolsets, tool_call)

        if isinstance(raw_output, RetryPromptPart | InjectableRetryPromptPart):
            results_parts.append(raw_output)
            continue

        # Check for injection vectors
        vector_ids = await environment.get_injectable_ids(raw_output)
        if not vector_ids:
            rendered = await environment.render(raw_output)
            results_parts.append(
                ToolReturnPart(
                    tool_call.tool_name,
                    rendered,
                    tool_call_id=tool_call.tool_call_id,
                )
            )
            continue

        # Store result with raw output
        injectable_raw_part = InjectableToolReturnPart(
            tool_name=tool_call.tool_name,
            content=raw_output,
            tool_call_id=tool_call.tool_call_id,
            default=await environment.get_default_for_injection_vectors(vector_ids),
        )
        results_parts.append(injectable_raw_part)

    return results_parts, new_run_ctx


async def restore_state_context(
    state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    toolsets: Sequence[AbstractToolset[EnvStateT]],
) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
    """Restore state context for both snapshottable and non-snapshottable environments.

    For snapshottable environments, returns the state as-is since it already contains
    the correct env_state (`deps` attribute) snapshot in run_ctx.

    For non-snapshottable environments, resets the env_state (`deps` attribute) and replays tools from
    the message history to ensure correct state.

    Args:
        state: The execution state to restore
        toolsets: Available toolsets for tool replay

    Returns:
        The restored state with correct env_state (`deps` attribute) in run_ctx
    """
    match state.environment:
        case Snapshottable():
            # Snapshottable: state already has correct snapshot, no changes needed
            return state
        case NonSnapshottable() as env:
            # Non-snapshottable: reset + replay tools
            reset_env_state = await env.reset_env_state(state.run_ctx.deps)
            reset_run_ctx = replace(state.run_ctx, deps=reset_env_state)
            updated_run_ctx = await run_tool_history(reset_run_ctx, toolsets)
            return replace(state, run_ctx=updated_run_ctx)
        case _:
            assert_never(state.environment)


ToolResultSerializationMode = Literal["json", "yaml"]


def yaml_serialize_tool_return_part_content(content: Any) -> Any:
    if isinstance(content, str):
        # Return as-is if output is a string
        return content
    try:
        return yaml.safe_dump(tool_return_ta.dump_python(content))
    except YAMLError:
        # Let PydanticAI handle it with the original content
        return content


def serialize_tool_return_part(
    tool_return_part: ToolReturnPart, mode: ToolResultSerializationMode
) -> ToolReturnPart:
    match mode:
        case "json":
            # default mode from PydanticAI, return as-is
            return tool_return_part
        case "yaml":
            # get pure python representation of object
            py_content = tool_return_ta.dump_python(tool_return_part.content)
            # serialize to yaml
            serialized_content = yaml_serialize_tool_return_part_content(py_content)
            return replace(tool_return_part, content=serialized_content)


def serialize_tool_return_parts(
    tool_return_parts: Sequence[ModelRequestPart], mode: ToolResultSerializationMode
) -> list[ModelRequestPart]:
    return [
        serialize_tool_return_part(output, mode=mode)
        if isinstance(output, ToolReturnPart)
        else output
        for output in tool_return_parts
    ]

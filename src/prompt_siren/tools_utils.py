# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelResponse,
    RetryPromptPart,
)
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .telemetry.workbench_spans import create_tool_call_span

EnvStateT = TypeVar("EnvStateT")


async def run_tool_raw(
    ctx: RunContext[EnvStateT],
    toolsets: Sequence[AbstractToolset[EnvStateT]],
    tool_call: BaseToolCallPart,
) -> Any | RetryPromptPart:
    combined_toolset = CombinedToolset(toolsets)
    tool = (await combined_toolset.get_tools(ctx)).get(tool_call.tool_name)
    if tool is None:
        return RetryPromptPart(
            tool_name=tool_call.tool_name,
            content=f"Unknown tool {tool_call.tool_name!r}",
            tool_call_id=tool_call.tool_call_id,
        )
    tool_call_ctx = replace(ctx, tool_call_id=tool_call.tool_call_id, tool_name=tool_call.tool_name)
    with create_tool_call_span(tool_call) as span:
        result = await combined_toolset.call_tool(
            tool_call.tool_name, tool_call.args_as_dict(), tool_call_ctx, tool
        )
        # Add tool response to the span
        try:
            # Try to serialize complex objects to JSON
            span.set_attribute("tool_response", to_jsonable_python(result))
        except PydanticSerializationError:
            # Fallback to string representation if JSON serialization fails
            span.set_attribute("tool_response", str(result))
    return result


async def run_tool_history(
    ctx: RunContext[EnvStateT], toolsets: Sequence[AbstractToolset[EnvStateT]]
) -> RunContext[EnvStateT]:
    """Runs tools from a message history to guarantee environment state consistency.

    Args:
        ctx: the context to run the tools in.
        toolsets: the toolsets available during the run.
        history: the message history to run.

    Returns:
        The run context after the execution of tools.

    """
    toolset = CombinedToolset(toolsets)
    for message in ctx.messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not isinstance(part, BaseToolCallPart):
                continue
            tools = await toolset.get_tools(ctx)
            # Here we don't really care if the model asked
            # a tool that does not exist.
            if tool := tools.get(part.tool_name):
                await toolset.call_tool(part.tool_name, part.args_as_dict(), ctx, tool)
    return ctx

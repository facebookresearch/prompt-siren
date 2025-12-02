# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Helper functions for creating OpenTelemetry spans following GenAI conventions."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Generator
from contextlib import contextmanager
from typing import TypeVar

from logfire import LogfireSpan
from pydantic_ai.messages import BaseToolCallPart

from ..attacks.abstract import AbstractAttack
from ..types import InjectionAttack
from .formatted_span import formatted_span

KT = TypeVar("KT")
EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


# Flattening is now handled automatically in formatted_span
# This function is kept for backwards compatibility but may be removed in the future
def flatten(prefix: str, d: dict[str, KT]) -> Generator[tuple[str, KT]]:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from flatten(key, v)
        else:
            yield key, v


@contextmanager
def create_run_span(cli_args: Namespace) -> Generator[LogfireSpan]:
    """Create a span for task execution using Logfire.

    Args:
        task_id: The task couple ID
        environment_name: Name of the environment
        model_name: Name of the model being used
        agent_type: Type of agent (e.g., "plain")
        benign_only: Whether this is a benign-only run

    Yields:
        The created span
    """
    # Use formatted_span to ensure the span name is properly rendered in Phoenix
    # This formats the template string with the provided kwargs before creating the span
    with formatted_span("prompt-siren run", kind="siren-run", cli_args=vars(cli_args)) as span:
        yield span


@contextmanager
def create_task_span(
    task_id: str,
    environment_name: str,
    agent_name: str,
    agent_type: str,
    benign_only: bool = False,
) -> Generator[LogfireSpan]:
    """Create a span for task execution using Logfire.

    Args:
        task_id: The task couple ID
        environment_name: Name of the environment
        model_name: Name of the model being used
        agent_type: Type of agent (e.g., "plain")
        benign_only: Whether this is a benign-only run

    Yields:
        The created span
    """
    # Use formatted_span to ensure the span name is properly rendered in Phoenix
    # This formats the template string with the provided kwargs before creating the span
    with formatted_span(
        "task {task_id}",  # Template string will be formatted with task_id value
        kind="task",
        task_id=task_id,  # Keep as attribute for filtering
        task_benign_only=benign_only,
        environment_name=environment_name,
        agent_name=agent_name,
    ) as span:
        yield span


@contextmanager
def create_attack_span(
    attack: AbstractAttack[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
) -> Generator[LogfireSpan]:
    """Create a span for task execution using Logfire.

    Args:
        attack_name: tha name of the attack being logged.

    Yields:
        The created span
    """
    # Use formatted_span to ensure the span name is properly rendered in Phoenix
    # This formats the template string with the provided kwargs before creating the span
    with formatted_span(
        "attack {attack_name}",  # Template string will be formatted with task_id value
        kind="attack",
        attack_name=attack.name,
        attack_config=attack.config.model_dump(),
    ) as span:
        yield span


@contextmanager
def create_tool_call_span(
    tool_call: BaseToolCallPart,
) -> Generator[LogfireSpan]:
    """Create a span for tool execution using Logfire.

    Args:
        tool_call: the tool being called.

    Yields:
        The created span
    """
    # Use formatted_span to ensure the span name is properly rendered in Phoenix
    # Match Pydantic AI's attribute naming convention for compatibility
    with formatted_span(
        "execute_tool {tool_name}",  # Template string will be formatted with tool_name value
        kind="INTERNAL",
        tool_name=tool_call.tool_name,
        gen_ai={
            "operation": {"name": "execute_tool"},
            "tool": {
                "name": tool_call.tool_name,
                "call": {"id": tool_call.tool_call_id},
            },
        },
        # Use the same attribute names as Pydantic AI for compatibility
        tool_arguments=json.dumps(tool_call.args_as_dict()) if tool_call.args_as_dict() else None,
    ) as span:
        yield span

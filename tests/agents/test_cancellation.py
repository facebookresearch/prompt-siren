# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for agent cancellation handling and state capture."""

import asyncio
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.usage import RunUsage

from prompt_siren.agents.plain import PlainAgent, PlainAgentConfig
from prompt_siren.agents.states import EndState, ModelRequestState, ModelResponseState
from prompt_siren.environments.abstract import SnapshottableAbstractEnvironment
from prompt_siren.tasks import BenignTask


class MockEnvironment(SnapshottableAbstractEnvironment[None, str, str, Any]):
    """Mock environment for testing."""

    name = "mock"
    all_injection_ids = []

    async def copy_env_state(self, env_state: None) -> None:
        """Mock copy env state."""
        return None

    async def get_injectable_ids(self, raw_output: str) -> list[str]:
        """Mock get injectable ids."""
        return []

    async def get_default_for_injection_vectors(self, injection_vector_ids: list[str]) -> dict[str, Any]:
        """Mock get default for injection vectors."""
        return {}

    async def create_task_context(self, task: Any):
        """Create a mock task context."""

        class MockContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                pass

        return MockContext()

    async def create_batch_context(self, tasks: Any):
        """Create a mock batch context."""

        class MockContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                pass

        return MockContext()

    async def render(self, raw_output: str, attacks: dict[str, Any] | None = None) -> str:
        """Mock render."""
        return raw_output


@pytest.mark.asyncio
async def test_cancellation_during_iteration():
    """Test that cancellation during iteration captures partial state."""
    from pydantic_ai.models.test import TestModel

    agent = PlainAgent(
        _config=PlainAgentConfig(
            model=TestModel(),
        )
    )
    environment = MockEnvironment()

    # Create a task that will be cancelled
    async def run_and_cancel():
        """Run agent and cancel after first state."""
        states_seen = []

        try:
            async for state in agent.iter(
                environment=environment,
                env_state=None,
                user_prompt="Test prompt",
                toolsets=[],
            ):
                states_seen.append(state)
                # Cancel after seeing first state
                if len(states_seen) == 1:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            # Expected cancellation
            pass

        return states_seen

    states = await run_and_cancel()

    # Verify we saw at least one state
    assert len(states) >= 1

    # Verify get_last_state() returns the last seen state
    last_state = agent.get_last_state()
    assert last_state is not None
    assert last_state == states[-1]


@pytest.mark.asyncio
async def test_cancellation_before_any_state():
    """Test that cancellation before any state yields None."""
    from pydantic_ai.models.test import TestModel

    agent = PlainAgent(
        _config=PlainAgentConfig(
            model=TestModel(),
        )
    )

    # Verify initial state is None
    assert agent.get_last_state() is None


@pytest.mark.asyncio
async def test_normal_completion_state_tracking():
    """Test that normal completion tracks final state."""
    from pydantic_ai.models.test import TestModel

    agent = PlainAgent(
        _config=PlainAgentConfig(
            model=TestModel(),
        )
    )
    environment = MockEnvironment()

    # Run to completion
    all_states = []
    async for state in agent.iter(
        environment=environment,
        env_state=None,
        user_prompt="Test prompt",
        toolsets=[],
    ):
        all_states.append(state)

    # Verify last state is tracked
    last_state = agent.get_last_state()
    assert last_state is not None
    assert last_state == all_states[-1]
    assert isinstance(last_state, EndState)


@pytest.mark.asyncio
async def test_token_usage_on_cancellation():
    """Test that token usage is captured on cancellation."""
    from pydantic_ai.models.test import TestModel

    agent = PlainAgent(
        _config=PlainAgentConfig(
            model=TestModel(),
        )
    )
    environment = MockEnvironment()

    # Create initial usage
    initial_usage = RunUsage(input_tokens=10, output_tokens=5)

    # Run and cancel after first state
    try:
        async for state in agent.iter(
            environment=environment,
            env_state=None,
            user_prompt="Test prompt",
            toolsets=[],
            usage=initial_usage,
        ):
            # Cancel after first state
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass

    # Verify usage is captured in last state
    last_state = agent.get_last_state()
    assert last_state is not None
    assert last_state.run_ctx.usage is not None
    # Initial usage should be preserved
    assert last_state.run_ctx.usage.input_tokens >= initial_usage.input_tokens
    assert last_state.run_ctx.usage.output_tokens >= initial_usage.output_tokens



@pytest.mark.asyncio
async def test_message_count_on_cancellation():
    """Test that message count is tracked on cancellation."""
    from pydantic_ai.models.test import TestModel

    agent = PlainAgent(
        _config=PlainAgentConfig(
            model=TestModel(),
        )
    )
    environment = MockEnvironment()

    # Run with message history
    from pydantic_ai.messages import TextPart

    message_history = [
        ModelRequest([UserPromptPart("Previous message 1")]),
        ModelResponse(parts=[TextPart("Response 1")], timestamp=None),
    ]

    try:
        async for state in agent.iter(
            environment=environment,
            env_state=None,
            user_prompt="New prompt",
            message_history=message_history,
            toolsets=[],
        ):
            # Cancel after first state
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        pass

    # Verify message history is preserved
    last_state = agent.get_last_state()
    assert last_state is not None
    assert len(last_state.run_ctx.messages) >= len(message_history)

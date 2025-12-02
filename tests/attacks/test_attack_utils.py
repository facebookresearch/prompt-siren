# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for edoardos_internship_2025.attacks.attack_utils module."""

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest
from prompt_siren.agents.plain import PlainAgent, PlainAgentConfig
from prompt_siren.agents.states import EndState, InjectableModelRequestState
from prompt_siren.attacks.attack_utils import (
    _make_fake_context,
    get_history_with_attack,
    run_tool_history,
    run_until_injectable,
)
from prompt_siren.types import InjectableToolReturnPart, StrContentAttack
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from ..conftest import MockEnvironment, MockEnvState

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


# Tests for make_fake_context
class TestMakeFakeContext:
    """Tests for the make_fake_context function."""

    async def test_make_fake_context_basic(self, mock_env_state: MockEnvState):
        """Test that make_fake_context creates a RunContext with provided env_state."""
        context = _make_fake_context(mock_env_state)

        assert isinstance(context, RunContext)
        assert context.deps == mock_env_state
        assert isinstance(context.model, TestModel)
        assert isinstance(context.usage, RunUsage)

    async def test_make_fake_context_different_env_state(self):
        """Test make_fake_context with different dependency types."""
        string_env_state = "string_dependency"
        context = _make_fake_context(string_env_state)
        assert context.deps == string_env_state

        dict_env_state = {"key": "value"}
        context = _make_fake_context(dict_env_state)
        assert context.deps == dict_env_state

        none_env_state = None
        context = _make_fake_context(none_env_state)
        assert context.deps is None

    async def test_make_fake_context_uses_test_model(self, mock_env_state: MockEnvState):
        """Test that make_fake_context always uses TestModel."""
        context = _make_fake_context(mock_env_state)
        assert isinstance(context.model, TestModel)


# Tests for get_history_with_attack
class TestGetHistoryWithAttack:
    """Tests for the get_history_with_attack function."""

    async def test_get_history_with_attack_success(self, mock_environment, mock_env_state):
        """Test successful attack rendering."""
        # Create history with a tool call in the last message
        history = [
            ModelRequest.user_text_prompt("User message"),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mock_tool_1",
                        args={"input_text": "test"},
                        tool_call_id="call_1",
                    )
                ]
            ),
        ]

        attacks = {"vector1": StrContentAttack(content="malicious_payload")}

        def mock_tool_1(input_text: str) -> str:
            return f"Response: {input_text} {{INJECT:vector1}}"

        toolsets: list[FunctionToolset[MockEnvState]] = [FunctionToolset([mock_tool_1])]

        result_history = await get_history_with_attack(
            mock_environment, mock_env_state, toolsets, history, attacks
        )

        # Should have original history + rendered attack response
        assert len(result_history) == len(history) + 1
        assert result_history[:-1] == history  # Original history unchanged

        last_message = result_history[-1]
        assert isinstance(last_message, ModelRequest)
        assert len(last_message.parts) == 1
        assert isinstance(last_message.parts[0], ToolReturnPart)
        assert last_message.parts[0].content == "Response: test malicious_payload"

    async def test_get_history_with_attack_invalid_last_message(
        self, mock_environment, mock_env_state
    ):
        """Test ValueError for non-ModelResponse last message."""
        history = [ModelRequest(parts=[UserPromptPart(content="User message")])]
        attacks = {"vector1": StrContentAttack(content="payload")}
        toolsets: list[FunctionToolset[MockEnvState]] = []

        with pytest.raises(ValueError, match="The last message should be a tool call"):
            await get_history_with_attack(
                mock_environment, mock_env_state, toolsets, history, attacks
            )

    async def test_get_history_with_attack_no_tool_calls(self, mock_environment, mock_env_state):
        """Test ValueError when last message has no tool calls."""
        history = [ModelResponse(parts=[TextPart(content="Just text, no tool calls")])]
        attacks = {"vector1": StrContentAttack(content="payload")}
        toolsets: list[FunctionToolset[MockEnvState]] = []

        with pytest.raises(ValueError, match="The last message should contain a tool call"):
            await get_history_with_attack(
                mock_environment, mock_env_state, toolsets, history, attacks
            )

    async def test_get_history_with_attack_immutable(self, mock_environment, mock_env_state):
        """Test that original history is not modified."""
        original_history = [
            ModelRequest(parts=[UserPromptPart(content="User message")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mock_tool_1",
                        args={"input_text": "test"},
                        tool_call_id="call_1",
                    )
                ]
            ),
        ]

        # Create a copy to compare later
        history_copy = original_history.copy()
        attacks = {"vector1": StrContentAttack(content="payload")}

        def mock_tool_1(input_text: str) -> str:
            return f"Response: {input_text}"

        toolsets: list[FunctionToolset[MockEnvState]] = [FunctionToolset([mock_tool_1])]

        result_history = await get_history_with_attack(
            mock_environment, mock_env_state, toolsets, original_history, attacks
        )

        # Original history should be unchanged
        assert original_history == history_copy
        assert result_history != original_history  # But result should be different

    async def test_get_history_with_attack_multiple_tool_calls_in_last_message(
        self, mock_environment, mock_env_state
    ):
        """Test get_history_with_attack with multiple tool calls in the last message."""
        history = [
            ModelRequest(parts=[UserPromptPart(content="User message")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mock_tool_1",
                        args={"input_text": "test1"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="mock_tool_2",
                        args={"query": "test2"},
                        tool_call_id="call_2",
                    ),
                ]
            ),
        ]

        attacks = {
            "vector1": StrContentAttack(content="payload1"),
            "vector2": StrContentAttack(content="payload2"),
        }

        def mock_tool_1(input_text: str) -> str:
            return f"Tool1: {input_text} {{INJECT:vector1}}"

        def mock_tool_2(query: str) -> str:
            return f"Tool2: {query} {{INJECT:vector2}}"

        toolsets: list[FunctionToolset[MockEnvState]] = [
            FunctionToolset([mock_tool_1, mock_tool_2])
        ]

        result_history = await get_history_with_attack(
            mock_environment, mock_env_state, toolsets, history, attacks
        )

        # Should have original history + one message with multiple tool returns
        assert len(result_history) == len(history) + 1

        last_message = result_history[-1]
        assert isinstance(last_message, ModelRequest)
        assert len(last_message.parts) == 2  # Two tool return parts

        for part in last_message.parts:
            assert isinstance(part, ToolReturnPart)

        assert last_message.parts[0].content == "Tool1: test1 payload1"
        assert last_message.parts[1].content == "Tool2: test2 payload2"

    async def test_get_history_with_attack_retry_prompt_in_last_message(
        self, mock_environment, mock_env_state
    ):
        """Test get_history_with_attack handles RetryPromptPart in last message."""
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="invalid_tool",  # This will generate a RetryPromptPart
                        args={"param": "value"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="mock_tool_1",
                        args={"input_text": "test"},
                        tool_call_id="call_2",
                    ),
                ]
            )
        ]

        attacks = {"vector1": StrContentAttack(content="payload")}

        def mock_tool_1(input_text: str) -> str:
            return f"Response: {input_text} {{INJECT:vector1}}"

        toolsets: list[FunctionToolset[MockEnvState]] = [FunctionToolset([mock_tool_1])]

        result_history = await get_history_with_attack(
            mock_environment, mock_env_state, toolsets, history, attacks
        )

        last_message = result_history[-1]
        assert isinstance(last_message, ModelRequest)
        assert len(last_message.parts) == 2  # RetryPromptPart + ToolReturnPart

        assert isinstance(last_message.parts[0], RetryPromptPart)
        assert isinstance(last_message.parts[1], ToolReturnPart)

    async def test_get_history_with_attack_empty_attack(self, mock_environment, mock_env_state):
        """Test get_history_with_attack with empty attack dict."""
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="mock_tool_1",
                        args={"input_text": "test"},
                        tool_call_id="call_1",
                    )
                ]
            )
        ]

        attacks = {}  # Empty attacks

        def mock_tool_1(input_text: str) -> str:
            return f"Response: {input_text}"

        toolsets: list[FunctionToolset[MockEnvState]] = [FunctionToolset([mock_tool_1])]

        result_history = await get_history_with_attack(
            mock_environment, mock_env_state, toolsets, history, attacks
        )

        # Should still work but with no attack injections
        assert len(result_history) == len(history) + 1
        last_message = result_history[-1]
        assert isinstance(last_message, ModelRequest)
        assert len(last_message.parts) == 1
        assert isinstance(last_message.parts[0], ToolReturnPart)
        assert last_message.parts[0].content == "Response: test"


# Tests for run_tool_history
class TestRunToolHistory:
    """Tests for the run_tool_history function."""

    async def test_run_tool_history_empty_history(self, mock_env_state: MockEnvState):
        """Test run_tool_history with empty message history."""
        ctx = _make_fake_context(mock_env_state)
        toolsets = []
        history = []

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Should return the same context unchanged
        assert result_ctx == ctx
        assert result_ctx.deps == mock_env_state

    async def test_run_tool_history_no_tool_calls(self, mock_env_state: MockEnvState):
        """Test run_tool_history with history containing no tool calls."""
        ctx = _make_fake_context(mock_env_state)

        def mock_tool(input_text: str) -> str:
            return f"Tool called with: {input_text}"

        toolsets: list[FunctionToolset[MockEnvState]] = [FunctionToolset([mock_tool])]

        # History with only text messages, no tool calls
        history = [
            ModelRequest.user_text_prompt("Hello"),
            ModelResponse(parts=[TextPart("Hi there!")]),
            ModelRequest.user_text_prompt("How are you?"),
            ModelResponse(parts=[TextPart("I'm doing well, thanks!")]),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Should return the same context since no tools were called
        assert result_ctx == ctx
        assert result_ctx.deps == mock_env_state

    async def test_run_tool_history_single_tool_call(self):
        """Test run_tool_history with single tool call in history."""
        # Create a stateful mock dependency that tools can modify
        stateful_env_state = MockEnvState(value="initial")
        ctx = _make_fake_context(stateful_env_state)

        # Track tool calls to verify execution
        tool_calls = []

        def stateful_tool(ctx: RunContext[MockEnvState], input_text: str) -> str:
            # Modify the deps to prove the tool was called
            ctx.deps.value = f"modified_by_{input_text}"
            tool_calls.append(("stateful_tool", input_text))
            return f"Tool result: {input_text}"

        toolsets = [FunctionToolset([stateful_tool])]

        # History with a tool call
        history = [
            ModelRequest.user_text_prompt("Please use the tool"),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="stateful_tool",
                        args={"input_text": "test_input"},
                        tool_call_id="call_1",
                    )
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Verify tool was called and state was modified
        assert len(tool_calls) == 1
        assert tool_calls[0] == ("stateful_tool", "test_input")
        assert result_ctx.deps.value == "modified_by_test_input"

    async def test_run_tool_history_multiple_tool_calls_different_messages(self):
        """Test run_tool_history with multiple tool calls across different messages."""
        stateful_env_state = MockEnvState(value="initial")
        ctx = _make_fake_context(stateful_env_state)

        tool_calls = []

        # Create agent with multiple tools
        agent = Agent("test", deps_type=MockEnvState)

        @agent.tool
        def tool_one(ctx: RunContext[MockEnvState], text: str) -> str:
            ctx.deps.value += f"_tool1_{text}"
            tool_calls.append(("tool_one", text))
            return f"Tool1: {text}"

        @agent.tool
        def tool_two(ctx: RunContext[MockEnvState], query: str) -> str:
            ctx.deps.value += f"_tool2_{query}"
            tool_calls.append(("tool_two", query))
            return f"Tool2: {query}"

        # Explicitly type toolsets for the type checker
        toolsets: Sequence[AbstractToolset[MockEnvState]] = cast(
            Sequence[AbstractToolset[MockEnvState]], agent.toolsets
        )

        # History with tool calls in different messages
        history = [
            ModelRequest.user_text_prompt("First request"),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_one",
                        args={"text": "first"},
                        tool_call_id="call_1",
                    )
                ]
            ),
            ModelRequest(parts=[ToolReturnPart("tool_one", "Tool1: first", "call_1")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_two",
                        args={"query": "second"},
                        tool_call_id="call_2",
                    )
                ]
            ),
            ModelRequest(parts=[ToolReturnPart("tool_two", "Tool2: second", "call_2")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_one",
                        args={"text": "third"},
                        tool_call_id="call_3",
                    )
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Verify all tools were called in correct order
        assert len(tool_calls) == 3
        assert tool_calls[0] == ("tool_one", "first")
        assert tool_calls[1] == ("tool_two", "second")
        assert tool_calls[2] == ("tool_one", "third")

        # Verify state was modified by all tool calls
        assert result_ctx.deps is not None  # Ensure deps is not None for type checker
        assert result_ctx.deps.value == "initial_tool1_first_tool2_second_tool1_third"

    async def test_run_tool_history_multiple_tool_calls_same_message(self):
        """Test run_tool_history with multiple tool calls in the same message."""
        stateful_env_state = MockEnvState(value="start")
        ctx = _make_fake_context(stateful_env_state)

        tool_calls = []

        # Create agent with tools
        agent = Agent("test", deps_type=MockEnvState)

        @agent.tool
        def append_tool(ctx: RunContext[MockEnvState], suffix: str) -> str:
            ctx.deps.value += f"_{suffix}"
            tool_calls.append(("append_tool", suffix))
            return f"Appended: {suffix}"

        # Explicitly type toolsets for the type checker
        toolsets: Sequence[AbstractToolset[MockEnvState]] = cast(
            Sequence[AbstractToolset[MockEnvState]], agent.toolsets
        )

        # History with multiple tool calls in same message
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="append_tool",
                        args={"suffix": "A"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="append_tool",
                        args={"suffix": "B"},
                        tool_call_id="call_2",
                    ),
                    ToolCallPart(
                        tool_name="append_tool",
                        args={"suffix": "C"},
                        tool_call_id="call_3",
                    ),
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Verify all tools were called
        assert len(tool_calls) == 3
        assert tool_calls[0] == ("append_tool", "A")
        assert tool_calls[1] == ("append_tool", "B")
        assert tool_calls[2] == ("append_tool", "C")

        # Verify state shows all modifications
        assert result_ctx.deps is not None  # Ensure deps is not None for type checker
        assert result_ctx.deps.value == "start_A_B_C"

    async def test_run_tool_history_nonexistent_tools_ignored(self):
        """Test run_tool_history ignores calls to non-existent tools."""
        stateful_env_state = MockEnvState(value="initial")
        ctx = _make_fake_context(stateful_env_state)

        tool_calls = []

        def existing_tool(ctx: RunContext[MockEnvState], data: str) -> str:
            ctx.deps.value = f"called_with_{data}"
            tool_calls.append(("existing_tool", data))
            return f"Result: {data}"

        toolsets = [FunctionToolset([existing_tool])]

        # History with both existing and non-existing tool calls
        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="nonexistent_tool",
                        args={"param": "ignored"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="existing_tool",
                        args={"data": "valid"},
                        tool_call_id="call_2",
                    ),
                    ToolCallPart(
                        tool_name="another_fake_tool",
                        args={"x": "y"},
                        tool_call_id="call_3",
                    ),
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Only the existing tool should have been called
        assert len(tool_calls) == 1
        assert tool_calls[0] == ("existing_tool", "valid")
        assert result_ctx.deps.value == "called_with_valid"

    async def test_run_tool_history_multiple_toolsets(self):
        """Test run_tool_history with multiple toolsets."""
        stateful_env_state = MockEnvState(value="start")
        ctx = _make_fake_context(stateful_env_state)

        tool_calls = []

        def tool_from_agent1(ctx: RunContext[MockEnvState], input_val: str) -> str:
            ctx.deps.value += f"_agent1_{input_val}"
            tool_calls.append(("tool_from_agent1", input_val))
            return f"Agent1: {input_val}"

        def tool_from_agent2(ctx: RunContext[MockEnvState], input_val: str) -> str:
            ctx.deps.value += f"_agent2_{input_val}"
            tool_calls.append(("tool_from_agent2", input_val))
            return f"Agent2: {input_val}"

        # Combine toolsets
        toolsets = [
            FunctionToolset([tool_from_agent1]),
            FunctionToolset([tool_from_agent2]),
        ]

        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="tool_from_agent1",
                        args={"input_val": "first"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="tool_from_agent2",
                        args={"input_val": "second"},
                        tool_call_id="call_2",
                    ),
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)
        result_ctx = await run_tool_history(ctx, toolsets)

        # Should find and call tools in ALL toolsets that have matching names
        assert (
            len(tool_calls) == 2
        )  # agent1.tool_from_agent1, agent2.tool_from_agent1, agent2.tool_from_agent2
        assert tool_calls[0] == ("tool_from_agent1", "first")  # From agent1
        assert tool_calls[1] == ("tool_from_agent2", "second")  # From agent2

        # Verify all tools executed and modified state
        assert result_ctx.deps.value == "start_agent1_first_agent2_second"

    async def test_run_tool_history_tool_call_with_context_dependency(self):
        """Test that tools are called with the correct context and can access the environment state."""
        original_env_state = MockEnvState(value="original")
        ctx = _make_fake_context(original_env_state)

        received_contexts = []

        def context_dependent_tool(ctx: RunContext[MockEnvState], multiplier: int) -> str:
            # Store the context we received to verify it's correct
            received_contexts.append(ctx)
            # Use the context deps
            current_value = ctx.deps.value
            ctx.deps.value = f"{current_value}_x{multiplier}"
            return f"Multiplied by {multiplier}"

        toolsets = [FunctionToolset([context_dependent_tool])]

        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="context_dependent_tool",
                        args={"multiplier": 5},
                        tool_call_id="call_1",
                    )
                ]
            ),
        ]

        ctx = replace(ctx, messages=history)

        result_ctx = await run_tool_history(ctx, toolsets)

        # Verify tool received the correct context
        assert len(received_contexts) == 1
        received_ctx = received_contexts[0]
        assert received_ctx.deps == original_env_state
        assert received_ctx.model == ctx.model
        assert received_ctx.usage == ctx.usage

        # Verify the tool modified the deps correctly
        assert result_ctx.deps.value == "original_x5"


# Tests for run_until_injectable
class TestRunUntilInjectable:
    """Tests for the run_until_injectable function."""

    async def test_run_until_injectable_found(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test run_until_injectable finds injectable outputs."""

        async def injectable_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Result with {INJECT:vector1} placeholder"

        toolset = FunctionToolset[MockEnvState]([injectable_tool])

        # Model that calls the injectable tool
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name="injectable_tool", args={})])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Create initial state
        initial_state = agent.create_initial_request_state(
            mock_environment, mock_env_state, user_prompt="Call injectable tool"
        )

        # Run until injectable
        state = await run_until_injectable(
            agent,
            initial_state,
            toolsets=[toolset],
        )

        # Check we found the injectable state
        assert isinstance(state, InjectableModelRequestState)
        injectable_parts = state.injectable_model_request_parts

        # Check messages structure in state.run_ctx
        run_ctx = state.run_ctx
        assert len(run_ctx.messages) == 2  # User prompt, model response with tool call
        assert isinstance(run_ctx.messages[0], ModelRequest)  # User prompt
        assert isinstance(run_ctx.messages[0].parts[0], UserPromptPart)
        assert isinstance(run_ctx.messages[1], ModelResponse)  # Model response with tool call
        assert isinstance(run_ctx.messages[1].parts[0], ToolCallPart)
        assert run_ctx.messages[1].parts[0].tool_name == "injectable_tool"

        # Check we found the injectable part
        assert len(injectable_parts) == 1
        assert isinstance(injectable_parts[0], InjectableToolReturnPart)
        assert injectable_parts[0].tool_name == "injectable_tool"
        assert (
            injectable_parts[0].content == "Result with {INJECT:vector1} placeholder"
        )  # Raw content
        assert injectable_parts[0].vector_ids == ["vector1"]  # Found vector IDs

    async def test_run_until_injectable_not_found(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test run_until_injectable returns EndState when no injectable found."""

        async def normal_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Normal result without placeholders"

        toolset = FunctionToolset[MockEnvState]([normal_tool])

        # Model that only calls non-injectable tools then stops
        call_count = 0

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="normal_tool", args={})])
            return ModelResponse(parts=[TextPart("Done, no more tools")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Create initial state
        initial_state = agent.create_initial_request_state(
            mock_environment, mock_env_state, user_prompt="Call normal tool"
        )

        # Run until injectable
        state = await run_until_injectable(agent, initial_state, toolsets=[toolset])

        # Should return EndState when no injectable outputs found
        assert isinstance(state, EndState)

    async def test_run_until_injectable_multiple_tools(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test run_until_injectable with multiple tools, some injectable."""

        async def normal_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Normal result"

        async def injectable_tool1(ctx: RunContext[MockEnvState]) -> str:
            return "Result with {INJECT:vector1}"

        async def injectable_tool2(ctx: RunContext[MockEnvState]) -> str:
            return "Result with {INJECT:vector2} and {INJECT:vector3}"

        toolset = FunctionToolset[MockEnvState]([normal_tool, injectable_tool1, injectable_tool2])

        # Model that calls multiple tools at once
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="normal_tool", args={}),
                    ToolCallPart(tool_name="injectable_tool1", args={}),
                    ToolCallPart(tool_name="injectable_tool2", args={}),
                ]
            )

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Create initial state
        initial_state = agent.create_initial_request_state(
            mock_environment, mock_env_state, user_prompt="Call all tools"
        )

        # Run until injectable
        state = await run_until_injectable(agent, initial_state, toolsets=[toolset])

        assert isinstance(state, InjectableModelRequestState)
        injectable_parts = state.injectable_model_request_parts

        # Check we found all parts - now only injectable parts are included
        assert len(injectable_parts) == 3

        # First tool: normal_tool - not injectable but included as ToolReturnPart
        assert isinstance(injectable_parts[0], ToolReturnPart)
        assert injectable_parts[0].tool_name == "normal_tool"
        assert injectable_parts[0].content == "Normal result"

        # Second tool: injectable_tool1 - has vector1
        assert isinstance(injectable_parts[1], InjectableToolReturnPart)
        assert injectable_parts[1].tool_name == "injectable_tool1"
        assert injectable_parts[1].content == "Result with {INJECT:vector1}"
        assert injectable_parts[1].vector_ids == ["vector1"]

        # Third tool: injectable_tool2 - has vector2 and vector3
        assert isinstance(injectable_parts[2], InjectableToolReturnPart)
        assert injectable_parts[2].tool_name == "injectable_tool2"
        assert injectable_parts[2].content == "Result with {INJECT:vector2} and {INJECT:vector3}"
        assert set(injectable_parts[2].vector_ids) == {"vector2", "vector3"}

    async def test_complex_injection_scenario(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test complex scenario with multiple rounds before finding injection."""

        async def search_tool(ctx: RunContext[MockEnvState], query: str) -> str:
            return f"Search results for: {query}"

        async def process_tool(ctx: RunContext[MockEnvState], data: str) -> str:
            if "special" in data:
                return f"Processed {data} with {{INJECT:vector1}}"
            return f"Processed {data}"

        toolset = FunctionToolset[MockEnvState]([search_tool, process_tool])

        # Model that takes multiple rounds to find injectable content
        round_num = 0

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal round_num
            round_num += 1

            if round_num == 1:  # First: search
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="search_tool", args={"query": "test"})]
                )
            if round_num == 2:  # Second: process normal
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="process_tool", args={"data": "normal"})]
                )
            if round_num == 3:  # Third: process with injection
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="process_tool", args={"data": "special"})]
                )
            # Should not reach here, but return empty response to satisfy type checker
            return ModelResponse(parts=[TextPart("Unexpected round")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Create initial state
        initial_state = agent.create_initial_request_state(
            mock_environment, mock_env_state, user_prompt="Find special data"
        )

        # Run until injectable
        state = await run_until_injectable(agent, initial_state, toolsets=[toolset])

        # Check we went through all rounds
        assert round_num == 3
        assert isinstance(state, InjectableModelRequestState)
        injectable_parts = state.injectable_model_request_parts
        assert len(injectable_parts) == 1
        assert isinstance(injectable_parts[0], InjectableToolReturnPart)
        assert injectable_parts[0].tool_name == "process_tool"
        assert "{INJECT:vector1}" in injectable_parts[0].content

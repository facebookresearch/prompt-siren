# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for prompt_siren.agents.plain module."""

import pytest
from prompt_siren.agents.plain import PlainAgent, PlainAgentConfig
from prompt_siren.agents.states import (
    EndState,
    ModelRequestState,
    ModelResponseState,
    NoPreviousStateError,
)
from pydantic_ai import models
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
from pydantic_ai.toolsets import FunctionToolset

from ..conftest import MockEnvironment, MockEnvState

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


# Test Agent class
class TestAgent:
    """Tests for the Agent class."""

    async def test_agent_run_simple(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent.run with simple text response."""

        # Use FunctionModel for predictable behavior
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Hello from agent")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )
        result = await agent.run(
            mock_environment,
            mock_env_state,
            user_prompt="Hello",
            toolsets=[],
        )

        assert isinstance(result, RunContext)
        assert len(result.messages) == 2  # User prompt + model response
        assert isinstance(result.messages[0].parts[0], UserPromptPart)
        assert result.messages[0].parts[0].content == "Hello"
        assert isinstance(result.messages[1].parts[0], TextPart)
        assert result.messages[1].parts[0].content == "Hello from agent"

    async def test_agent_run_with_tool_call(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent.run with tool calls."""

        async def greeting_tool(ctx: RunContext[MockEnvState], name: str) -> str:
            return f"Hello, {name}! From {ctx.deps.value}"

        toolset = FunctionToolset[MockEnvState]([greeting_tool])

        # Model that calls the tool then responds
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="greeting_tool", args={"name": "Alice"})]
                )
            # After tool response
            return ModelResponse(parts=[TextPart("Tool was called successfully")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )
        result = await agent.run(
            mock_environment,
            mock_env_state,
            user_prompt="Say hello to Alice",
            toolsets=[toolset],
        )

        # Check we have all expected messages
        assert len(result.messages) == 4  # User prompt, tool call, tool result, final response
        assert isinstance(result.messages[2], ModelRequest)  # Tool result
        assert isinstance(result.messages[2].parts[0], ToolReturnPart)
        assert result.messages[2].parts[0].content == "Hello, Alice! From test_env_state"

    async def test_agent_iter_yields_contexts(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent.iter yields contexts at each step."""

        async def slow_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Tool completed"

        toolset = FunctionToolset[MockEnvState]([slow_tool])

        # Model that calls tool
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(parts=[ToolCallPart(tool_name="slow_tool", args={})])
            # After tool
            return ModelResponse(parts=[TextPart("Done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )
        execution_states = [
            ctx
            async for ctx in agent.iter(
                mock_environment,
                mock_env_state,
                user_prompt=["Run slow tool"],
                toolsets=[toolset],
            )
        ]

        # With the state machine implementation, we get these states:
        # 0. Initial ModelRequestState with user prompt
        # 1. ModelResponseState with first model response (tool call)
        # 2. ModelRequestState with tool result
        # 3. ModelResponseState with final model response
        # 4. EndState
        assert len(execution_states) == 5

        # State 0: Initial ModelRequestState with user prompt
        assert isinstance(execution_states[0], ModelRequestState)
        assert len(execution_states[0].run_ctx.messages) == 0

        # State 1: ModelResponseState with first model response (tool call)
        assert isinstance(execution_states[1], ModelResponseState)
        assert len(execution_states[1].run_ctx.messages) == 2  # User prompt + first model response

        # State 2: ModelRequestState with tool result (not yet added to messages)
        assert isinstance(execution_states[2], ModelRequestState)
        assert (
            len(execution_states[2].run_ctx.messages) == 2
        )  # Still just user prompt + first model response

        # State 3: ModelResponseState with final model response
        assert isinstance(execution_states[3], ModelResponseState)
        assert len(execution_states[3].run_ctx.messages) == 4  # + final response

        # State 4: EndState
        assert isinstance(execution_states[4], EndState)
        assert len(execution_states[4].run_ctx.messages) == 4

    async def test_agent_run_with_message_history(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent.run with pre-existing message history."""
        # Pre-existing history
        history = [
            ModelRequest.user_text_prompt("Previous question"),
            ModelResponse(parts=[TextPart("Previous answer")]),
        ]

        # Model continues conversation
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Continuing the conversation")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )
        result = await agent.run(
            mock_environment,
            mock_env_state,
            user_prompt="Follow up question",
            message_history=history,
            toolsets=[],
        )

        # Check history was preserved and extended
        assert len(result.messages) == 4  # 2 from history + user prompt + response
        assert result.messages[0] == history[0]
        assert result.messages[1] == history[1]

    async def test_agent_iter_no_iteration_executed(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent.run raises error when iter doesn't execute."""

        # Create a subclass that overrides the iter method to not yield anything
        class NoYieldAgent(PlainAgent):
            async def iter(self, *args, **kwargs):
                if False:  # This ensures it's a generator but never yields
                    yield

        # Use our subclass instead of the original
        agent = NoYieldAgent(
            _config=PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()),
        )
        with pytest.raises(RuntimeError, match="No loop iteration was executed"):
            await agent.run(
                mock_environment,
                mock_env_state,
                user_prompt="Test",
                toolsets=[],
            )


class TestAgentIntegration:
    """Integration tests for Agent with more complex scenarios."""

    async def test_agent_with_retry_prompt_part(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test Agent handles RetryPromptPart from unknown tools."""

        async def existing_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Existing tool result"

        toolset = FunctionToolset[MockEnvState]([existing_tool])

        # Model that calls both existing and non-existing tools
        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name="existing_tool", args={}),
                        ToolCallPart(tool_name="nonexistent_tool", args={}),
                    ]
                )
            # After tool responses
            return ModelResponse(parts=[TextPart("Handled retry")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )
        result = await agent.run(
            mock_environment,
            mock_env_state,
            user_prompt="Call tools",
            toolsets=[toolset],
        )

        # Check that both tool results are in the messages
        tool_result_message = result.messages[2]
        assert isinstance(tool_result_message, ModelRequest)
        assert len(tool_result_message.parts) == 2
        assert isinstance(tool_result_message.parts[0], ToolReturnPart)
        assert isinstance(tool_result_message.parts[1], RetryPromptPart)


class TestPrevIter:
    """Tests for PlainAgent.prev_state method."""

    async def test_prev_state_basic(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test prev_state returns the previous state correctly."""

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Hello")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect states during iteration
        states = [
            state
            async for state in agent.iter(
                mock_environment, mock_env_state, user_prompt="Test", toolsets=[]
            )
        ]

        # We should have: ModelRequestState -> ModelResponseState -> EndState
        assert len(states) == 3
        assert isinstance(states[0], ModelRequestState)
        assert isinstance(states[1], ModelResponseState)
        assert isinstance(states[2], EndState)

        # Test going back from EndState to ModelResponseState
        prev_state = await agent.prev_state(current_state=states[2], toolsets=[])
        assert isinstance(prev_state, ModelResponseState)
        assert prev_state.run_ctx.messages == states[1].run_ctx.messages

        # Test going back from ModelResponseState to ModelRequestState
        prev_prev_state = await agent.prev_state(current_state=states[1], toolsets=[])
        assert isinstance(prev_prev_state, ModelRequestState)
        assert prev_prev_state.run_ctx.messages == states[0].run_ctx.messages

    async def test_prev_state_no_previous_state(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test prev_state raises NoPreviousStateError when at initial state."""

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Hello")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Get the initial state
        initial_state = None
        async for state in agent.iter(
            mock_environment, mock_env_state, user_prompt="Test", toolsets=[]
        ):
            initial_state = state
            break  # Stop after first state

        assert isinstance(initial_state, ModelRequestState)

        # Trying to go back from initial state should raise error
        with pytest.raises(NoPreviousStateError, match="initial state"):
            await agent.prev_state(current_state=initial_state, toolsets=[])

    async def test_prev_state_with_tool_calls(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test prev_state with states that involve tool calls."""

        call_count = 0

        async def test_tool(ctx: RunContext[MockEnvState]) -> str:
            nonlocal call_count
            call_count += 1
            return f"Tool result {call_count}"

        toolset = FunctionToolset[MockEnvState]([test_tool])

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(parts=[ToolCallPart(tool_name="test_tool", args={})])
            return ModelResponse(parts=[TextPart("Done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect all states
        states = [
            state
            async for state in agent.iter(
                mock_environment,
                mock_env_state,
                user_prompt="Test with tool",
                toolsets=[toolset],
            )
        ]

        # We should have: ModelRequestState -> ModelResponseState -> ModelRequestState -> ModelResponseState -> EndState
        assert len(states) == 5
        assert call_count == 1  # Tool called once during iteration

        # Go back from EndState through the chain
        prev_state = await agent.prev_state(
            current_state=states[4],  # EndState
            toolsets=[toolset],
        )
        assert isinstance(prev_state, ModelResponseState)
        assert call_count == 1  # No additional tool call (snapshottable environment)

        # Go back again
        prev_prev_state = await agent.prev_state(current_state=prev_state, toolsets=[toolset])
        assert isinstance(prev_prev_state, ModelRequestState)
        assert call_count == 1  # Still no additional tool call

    async def test_prev_state_non_snapshottable_environment(
        self, mock_non_snapshottable_environment, mock_env_state: MockEnvState
    ):
        """Test prev_state with non-snapshottable environment replays tools."""
        call_count = 0

        async def stateful_tool(ctx: RunContext[MockEnvState]) -> str:
            nonlocal call_count
            call_count += 1
            return f"State {call_count}"

        toolset = FunctionToolset[MockEnvState]([stateful_tool])

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="stateful_tool", args={})])
            return ModelResponse(parts=[TextPart("Complete")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect states
        states = [
            state
            async for state in agent.iter(
                mock_non_snapshottable_environment,
                mock_env_state,
                user_prompt="Test non-snapshottable",
                toolsets=[toolset],
            )
        ]

        assert call_count == 1  # Initial tool execution

        # Go back with non-snapshottable environment
        await agent.prev_state(current_state=states[-1], toolsets=[toolset])

        # Tool should be replayed because environment isn't snapshottable
        assert call_count == 2

    async def test_prev_state_chain_multiple_steps(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test navigating back multiple steps using prev_state."""

        async def tool_a(ctx: RunContext[MockEnvState]) -> str:
            return "Result A"

        async def tool_b(ctx: RunContext[MockEnvState]) -> str:
            return "Result B"

        toolset = FunctionToolset[MockEnvState]([tool_a, tool_b])

        call_sequence = 0

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_sequence
            if call_sequence == 0:  # First call
                call_sequence += 1
                return ModelResponse(parts=[ToolCallPart(tool_name="tool_a", args={})])
            if call_sequence == 1:  # After first tool
                call_sequence += 1
                return ModelResponse(parts=[ToolCallPart(tool_name="tool_b", args={})])
            # After second tool
            return ModelResponse(parts=[TextPart("All done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect all states
        states = [
            state
            async for state in agent.iter(
                mock_environment,
                mock_env_state,
                user_prompt="Multi-step test",
                toolsets=[toolset],
            )
        ]

        # Should have multiple states from the multi-step interaction
        assert len(states) > 4

        # Navigate back multiple steps
        current = states[-1]  # Start from EndState

        # Go back step by step and verify we can reach each previous state
        for i in range(len(states) - 2, -1, -1):  # Skip the last (current) and go to initial state
            current = await agent.prev_state(current_state=current, toolsets=[toolset])
            # Verify the state type matches what we expect
            assert isinstance(current, type(states[i]))

        # Current should now be the initial state (states[0])
        assert current == states[0]

        # Verify we can't go back from the initial state
        with pytest.raises(NoPreviousStateError):
            await agent.prev_state(current_state=current, toolsets=[toolset])


class TestStateIsolation:
    """Tests for verifying state isolation - each ExecutionState should maintain independent environment state."""

    async def test_state_deps_isolation_with_tools(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test that states maintain independent deps when tools modify them."""

        async def modifying_tool(ctx: RunContext[MockEnvState], new_value: str) -> str:
            # Modify deps in place
            ctx.deps.value = new_value
            return f"Set value to {new_value}"

        toolset = FunctionToolset[MockEnvState]([modifying_tool])

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="modifying_tool",
                            args={"new_value": "modified"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("Done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect all states
        states = [
            state
            async for state in agent.iter(
                mock_environment,
                mock_env_state,
                user_prompt="Test isolation",
                toolsets=[toolset],
            )
        ]

        # Find the ModelResponseState after tool execution (state 1)
        # and the ModelRequestState after tool result (state 2)
        assert (
            len(states) == 5
        )  # Initial, Response(tool call), Request(tool result), Response(final), End
        response_with_tool_call = states[1]
        request_with_tool_result = states[2]

        # The ModelResponseState (before tool execution) should have original deps
        assert response_with_tool_call.run_ctx.deps.value == "test_env_state"

        # The ModelRequestState (after tool execution) should have modified deps
        assert request_with_tool_result.run_ctx.deps.value == "modified"

        # CRITICAL: Going back to the ModelResponseState should still have original deps
        # This verifies that the tool execution didn't retroactively modify previous state's deps
        assert response_with_tool_call.run_ctx.deps.value == "test_env_state"

    async def test_state_branches_independence(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test that continuing from the same state creates independent branches."""

        async def counter_tool(ctx: RunContext[MockEnvState]) -> str:
            # Read and modify deps
            current = int(ctx.deps.value) if ctx.deps.value.isdigit() else 0
            ctx.deps.value = str(current + 1)
            return f"Count: {ctx.deps.value}"

        toolset = FunctionToolset[MockEnvState]([counter_tool])

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:  # First call
                return ModelResponse(parts=[ToolCallPart(tool_name="counter_tool", args={})])
            return ModelResponse(parts=[TextPart("Done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Start with deps value as "0"
        branch_env_state = MockEnvState(value="0")

        # Get the initial ModelResponseState (after first tool call)
        states = [
            state
            async for state in agent.iter(
                mock_environment,
                branch_env_state,
                user_prompt="Test branches",
                toolsets=[toolset],
            )
        ]

        # states[1] is the ModelResponseState with the tool call
        branch_point = states[1]

        # Advance from this state once
        next_state_1 = await agent.next_state(current_state=branch_point, toolsets=[toolset])

        # Advance from the SAME state again - should create independent branch
        next_state_2 = await agent.next_state(current_state=branch_point, toolsets=[toolset])

        # Both branches should have executed the tool, but independently
        # They should both have incremented from 0 to 1 (not 1 to 2, since they branched independently)
        assert next_state_1.run_ctx.deps.value == "1"
        assert next_state_2.run_ctx.deps.value == "1"

        # The original branch point should still be unchanged
        assert branch_point.run_ctx.deps.value == "0"

    async def test_state_chain_preservation(
        self, mock_environment: MockEnvironment, mock_env_state: MockEnvState
    ):
        """Test that navigating through states preserves the state chain correctly."""

        call_count = 0

        async def tracking_tool(ctx: RunContext[MockEnvState]) -> str:
            nonlocal call_count
            call_count += 1
            original = ctx.deps.value
            ctx.deps.value = f"{original}_v{call_count}"
            return f"Modified to {ctx.deps.value}"

        toolset = FunctionToolset[MockEnvState]([tracking_tool])

        def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name="tracking_tool", args={})])
            return ModelResponse(parts=[TextPart("Done")])

        agent = PlainAgent(
            _config=PlainAgentConfig(
                model=FunctionModel(model_response),
                model_settings=ModelSettings(),
            ),
        )

        # Collect all states
        states = [
            state
            async for state in agent.iter(
                mock_environment,
                mock_env_state,
                user_prompt="Test preservation",
                toolsets=[toolset],
            )
        ]

        # We should have: ModelRequestState(initial) -> ModelResponseState(tool call) ->
        #                 ModelRequestState(tool result) -> ModelResponseState(final) -> EndState
        assert len(states) == 5

        # Check that each state has the expected deps value
        assert states[0].run_ctx.deps.value == "test_env_state"  # Initial state
        assert states[1].run_ctx.deps.value == "test_env_state"  # Before tool execution
        assert states[2].run_ctx.deps.value == "test_env_state_v1"  # After tool execution
        assert states[3].run_ctx.deps.value == "test_env_state_v1"  # Final response
        assert states[4].run_ctx.deps.value == "test_env_state_v1"  # End state

        # Now go back and verify states are preserved
        prev_state = await agent.prev_state(current_state=states[4], toolsets=[toolset])
        assert prev_state.run_ctx.deps.value == "test_env_state_v1"

        prev_prev_state = await agent.prev_state(current_state=prev_state, toolsets=[toolset])
        assert prev_prev_state.run_ctx.deps.value == "test_env_state_v1"

        # Go back further
        prev_prev_prev_state = await agent.prev_state(
            current_state=prev_prev_state, toolsets=[toolset]
        )
        assert prev_prev_prev_state.run_ctx.deps.value == "test_env_state"  # Before tool execution

        # Verify original states are still unchanged
        assert states[0].run_ctx.deps.value == "test_env_state"
        assert states[1].run_ctx.deps.value == "test_env_state"
        assert states[2].run_ctx.deps.value == "test_env_state_v1"

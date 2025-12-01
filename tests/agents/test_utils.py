# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for prompt_siren.agents.utils module."""

import pytest
import yaml
from prompt_siren.agents.states import ModelRequestState
from prompt_siren.agents.utils import (
    extract_tool_call_parts,
    handle_tool_calls,
    parts_contain_only_model_request_parts,
    query_model,
    restore_state_context,
    serialize_tool_return_part,
)
from prompt_siren.types import (
    InjectableModelRequestPart,
    InjectableToolReturnPart,
    StrContentAttack,
)
from pydantic_ai import models, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from ..conftest import MockEnvState

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


# Test helper functions
class TestUtils:
    """Tests for the helper functions in agent module."""

    def test_extract_tool_call_parts(self):
        """Test _extract_tool_call_parts extracts tool calls from last message."""
        # Create a context with tool calls in the last message
        tool_call = ToolCallPart(tool_name="test_tool", args={"arg": "value"})
        messages: list[ModelMessage] = [
            ModelRequest.user_text_prompt("test"),
            ModelResponse(parts=[TextPart("response"), tool_call]),
        ]
        ctx = RunContext(
            deps=MockEnvState("test"),
            model=TestModel(),
            usage=RunUsage(),
            messages=messages,
        )

        result = extract_tool_call_parts(ctx)
        assert len(result) == 1
        assert result[0] == tool_call

    async def test_extract_tool_call_parts_wrong_message_type(self):
        """Test _extract_tool_call_parts raises error when last message is not ModelResponse."""
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt("test")]
        ctx = RunContext(
            deps=MockEnvState("test"),
            model=TestModel(),
            usage=RunUsage(),
            messages=messages,
        )

        with pytest.raises(
            ValueError,
            match="Last message in `ctx` should be of type `ModelResponse`",
        ):
            extract_tool_call_parts(ctx)

    async def test_query_model_basic(self, mock_env_state: MockEnvState):
        """Test _query_model sends request to model with tools."""

        async def test_tool(ctx: RunContext[MockEnvState]) -> str:
            return "Tool result"

        toolset = FunctionToolset[MockEnvState]([test_tool])
        ctx = RunContext(deps=mock_env_state, model=TestModel(), usage=RunUsage())

        _, result = await query_model(
            ModelRequest.user_text_prompt("Hey"),
            ctx,
            None,
            ModelSettings(),
            [toolset],
        )

        # Check both request and response were added to messages
        assert len(result.messages) == 2
        assert isinstance(result.messages[0], ModelRequest)
        assert isinstance(result.messages[1], ModelResponse)

    async def test_query_model_with_usage_limits(self, mock_env_state: MockEnvState):
        """Test _query_model respects usage limits."""
        usage = RunUsage()
        usage.requests = 5
        ctx = RunContext(deps=mock_env_state, model=TestModel(), usage=usage)
        usage_limits = UsageLimits(request_limit=5)  # Already at limit

        with pytest.raises(UsageLimitExceeded, match="request_limit"):
            await query_model(
                ModelRequest.user_text_prompt("Hey"),
                ctx,
                usage_limits,
                ModelSettings(),
                [],
            )

    def test_parts_contain_only_model_request_parts(self):
        """Test _parts_contain_only_model_request_parts correctly checks part types."""
        # Create regular parts
        regular_parts: list[ModelRequestPart | InjectableModelRequestPart] = [
            UserPromptPart("test"),
            ToolReturnPart("tool_name", "result", tool_call_id="123"),
        ]

        # Create a mix of regular and injectable parts
        injectable_part = InjectableToolReturnPart(
            tool_name="inject_tool",
            content="result",
            tool_call_id="456",
            default={"vector_1": StrContentAttack("hi")},
        )
        mixed_parts = [*regular_parts, injectable_part]

        # Test with only ModelRequestPart instances
        assert parts_contain_only_model_request_parts(regular_parts) is True

        # Test with mixed part types
        assert parts_contain_only_model_request_parts(mixed_parts) is False


class TestHandleToolCalls:
    """Tests for handle_tool_calls utility function."""

    async def test_handle_tool_calls_snapshottable_copies_deps(
        self, mock_environment, mock_env_state
    ):
        """Test that handle_tool_calls copies deps for snapshottable environments."""

        async def test_tool(ctx: RunContext[MockEnvState]) -> str:
            # Modify deps to verify we're working with a copy
            ctx.deps.value = "modified"
            return "result"

        toolset = FunctionToolset[MockEnvState]([test_tool])
        run_ctx = RunContext(deps=mock_env_state, model=TestModel(), usage=RunUsage())

        # Add a model response with tool call
        tool_call = ToolCallPart(tool_name="test_tool", args={})
        response = ModelResponse(parts=[tool_call])
        run_ctx = RunContext(
            deps=mock_env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[ModelRequest.user_text_prompt("test"), response],
        )

        # Call handle_tool_calls
        _results_parts, new_run_ctx = await handle_tool_calls(
            run_ctx, mock_environment, [tool_call], [toolset]
        )

        # Verify env_state were copied (original should be unchanged)
        assert mock_env_state.value == "test_env_state"
        # New context should have modified env_state
        assert new_run_ctx.deps.value == "modified"
        # Verify contexts are different objects
        assert new_run_ctx.deps is not run_ctx.deps

    async def test_handle_tool_calls_non_snapshottable_uses_same_deps(
        self, mock_non_snapshottable_environment, mock_env_state
    ):
        """Test that handle_tool_calls uses same deps for non-snapshottable environments."""

        async def test_tool(ctx: RunContext[MockEnvState]) -> str:
            ctx.deps.value = "modified"
            return "result"

        toolset = FunctionToolset[MockEnvState]([test_tool])

        tool_call = ToolCallPart(tool_name="test_tool", args={})
        response = ModelResponse(parts=[tool_call])
        run_ctx = RunContext(
            deps=mock_env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[ModelRequest.user_text_prompt("test"), response],
        )

        # Call handle_tool_calls
        _results_parts, new_run_ctx = await handle_tool_calls(
            run_ctx, mock_non_snapshottable_environment, [tool_call], [toolset]
        )

        # For non-snapshottable, should use same deps (no copy)
        assert new_run_ctx.deps is run_ctx.deps
        # Env state should be modified in place
        assert mock_env_state.value == "modified"


class TestRestoreStateContext:
    """Tests for restore_state_context utility function."""

    async def test_restore_snapshottable_returns_unchanged(self, mock_environment, mock_env_state):
        """Test that restore_state_context returns snapshottable state unchanged."""
        run_ctx = RunContext(deps=mock_env_state, model=TestModel(), usage=RunUsage())

        # Create a state
        state = ModelRequestState(
            run_ctx=run_ctx,
            environment=mock_environment,
            model_request=ModelRequest.user_text_prompt("test"),
            _previous_state=None,
        )

        # Restore should return same state for snapshottable
        restored = await restore_state_context(state, [])

        assert restored is state
        assert restored.run_ctx is state.run_ctx
        assert restored.environment is state.environment

    async def test_restore_non_snapshottable_resets_and_replays(
        self, mock_non_snapshottable_environment, mock_env_state
    ):
        """Test that restore_state_context resets and replays tools for non-snapshottable."""
        call_count = 0

        async def tracked_tool(ctx: RunContext[MockEnvState]) -> str:
            nonlocal call_count
            call_count += 1
            ctx.deps.value = f"call_{call_count}"
            return f"result_{call_count}"

        toolset = FunctionToolset[MockEnvState]([tracked_tool])

        # Modify env_state before creating state
        mock_env_state.value = "modified"

        # Create run context with tool history
        tool_call = ToolCallPart(tool_name="tracked_tool", args={})
        response = ModelResponse(parts=[tool_call])
        run_ctx = RunContext(
            deps=mock_env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[
                ModelRequest.user_text_prompt("test"),
                response,
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "tracked_tool",
                            "result_1",
                            tool_call_id=tool_call.tool_call_id,
                        )
                    ]
                ),
            ],
        )

        state = ModelRequestState(
            run_ctx=run_ctx,
            environment=mock_non_snapshottable_environment,
            model_request=ModelRequest.user_text_prompt("test"),
            _previous_state=None,
        )

        # Restore should reset and replay tools from history
        restored = await restore_state_context(state, [toolset])

        # Tool should be called once during replay
        assert call_count == 1
        # Env state should be reset and updated by replay
        assert restored.run_ctx.deps.value == "call_1"

    async def test_restore_non_snapshottable_empty_history(
        self, mock_non_snapshottable_environment, mock_env_state
    ):
        """Test restore_state_context with empty tool history."""
        run_ctx = RunContext(deps=mock_env_state, model=TestModel(), usage=RunUsage(), messages=[])

        state = ModelRequestState(
            run_ctx=run_ctx,
            environment=mock_non_snapshottable_environment,
            model_request=ModelRequest.user_text_prompt("test"),
            _previous_state=None,
        )

        # Restore with no tools in history
        restored = await restore_state_context(state, [])

        # Should still work, just reset deps
        assert restored.run_ctx.deps.value == "test_env_state"

    async def test_restore_with_multiple_tools_in_history(
        self, mock_non_snapshottable_environment, mock_env_state
    ):
        """Test restore_state_context replays multiple tools correctly."""
        call_sequence = []

        async def tool_a(ctx: RunContext[MockEnvState]) -> str:
            call_sequence.append("a")
            return "result_a"

        async def tool_b(ctx: RunContext[MockEnvState]) -> str:
            call_sequence.append("b")
            return "result_b"

        toolset = FunctionToolset[MockEnvState]([tool_a, tool_b])

        # Create history with both tools
        tool_call_a = ToolCallPart(tool_name="tool_a", args={})
        tool_call_b = ToolCallPart(tool_name="tool_b", args={})
        run_ctx = RunContext(
            deps=mock_env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[
                ModelRequest.user_text_prompt("test"),
                ModelResponse(parts=[tool_call_a]),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "tool_a",
                            "result_a",
                            tool_call_id=tool_call_a.tool_call_id,
                        )
                    ]
                ),
                ModelResponse(parts=[tool_call_b]),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "tool_b",
                            "result_b",
                            tool_call_id=tool_call_b.tool_call_id,
                        )
                    ]
                ),
            ],
        )

        state = ModelRequestState(
            run_ctx=run_ctx,
            environment=mock_non_snapshottable_environment,
            model_request=ModelRequest.user_text_prompt("test"),
            _previous_state=None,
        )

        # Clear call sequence (from initial execution)
        call_sequence.clear()

        # Restore should replay both tools
        await restore_state_context(state, [toolset])

        # Both tools should be replayed in order
        assert call_sequence == ["a", "b"]

    async def test_restore_updates_contexts_consistently(
        self, mock_non_snapshottable_environment, mock_env_state
    ):
        """Test that restore_state_context updates both run_ctx and env_ctx consistently."""

        async def modifying_tool(ctx: RunContext[MockEnvState]) -> str:
            ctx.deps.value = "tool_modified"
            return "result"

        toolset = FunctionToolset[MockEnvState]([modifying_tool])

        tool_call = ToolCallPart(tool_name="modifying_tool", args={})
        run_ctx = RunContext(
            deps=mock_env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[
                ModelRequest.user_text_prompt("test"),
                ModelResponse(parts=[tool_call]),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "modifying_tool",
                            "result",
                            tool_call_id=tool_call.tool_call_id,
                        )
                    ]
                ),
            ],
        )

        state = ModelRequestState(
            run_ctx=run_ctx,
            environment=mock_non_snapshottable_environment,
            model_request=ModelRequest.user_text_prompt("test"),
            _previous_state=None,
        )

        # Restore state
        restored = await restore_state_context(state, [toolset])

        # For non-snapshottable, run_ctx should use the same deps object that was modified
        # Note: The environment holds the original deps, but run_ctx gets the modified version
        assert restored.run_ctx.deps.value == "tool_modified"


class TestSerializeToolOutput:
    """Tests for serialize_tool_output utility function."""

    def test_json_mode_returns_unchanged(self):
        """Test that JSON mode returns the tool result unchanged."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content={"key": "value", "nested": {"data": [1, 2, 3]}},
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "json")

        # Should return exact same object
        assert result is tool_result
        assert result.content == {"key": "value", "nested": {"data": [1, 2, 3]}}

    def test_yaml_mode_string_content_unchanged(self):
        """Test that YAML mode returns string content as-is without YAML encoding."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content="This is a plain string response",
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # String content should be returned as-is
        assert result.content == "This is a plain string response"
        assert result.tool_name == "test_tool"
        assert result.tool_call_id == "123"

    def test_yaml_mode_dict_converts_to_yaml(self):
        """Test that YAML mode converts dict content to YAML string."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content={"status": "success", "count": 42, "items": ["a", "b", "c"]},
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Content should be converted to YAML string
        assert isinstance(result.content, str)

        # Parse it back to verify it's valid YAML
        parsed = yaml.safe_load(result.content)
        assert parsed == {"status": "success", "count": 42, "items": ["a", "b", "c"]}

        # YAML string should be human-readable
        assert "status: success" in result.content
        assert "count: 42" in result.content

    def test_yaml_mode_list_converts_to_yaml(self):
        """Test that YAML mode converts list content to YAML string."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content=[{"id": 1, "name": "first"}, {"id": 2, "name": "second"}],
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Content should be converted to YAML string
        assert isinstance(result.content, str)

        # Parse it back to verify structure is preserved
        parsed = yaml.safe_load(result.content)
        assert parsed == [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]

    def test_yaml_mode_nested_structures(self):
        """Test YAML mode with deeply nested data structures."""
        complex_data = {
            "user": {
                "name": "Alice",
                "settings": {
                    "preferences": {"theme": "dark", "notifications": True},
                    "history": [
                        {"action": "login", "timestamp": "2024-01-01"},
                        {"action": "logout", "timestamp": "2024-01-02"},
                    ],
                },
            },
            "metadata": {"version": "1.0", "flags": [True, False, True]},
        }

        tool_result = ToolReturnPart(
            tool_name="test_tool", content=complex_data, tool_call_id="123"
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Verify it's valid YAML that preserves structure
        parsed = yaml.safe_load(result.content)
        assert parsed == complex_data
        assert parsed["user"]["settings"]["preferences"]["theme"] == "dark"
        assert len(parsed["user"]["settings"]["history"]) == 2

    def test_yaml_mode_preserves_special_yaml_types(self):
        """Test that YAML mode preserves special types like None, booleans, numbers."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content={
                "null_value": None,
                "bool_true": True,
                "bool_false": False,
                "int_value": 123,
                "float_value": 45.67,
                "empty_list": [],
                "empty_dict": {},
            },
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        parsed = yaml.safe_load(result.content)
        assert parsed["null_value"] is None
        assert parsed["bool_true"] is True
        assert parsed["bool_false"] is False
        assert parsed["int_value"] == 123
        assert parsed["float_value"] == 45.67
        assert parsed["empty_list"] == []
        assert parsed["empty_dict"] == {}

    def test_yaml_mode_multiline_strings_in_dict(self):
        """Test YAML mode handles multiline strings within dictionaries correctly."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content={
                "description": "Line 1\nLine 2\nLine 3",
                "code": "def foo():\n    return 42",
            },
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Verify multiline strings are preserved in YAML
        parsed = yaml.safe_load(result.content)
        assert parsed["description"] == "Line 1\nLine 2\nLine 3"
        assert parsed["code"] == "def foo():\n    return 42"

    def test_yaml_mode_unicode_content(self):
        """Test YAML mode handles unicode characters correctly."""
        tool_result = ToolReturnPart(
            tool_name="test_tool",
            content={"message": "Hello 世界 🌍", "emoji": "🚀✨", "special": "café"},
            tool_call_id="123",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Unicode should be preserved
        parsed = yaml.safe_load(result.content)
        assert parsed["message"] == "Hello 世界 🌍"
        assert parsed["emoji"] == "🚀✨"
        assert parsed["special"] == "café"

    def test_yaml_mode_empty_string_content(self):
        """Test YAML mode with empty string content."""
        tool_result = ToolReturnPart(tool_name="test_tool", content="", tool_call_id="123")

        result = serialize_tool_return_part(tool_result, "yaml")

        # Empty string should be preserved
        assert result.content == ""

    def test_yaml_mode_numeric_string_content(self):
        """Test YAML mode with string that looks numeric."""
        tool_result = ToolReturnPart(tool_name="test_tool", content="12345", tool_call_id="123")

        result = serialize_tool_return_part(tool_result, "yaml")

        # Should remain as string
        assert result.content == "12345"
        assert isinstance(result.content, str)

    def test_yaml_mode_preserves_tool_metadata(self):
        """Test that YAML mode preserves all ToolReturnPart metadata."""
        tool_result = ToolReturnPart(
            tool_name="my_special_tool",
            content={"data": "test"},
            tool_call_id="unique-id-456",
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Metadata should be preserved
        assert result.tool_name == "my_special_tool"
        assert result.tool_call_id == "unique-id-456"
        assert isinstance(result, ToolReturnPart)

    def test_yaml_mode_returns_new_instance(self):
        """Test that YAML mode returns a new ToolReturnPart instance, not the original."""
        tool_result = ToolReturnPart(
            tool_name="test_tool", content={"key": "value"}, tool_call_id="123"
        )

        result = serialize_tool_return_part(tool_result, "yaml")

        # Should be a different instance (due to replace())
        assert result is not tool_result
        # But content should be different (converted to YAML)
        assert result.content != tool_result.content
        assert isinstance(result.content, str)

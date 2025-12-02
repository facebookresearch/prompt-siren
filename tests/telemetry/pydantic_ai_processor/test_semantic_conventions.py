# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for semantic conventions conversion from OTEL to OpenInference."""

import json

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
)
from prompt_siren.telemetry.pydantic_ai_processor.semantic_conventions import (
    _extract_common_attributes,
    _map_operation_to_span_kind,
    get_attributes,
)


def test_mapping_operation_to_span_kind():
    """Test mapping of GenAI operations to OpenInference span kinds."""
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.CHAT.value)
        == OpenInferenceSpanKindValues.LLM
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.TEXT_COMPLETION.value)
        == OpenInferenceSpanKindValues.LLM
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.GENERATE_CONTENT.value)
        == OpenInferenceSpanKindValues.LLM
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.EMBEDDINGS.value)
        == OpenInferenceSpanKindValues.EMBEDDING
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.EXECUTE_TOOL.value)
        == OpenInferenceSpanKindValues.TOOL
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.CREATE_AGENT.value)
        == OpenInferenceSpanKindValues.AGENT
    )
    assert (
        _map_operation_to_span_kind(GenAiOperationNameValues.INVOKE_AGENT.value)
        == OpenInferenceSpanKindValues.AGENT
    )
    assert _map_operation_to_span_kind("unknown_operation") == OpenInferenceSpanKindValues.UNKNOWN


def test_extract_common_attributes():
    """Test extraction of common attributes."""
    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_REQUEST_MODEL: "gpt-4",
        GEN_AI_USAGE_INPUT_TOKENS: 10,
        GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }

    result = dict(_extract_common_attributes(attrs))

    assert result[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert result[SpanAttributes.LLM_SYSTEM] == "openai"
    assert result[SpanAttributes.LLM_MODEL_NAME] == "gpt-4"
    assert result[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    assert result[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert result[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30


def test_simple_text_message_extraction():
    """Test extraction of simple text message attributes."""
    # Create input messages in GenAI format
    input_messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Hello, world!"}],
        }
    ]

    # Create output messages in GenAI format
    output_messages = [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hi there!"}],
        }
    ]

    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }

    result = dict(get_attributes(attrs))

    # Check input message extraction
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Hello, world!"
    )
    assert result[SpanAttributes.INPUT_VALUE] == "Hello, world!"

    # Check output message extraction
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Hi there!"
    )


def test_tool_call_in_output():
    """Test extraction of tool calls in output messages."""
    # Create input messages in GenAI format
    input_messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "What's the weather?"}],
        }
    ]

    # Create output messages with tool call in GenAI format
    output_messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "tool_123",
                    "name": "get_weather",
                    "arguments": {"location": "New York"},
                }
            ],
        }
    ]

    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }

    result = dict(get_attributes(attrs))

    # Check tool call extraction
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        result[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        ]
        == "tool_123"
    )
    assert (
        result[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        ]
        == "get_weather"
    )
    # Compare JSON content, not formatting
    args_json = result[
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    ]
    assert json.loads(args_json) == {"location": "New York"}


def test_tool_call_in_input():
    """Test handling of tool calls in input messages (conversation history)."""
    # Create input messages with conversation history in GenAI format
    input_messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "What's the weather?"}],
        },
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "tool_123",
                    "name": "get_weather",
                    "arguments": {"location": "New York"},
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "tool_123",
                    "name": "get_weather",
                    "result": "Sunny, 75°F",
                }
            ],
        },
    ]

    # Create output messages in GenAI format
    output_messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "It's sunny in New York with a temperature of 75°F.",
                }
            ],
        }
    ]

    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }

    result = dict(get_attributes(attrs))

    # Check input message extraction with tool calls
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "What's the weather?"
    )

    # Check assistant message with tool call
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        result[
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        ]
        == "tool_123"
    )
    assert (
        result[
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        ]
        == "get_weather"
    )
    # Compare JSON content, not formatting
    args_json = result[
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    ]
    assert json.loads(args_json) == {"location": "New York"}

    # Check tool call response (should have role="tool" for OpenInference)
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_ROLE}"] == "tool"
    )
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Sunny, 75°F"
    )
    assert (
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_NAME}"]
        == "get_weather"
    )

    # Check final output message
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "It's sunny in New York with a temperature of 75°F."
    )


def test_tool_attributes():
    """Test extraction of tool attributes."""
    attrs = {
        "gen_ai.tool.name": "weather_tool",
        "gen_ai.tool.call.id": "tool_123",
    }

    result = dict(get_attributes(attrs))

    assert result[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value
    assert result[SpanAttributes.TOOL_NAME] == "weather_tool"
    assert result[ToolCallAttributes.TOOL_CALL_ID] == "tool_123"


def test_agent_attributes():
    """Test extraction of agent attributes."""
    # Create a message in JSON format
    all_messages = [{"role": "user", "parts": [{"type": "text", "content": "Hello agent!"}]}]

    attrs = {
        "agent_name": "test_agent",
        "pydantic_ai.all_messages": json.dumps(all_messages),
        "final_result": "Hello from agent!",
    }

    result = dict(get_attributes(attrs))

    assert result[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.AGENT.value
    assert result[SpanAttributes.OUTPUT_VALUE] == "Hello from agent!"
    assert result[SpanAttributes.INPUT_VALUE] == "Hello agent!"


def test_message_with_text_and_tool_call():
    """Test extraction of messages that have both text content and tool calls."""
    # Create output messages with both text and tool call (like the user's example)
    output_messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "I'll check your appointments for May 15th, 2024. Let me retrieve that information for you.",
                },
                {
                    "type": "tool_call",
                    "id": "toolu_bdrk_014L89m3FZsRjNePGyukUipD",
                    "name": "get_day_calendar_events",
                    "arguments": {"day": "2024-05-15"},
                },
            ],
            "finish_reason": "tool_call",
        }
    ]

    input_messages = [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "How many appointments do I have on May 15th, 2024?",
                }
            ],
        }
    ]

    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }

    result = dict(get_attributes(attrs))

    # Check that BOTH the text content AND the tool call are extracted
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )

    # Check text content is present
    assert (
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "I'll check your appointments for May 15th, 2024. Let me retrieve that information for you."
    )

    # Check tool call is also present
    assert (
        result[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        ]
        == "toolu_bdrk_014L89m3FZsRjNePGyukUipD"
    )
    assert (
        result[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        ]
        == "get_day_calendar_events"
    )

    # Check tool call arguments
    args_json = result[
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    ]
    assert json.loads(args_json) == {"day": "2024-05-15"}


def test_tool_span_attributes():
    """Test extraction of tool span attributes (tool_arguments and tool_response)."""
    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        "gen_ai.tool.name": "weather",
        "gen_ai.tool.call.id": "call_123",
        "tool_arguments": json.dumps({"city": "Chicago, IL"}),
        "tool_response": "Sunny",
    }

    result = dict(get_attributes(attrs))

    # Check span kind
    assert result[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value

    # Check tool attributes
    assert result[SpanAttributes.TOOL_NAME] == "weather"
    assert result[ToolCallAttributes.TOOL_CALL_ID] == "call_123"

    # Check input/output values
    assert result[SpanAttributes.INPUT_VALUE] == json.dumps({"city": "Chicago, IL"})
    assert result[SpanAttributes.OUTPUT_VALUE] == "Sunny"


def test_tool_span_with_complex_response():
    """Test extraction of tool span attributes with complex JSON response."""
    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        "gen_ai.tool.name": "get_user_info",
        "gen_ai.tool.call.id": "call_456",
        "tool_arguments": json.dumps({"user_id": 123}),
        "tool_response": json.dumps(
            {"name": "John Doe", "email": "john@example.com", "active": True}
        ),
    }

    result = dict(get_attributes(attrs))

    # Check span kind
    assert result[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value

    # Check tool attributes
    assert result[SpanAttributes.TOOL_NAME] == "get_user_info"
    assert result[ToolCallAttributes.TOOL_CALL_ID] == "call_456"

    # Check input/output values
    assert result[SpanAttributes.INPUT_VALUE] == json.dumps({"user_id": 123})
    assert result[SpanAttributes.OUTPUT_VALUE] == json.dumps(
        {"name": "John Doe", "email": "john@example.com", "active": True}
    )


def test_tool_extraction_from_model_request_parameters():
    """Test extraction of tool definitions from model_request_parameters."""
    # Create model_request_parameters with function_tools (pydantic-ai format)
    model_request_params = {
        "function_tools": [
            {
                "name": "weather",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "description": "Get weather for a city",
            },
            {
                "name": "calculator",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                    },
                    "required": ["a", "b", "operation"],
                },
                "description": "Perform basic arithmetic operations",
            },
        ]
    }

    attrs = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        "model_request_parameters": json.dumps(model_request_params),
    }

    result = dict(get_attributes(attrs))

    # Check that tools are extracted
    assert f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}" in result
    assert result[f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}"] == "weather"
    assert (
        result[f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}"]
        == "Get weather for a city"
    )

    tool_schema_0 = result[f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema"]
    assert json.loads(tool_schema_0) == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    assert result[f"{SpanAttributes.LLM_TOOLS}.1.{SpanAttributes.TOOL_NAME}"] == "calculator"
    assert (
        result[f"{SpanAttributes.LLM_TOOLS}.1.{SpanAttributes.TOOL_DESCRIPTION}"]
        == "Perform basic arithmetic operations"
    )

    tool_schema_1 = result[f"{SpanAttributes.LLM_TOOLS}.1.tool.json_schema"]
    assert json.loads(tool_schema_1) == {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
        },
        "required": ["a", "b", "operation"],
    }

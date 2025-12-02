# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the formatted_span module and attribute flattening functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from prompt_siren.telemetry.formatted_span import (
    flatten_attribute,
    flatten_attributes,
    formatted_span,
    is_primitive,
)


def test_is_primitive():
    """Test the is_primitive type guard function."""
    # Primitives should return True
    assert is_primitive("string") is True
    assert is_primitive(True) is True
    assert is_primitive(123) is True
    assert is_primitive(123.456) is True

    # Non-primitives should return False
    assert is_primitive(None) is False
    assert is_primitive([1, 2, 3]) is False
    assert is_primitive({"a": 1}) is False
    assert is_primitive({1, 2, 3}) is False


class TestFlattenAttribute:
    """Tests for the flatten_attribute function."""

    def test_none_values(self):
        """Test flattening None values (should yield nothing)."""
        result = list(flatten_attribute("test", None))
        assert result == []

    def test_primitive_values(self):
        """Test flattening primitive values."""
        # String
        assert list(flatten_attribute("test", "value")) == [("test", "value")]
        # Integer
        assert list(flatten_attribute("test", 123)) == [("test", 123)]
        # Float
        assert list(flatten_attribute("test", 123.456)) == [("test", 123.456)]
        # Boolean
        assert list(flatten_attribute("test", True)) == [("test", True)]

    def test_dict_values(self):
        """Test flattening dictionary values."""
        input_dict = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": 4}}}
        result = list(flatten_attribute("test", input_dict))

        expected = [
            ("test.a", 1),
            ("test.b.c", 2),
            ("test.b.d", 3),
            ("test.e.f.g", 4),
        ]

        # Sort both lists to ensure order doesn't matter for comparison
        assert sorted(result) == sorted(expected)

    def test_list_values(self):
        """Test flattening list values."""
        input_list = [1, "two", True]
        result = list(flatten_attribute("test", input_list))

        expected = [
            ("test[0]", 1),
            ("test[1]", "two"),
            ("test[2]", True),
        ]

        assert result == expected

    def test_nested_collections(self):
        """Test flattening complex nested collections."""
        input_value = {
            "simple": "value",
            "list": [1, 2, {"nested": "value"}],
            "dict": {"a": 1, "b": [3, 4, 5], "c": {"d": {"e": 6}}},
        }

        result = list(flatten_attribute("root", input_value))

        expected = [
            ("root.simple", "value"),
            ("root.list[0]", 1),
            ("root.list[1]", 2),
            ("root.list[2].nested", "value"),
            ("root.dict.a", 1),
            ("root.dict.b[0]", 3),
            ("root.dict.b[1]", 4),
            ("root.dict.b[2]", 5),
            ("root.dict.c.d.e", 6),
        ]

        # Sort both lists to ensure order doesn't matter for comparison
        assert sorted(result) == sorted(expected)

    def test_custom_object_to_string(self):
        """Test that custom objects are converted to string."""

        class CustomClass:
            def __str__(self):
                return "custom_string_representation"

        result = list(flatten_attribute("test", CustomClass()))
        assert result == [("test", "custom_string_representation")]

        # Test object that raises in __str__ falls back to repr
        class BadStrClass:
            def __str__(self):
                raise ValueError("Cannot stringify")

            def __repr__(self):
                return "repr_fallback"

        result = list(flatten_attribute("test", BadStrClass()))
        assert result == [("test", "repr_fallback")]


def test_flatten_attributes():
    """Test the flatten_attributes wrapper function."""
    # Test with a dictionary
    input_dict = {"a": 1, "b": {"c": 2}}
    result = flatten_attributes("test", input_dict)

    expected = {
        "test.a": 1,
        "test.b.c": 2,
    }

    assert result == expected

    # Test with a list
    input_list = [1, 2, 3]
    result = flatten_attributes("items", input_list)

    expected = {
        "items[0]": 1,
        "items[1]": 2,
        "items[2]": 3,
    }

    assert result == expected


class TestFormattedSpan:
    """Tests for the formatted_span function."""

    @patch("logfire.span")
    def test_formatting_with_simple_kwargs(self, mock_span):
        """Test that name formatting works with simple values."""
        mock_span_instance = MagicMock()
        mock_span.return_value = mock_span_instance

        with formatted_span("task {task_id}", task_id=123):
            pass

        # Verify span was created with formatted name and correct attributes
        mock_span.assert_called_once_with("task 123", task_id=123)

    @patch("logfire.span")
    def test_formatting_with_nested_kwargs(self, mock_span):
        """Test that nested attributes are flattened."""
        mock_span_instance = MagicMock()
        mock_span.return_value = mock_span_instance

        with formatted_span(
            "run {run_id}",
            run_id="abc123",
            config={"model": "gpt-4", "params": {"temp": 0.7}},
        ):
            pass

        # Verify span was created with flattened attributes
        mock_span.assert_called_once()
        args, kwargs = mock_span.call_args
        assert args[0] == "run abc123"
        assert "run_id" in kwargs
        assert kwargs["run_id"] == "abc123"
        assert "config.model" in kwargs
        assert kwargs["config.model"] == "gpt-4"
        assert "config.params.temp" in kwargs
        assert kwargs["config.params.temp"] == 0.7

    @patch("logfire.span")
    def test_special_keys_not_flattened(self, mock_span):
        """Test that special keys like 'kind' aren't flattened."""
        mock_span_instance = MagicMock()
        mock_span.return_value = mock_span_instance

        with formatted_span(
            "test span",
            kind={
                "type": "test",
                "level": "debug",
            },  # This would normally be flattened
            data={"a": 1, "b": 2},  # This should be flattened
        ):
            pass

        # Verify that 'kind' was preserved but 'data' was flattened
        mock_span.assert_called_once()
        _, kwargs = mock_span.call_args
        assert "kind" in kwargs
        assert kwargs["kind"] == {"type": "test", "level": "debug"}
        assert "data.a" in kwargs
        assert kwargs["data.a"] == 1
        assert "data.b" in kwargs
        assert kwargs["data.b"] == 2

    @patch("logfire.span")
    def test_formatting_failure(self, mock_span):
        """Test that formatting failures are handled gracefully."""
        mock_span_instance = MagicMock()
        mock_span.return_value = mock_span_instance

        # Missing key should use template as fallback
        with formatted_span("task {missing_key}", task_id=123):
            pass

        mock_span.assert_called_once()
        args, kwargs = mock_span.call_args
        assert args[0] == "task {missing_key}"
        assert "task_id" in kwargs
        assert kwargs["task_id"] == 123

    @patch("logfire.span")
    def test_complex_nested_structure(self, mock_span):
        """Test with a complex nested structure to verify deep flattening."""
        mock_span_instance = MagicMock()
        mock_span.return_value = mock_span_instance

        complex_data = {
            "user": {"id": 123, "name": "Test User"},
            "items": [
                {"id": 1, "tags": ["red", "blue"]},
                {"id": 2, "tags": ["green"]},
            ],
            "metadata": {
                "source": "api",
                "timestamps": {
                    "created": "2023-01-01",
                    "updated": "2023-01-02",
                },
            },
        }

        with formatted_span("test span", data=complex_data):
            pass

        # Verify all nested structures were flattened correctly
        mock_span.assert_called_once()
        _, kwargs = mock_span.call_args

        assert "data.user.id" in kwargs
        assert kwargs["data.user.id"] == 123
        assert "data.user.name" in kwargs
        assert kwargs["data.user.name"] == "Test User"

        assert "data.items[0].id" in kwargs
        assert kwargs["data.items[0].id"] == 1
        assert "data.items[0].tags[0]" in kwargs
        assert kwargs["data.items[0].tags[0]"] == "red"
        assert "data.items[0].tags[1]" in kwargs
        assert kwargs["data.items[0].tags[1]"] == "blue"

        assert "data.items[1].id" in kwargs
        assert kwargs["data.items[1].id"] == 2
        assert "data.items[1].tags[0]" in kwargs
        assert kwargs["data.items[1].tags[0]"] == "green"

        assert "data.metadata.source" in kwargs
        assert kwargs["data.metadata.source"] == "api"
        assert "data.metadata.timestamps.created" in kwargs
        assert kwargs["data.metadata.timestamps.created"] == "2023-01-01"
        assert "data.metadata.timestamps.updated" in kwargs
        assert kwargs["data.metadata.timestamps.updated"] == "2023-01-02"

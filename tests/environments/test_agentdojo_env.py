# Copyright (c) Meta Platforms, Inc. and affiliates.
import pytest
from agentdojo.default_suites.v1.tools.types import Email
from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)
from prompt_siren.attacks.attack_utils import _make_fake_context
from prompt_siren.datasets.agentdojo_dataset import (
    AgentDojoDatasetConfig,
    load_agentdojo_dataset,
    make_agentdojo_toolsets,
)
from prompt_siren.environments.agentdojo_env import (
    _find_vector_ids,
    _substitute_placeholders,
    make_agentdojo_env,
)
from pydantic import BaseModel


class ExampleModel(BaseModel):
    """Example Pydantic model for testing."""

    name: str
    value: int
    description: str


pytestmark = pytest.mark.anyio


class TestSubstitutePlaceholders:
    """Tests for the substitute_placeholders function."""

    def test_substitute_placeholders_string(self):
        """Test substituting placeholders in a simple string."""
        obj = "Hello {name}, your score is {score}!"
        values = {"name": "Alice", "score": "95"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        assert result == "Hello Alice, your score is 95!"
        assert isinstance(result, str)

    def test_substitute_placeholders_string_no_placeholders(self):
        """Test string without placeholders remains unchanged."""
        obj = "This is just a regular string"
        values = {"unused": "value"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        assert result == "This is just a regular string"

    def test_substitute_placeholders_string_empty_values(self):
        """Test string with placeholders but empty values dict leaves placeholders unchanged."""
        obj = "Hello {name}!"
        values = {}
        defaults = {}

        # With the new implementation, unknown placeholders are left as-is
        result = _substitute_placeholders(obj, values, defaults)
        assert result == "Hello {name}!"

    def test_substitute_placeholders_dict(self):
        """Test substituting placeholders in a dictionary."""
        obj = {
            "greeting": "Hello {name}!",
            "score": "Your score is {points}",
            "constant": "No placeholders here",
        }
        values = {"name": "Bob", "points": "87"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        expected = {
            "greeting": "Hello Bob!",
            "score": "Your score is 87",
            "constant": "No placeholders here",
        }
        assert result == expected
        assert isinstance(result, dict)
        assert result is not obj  # Should be a new object

    def test_substitute_placeholders_nested_dict(self):
        """Test substituting placeholders in nested dictionaries."""
        obj = {
            "user": {
                "name": "Welcome {username}",
                "settings": {"theme": "{theme_name} theme", "language": "en"},
            },
            "status": "User {username} logged in",
        }
        values = {"username": "charlie", "theme_name": "dark"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        expected = {
            "user": {
                "name": "Welcome charlie",
                "settings": {"theme": "dark theme", "language": "en"},
            },
            "status": "User charlie logged in",
        }
        assert result == expected

    def test_substitute_placeholders_list(self):
        """Test substituting placeholders in a list."""
        obj = ["Hello {name}", "Score: {score}", "Regular string", 42]
        values = {"name": "Diana", "score": "100"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        expected = ["Hello Diana", "Score: 100", "Regular string", 42]
        assert result == expected
        assert isinstance(result, list)
        assert result is not obj

    def test_substitute_placeholders_tuple(self):
        """Test substituting placeholders in a tuple."""
        obj = ("Hello {name}", "Value: {value}", 123)
        values = {"name": "Eve", "value": "test"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        expected = ("Hello Eve", "Value: test", 123)
        assert result == expected
        assert isinstance(result, tuple)
        assert result is not obj

    def test_substitute_placeholders_nested_structures(self):
        """Test substituting placeholders in complex nested structures."""
        obj = {
            "users": [
                {"name": "User {id}", "active": True},
                {"name": "Admin {admin_id}", "active": False},
            ],
            "settings": ("Theme: {theme}", ["Color: {color}", "Size: {size}"]),
        }
        values = {
            "id": "001",
            "admin_id": "admin",
            "theme": "dark",
            "color": "blue",
            "size": "large",
        }
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        expected = {
            "users": [
                {"name": "User 001", "active": True},
                {"name": "Admin admin", "active": False},
            ],
            "settings": ("Theme: dark", ["Color: blue", "Size: large"]),
        }
        assert result == expected

    def test_substitute_placeholders_pydantic_model(self):
        """Test substituting placeholders in a Pydantic BaseModel."""
        obj = ExampleModel(
            name="User {user_id}",
            value=42,
            description="Description for {user_id}",
        )
        values = {"user_id": "123"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        assert isinstance(result, ExampleModel)
        assert result.name == "User 123"
        assert result.value == 42
        assert result.description == "Description for 123"
        assert result is not obj  # Should be a new object

    def test_substitute_placeholders_primitive_types(self):
        """Test that primitive types are returned unchanged."""
        values = {"key": "value"}
        defaults = {}

        # Test int
        assert _substitute_placeholders(42, values, defaults) == 42
        assert _substitute_placeholders(-10, values, defaults) == -10

        # Test float
        assert _substitute_placeholders(3.14, values, defaults) == 3.14
        assert _substitute_placeholders(-2.5, values, defaults) == -2.5

        # Test bool
        assert _substitute_placeholders(True, values, defaults) is True
        assert _substitute_placeholders(False, values, defaults) is False

        # Test None
        assert _substitute_placeholders(None, values, defaults) is None

    def test_substitute_placeholders_invalid_type(self):
        """Test that invalid types raise RuntimeError."""
        values = {"key": "value"}
        defaults = {}

        # Custom object that's not handled
        class CustomObject:
            pass

        with pytest.raises(RuntimeError, match="Invalid return type"):
            _substitute_placeholders(CustomObject(), values, defaults)  # type: ignore -- this is on purpose

    def test_substitute_placeholders_immutability(self):
        """Test that original objects are not modified."""
        original_dict = {"message": "Hello {name}"}
        original_list = ["Hello {name}", "Goodbye {name}"]
        values = {"name": "Test"}
        defaults = {}

        # Test dict immutability
        result_dict = _substitute_placeholders(original_dict, values, defaults)
        assert original_dict == {"message": "Hello {name}"}  # Unchanged
        assert result_dict == {"message": "Hello Test"}

        # Test list immutability
        result_list = _substitute_placeholders(original_list, values, defaults)
        assert original_list == ["Hello {name}", "Goodbye {name}"]  # Unchanged
        assert result_list == ["Hello Test", "Goodbye Test"]

    def test_substitute_placeholders_with_defaults_needed(self):
        """Test that default values are used when values are missing."""
        obj = "Hello {name}, your score is {score} and your rank is {rank}!"
        values = {"name": "Alice"}  # Only name is provided
        defaults = {
            "score": "50",
            "rank": "bronze",
            "unused": "value",
        }  # Default values

        result = _substitute_placeholders(obj, values, defaults)

        # name comes from values, score and rank come from defaults
        assert result == "Hello Alice, your score is 50 and your rank is bronze!"

    def test_substitute_placeholders_with_defaults_overridden(self):
        """Test that values override defaults when both are provided."""
        obj = "Hello {name}, your score is {score}!"
        values = {"name": "Alice", "score": "95"}  # Both values provided
        defaults = {
            "name": "Default",
            "score": "50",
            "unused": "value",
        }  # Defaults that will be overridden

        result = _substitute_placeholders(obj, values, defaults)

        # Values should override defaults
        assert result == "Hello Alice, your score is 95!"

    def test_substitute_placeholders_dict_with_defaults(self):
        """Test defaults work with nested structures."""
        obj = {
            "greeting": "Hello {name}!",
            "score": "Your score is {points}",
            "level": "Current level: {level}",
        }
        values = {"name": "Bob"}  # Only name provided
        defaults = {
            "points": "0",
            "level": "beginner",
            "name": "Anonymous",
        }  # Defaults

        result = _substitute_placeholders(obj, values, defaults)

        expected = {
            "greeting": "Hello Bob!",  # Uses value
            "score": "Your score is 0",  # Uses default
            "level": "Current level: beginner",  # Uses default
        }
        assert result == expected

    def test_substitute_placeholders_with_literal_braces(self):
        """Test that literal braces (e.g., JSON) don't cause errors."""
        # This tests the fix for ValueError: unexpected '{' in field name
        obj = 'Response: {"key": "value", "nested": {"inner": "data"}}'
        values = {}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        # Should return unchanged since there are no actual placeholders
        assert result == 'Response: {"key": "value", "nested": {"inner": "data"}}'

    def test_substitute_placeholders_with_json_and_placeholders(self):
        """Test that JSON with actual placeholders works correctly."""
        obj = 'User {name} sent: {"message": "Hello", "status": "{status}"}'
        values = {"name": "Alice", "status": "active"}
        defaults = {}

        result = _substitute_placeholders(obj, values, defaults)

        # Placeholders should be replaced, but literal JSON braces left alone
        assert result == 'User Alice sent: {"message": "Hello", "status": "active"}'


class TestFindVectorIds:
    """Tests for the find_vector_ids function."""

    def test_find_vector_ids_string_with_vectors(self):
        """Test finding vector IDs in a string containing them."""
        obj = "This contains {vector1} and also {vector2} and {vector3}, but not {\\vector4}"
        vector_ids = ["vector1", "vector2", "vector3", "vector4"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2", "vector3"}
        assert isinstance(result, list)

    def test_find_vector_ids_string_no_vectors(self):
        """Test finding vector IDs in a string containing none of them."""
        obj = "This is just a regular string without any vectors"
        vector_ids = ["vector1", "vector2", "vector3"]

        result = _find_vector_ids(obj, vector_ids)

        assert result == []

    def test_find_vector_ids_string_empty_vector_list(self):
        """Test finding vector IDs with empty vector list."""
        obj = "This contains {vector1} and {vector2}"
        vector_ids = []

        result = _find_vector_ids(obj, vector_ids)

        assert result == []

    def test_find_vector_ids_string_partial_matches(self):
        """Test that partial matches are still found (substring matching)."""
        obj = "This contains {super_vector1} and {vector2_suffix}"
        vector_ids = ["vector1", "vector2"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == set()

    def test_find_vector_ids_dict(self):
        """Test finding vector IDs in a dictionary."""
        obj = {
            "message1": "Contains {vector1}",
            "message2": "Contains {vector2} and {vector3}",
            "message3": "No vectors here",
            "number": 42,
        }
        vector_ids = ["vector1", "vector2", "vector3", "vector4"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2", "vector3"}

    def test_find_vector_ids_nested_dict(self):
        """Test finding vector IDs in nested dictionaries."""
        obj = {
            "level1": {
                "message": "Contains {vector1}",
                "level2": {"deep_message": "Contains {vector2}", "number": 123},
            },
            "other": "Contains {vector3}",
        }
        vector_ids = ["vector1", "vector2", "vector3", "vector4"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2", "vector3"}

    def test_find_vector_ids_list(self):
        """Test finding vector IDs in a list."""
        obj = [
            "First message with {vector1}",
            "Second message with {vector2}",
            "Third message without vectors",
            42,
            True,
        ]
        vector_ids = ["vector1", "vector2", "vector3"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2"}

    def test_find_vector_ids_tuple(self):
        """Test finding vector IDs in a tuple."""
        obj = (
            "Message with {vector1}",
            "Another message with {vector2}",
            123,
            "{vector3} is here too",
        )
        vector_ids = ["vector1", "vector2", "vector3", "vector4"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2", "vector3"}

    def test_find_vector_ids_complex_nested_structure(self):
        """Test finding vector IDs in complex nested structures."""
        obj = {
            "users": [
                {"name": "Contains {vector1}", "active": True},
                {"name": "Contains {vector2}", "active": False},
            ],
            "settings": (
                "Theme contains {vector3}",
                ["Color {vector4}", "Size normal"],
                {"nested": "Deep {vector1} again"},
            ),
            "count": 42,
        }
        vector_ids = ["vector1", "vector2", "vector3", "vector4", "vector5"]

        result = _find_vector_ids(obj, vector_ids)

        # vector1 appears twice, but should only be in result once per find
        # Note: the function returns a list, so duplicates are possible
        result_set = set(result)
        assert result_set == {"vector1", "vector2", "vector3", "vector4"}

    def test_find_vector_ids_pydantic_model(self):
        """Test finding vector IDs in a Pydantic BaseModel."""
        obj = ExampleModel(
            name="User with {vector1}",
            value=42,
            description="Description contains {vector2} and {vector3}",
        )
        vector_ids = ["vector1", "vector2", "vector3", "vector4"]

        result = _find_vector_ids(obj, vector_ids)

        assert set(result) == {"vector1", "vector2", "vector3"}

    def test_find_vector_ids_primitive_types(self):
        """Test that primitive types return empty list."""
        vector_ids = ["vector1", "vector2"]

        # Test int
        assert _find_vector_ids(42, vector_ids) == []
        assert _find_vector_ids(-10, vector_ids) == []

        # Test float
        assert _find_vector_ids(3.14, vector_ids) == []
        assert _find_vector_ids(-2.5, vector_ids) == []

        # Test bool
        assert _find_vector_ids(True, vector_ids) == []
        assert _find_vector_ids(False, vector_ids) == []

        # Test None
        assert _find_vector_ids(None, vector_ids) == []

    def test_find_vector_ids_invalid_type(self):
        """Test that invalid types raise RuntimeError."""
        vector_ids = ["vector1", "vector2"]

        # Custom object that's not handled
        class CustomObject:
            pass

        with pytest.raises(RuntimeError, match="Invalid return type"):
            _find_vector_ids(CustomObject(), vector_ids)  # type: ignore -- this is on purpose

    def test_find_vector_ids_duplicate_results(self):
        """Test that the same vector ID can appear multiple times in results."""
        obj = {
            "message1": "Contains {vector1}",
            "message2": "Also contains {vector1}",
            "list": ["{vector1} again", "and {vector2}"],
        }
        vector_ids = ["vector1", "vector2"]

        assert set(_find_vector_ids(obj, vector_ids)) == set(vector_ids)

    def test_find_vector_ids_empty_structures(self):
        """Test finding vector IDs in empty structures."""
        vector_ids = ["vector1", "vector2"]

        # Empty dict
        assert _find_vector_ids({}, vector_ids) == []

        # Empty list
        assert _find_vector_ids([], vector_ids) == []

        # Empty tuple
        assert _find_vector_ids((), vector_ids) == []

        # Empty string
        assert _find_vector_ids("", vector_ids) == []

    def test_find_vector_ids_case_sensitivity(self):
        """Test that vector ID matching is case sensitive."""
        obj = "Contains {Vector1} and {VECTOR2} and {vector1}"
        vector_ids = ["vector1", "vector2", "Vector1", "VECTOR2"]

        result = _find_vector_ids(obj, vector_ids)

        # Should find exact case matches
        assert set(result) == {"vector1", "Vector1", "VECTOR2"}
        assert "vector2" not in result  # lowercase vector2 not in the string

    def test_find_vector_ids_iterable_types(self):
        """Test that function works with different iterable types for vector_ids."""
        obj = "Contains {vector1} and {vector2}"

        # Test with list
        result_list = _find_vector_ids(obj, ["vector1", "vector2", "vector3"])
        assert set(result_list) == {"vector1", "vector2"}

        # Test with tuple
        result_tuple = _find_vector_ids(obj, ("vector1", "vector2", "vector3"))
        assert set(result_tuple) == {"vector1", "vector2"}

        # Test with set
        result_set = _find_vector_ids(obj, {"vector1", "vector2", "vector3"})
        assert set(result_set) == {"vector1", "vector2"}


class TestAgentDojoEnv:
    async def test_run_tool_toolset(self):
        # Load dataset to get task couples
        dataset_config = AgentDojoDatasetConfig(suite_name="workspace", version="v1.2.2")
        dataset = load_agentdojo_dataset(dataset_config)

        config = AgentDojoDatasetConfig(suite_name="workspace", version="v1.2.2")
        toolsets = make_agentdojo_toolsets(config)
        toolset = toolsets[0]
        env = make_agentdojo_env(suite_name=config.suite_name, version=config.version)
        async with env.create_task_context(dataset.task_couples[0]) as env_state:
            assert isinstance(env_state, WorkspaceEnvironment)
            ctx = _make_fake_context(env_state)
            tools = await toolset.get_tools(ctx)
            tool = tools.get("send_email")
            assert tool is not None
            args = {
                "recipients": ["example@example.com"],
                "subject": "Hello",
                "body": "Hey, how is it going?",
            }
            new_email = await toolset.call_tool("send_email", args, ctx, tool)

            assert isinstance(new_email, Email)
            assert new_email.recipients == args["recipients"]
            assert new_email.subject == args["subject"]
            assert new_email.body == args["body"]

            # Check that env state was updated
            assert env_state.inbox.sent[-1] == new_email

    async def test_run_tool_fresh_env(self):
        # Load dataset to get task couples
        dataset_config = AgentDojoDatasetConfig(suite_name="workspace", version="v1.2.2")
        dataset = load_agentdojo_dataset(dataset_config)

        config = AgentDojoDatasetConfig(suite_name="workspace", version="v1.2.2")
        toolsets = make_agentdojo_toolsets(config)
        toolset = toolsets[0]
        env = make_agentdojo_env(suite_name=config.suite_name, version=config.version)
        async with env.create_task_context(dataset.task_couples[0]) as env_state:
            original_sent_len = len(env_state.inbox.sent)
            assert isinstance(env_state, WorkspaceEnvironment)
            ctx = _make_fake_context(env_state)
            tools = await toolset.get_tools(ctx)
            tool = tools.get("send_email")
            assert tool is not None
            args = {
                "recipients": ["example@example.com"],
                "subject": "Hello",
                "body": "Hey, how is it going?",
            }
            await toolset.call_tool("send_email", args, ctx, tool)

        async with env.create_task_context(dataset.task_couples[0]) as env_state:
            assert len(env_state.inbox.sent) == original_sent_len

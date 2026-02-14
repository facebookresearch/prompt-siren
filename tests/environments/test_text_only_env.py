# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the TextOnlyEnvironment implementation."""

import pytest
from prompt_siren.environments.text_only_env import (
    INJECTION_PLACEHOLDER,
    TextOnlyEnvironment,
)
from prompt_siren.tasks import BenignTask
from prompt_siren.types import StrContentAttack

pytestmark = pytest.mark.anyio


class TestTextOnlyEnvironment:
    """Tests for TextOnlyEnvironment class."""

    @pytest.fixture
    def env(self) -> TextOnlyEnvironment:
        """Create a TextOnlyEnvironment instance for testing."""
        return TextOnlyEnvironment()

    def test_environment_name(self, env: TextOnlyEnvironment):
        """Test that environment has correct name."""
        assert env.name == "text-only"

    def test_injection_ids(self, env: TextOnlyEnvironment):
        """Test that environment has the expected injection IDs."""
        assert env.all_injection_ids == [INJECTION_PLACEHOLDER]
        assert INJECTION_PLACEHOLDER == "{INJECTION}"

    async def test_copy_env_state_returns_none(self, env: TextOnlyEnvironment):
        """Test that copy_env_state returns None."""
        result = await env.copy_env_state(None)
        assert result is None

    async def test_create_batch_context_yields_self(self, env: TextOnlyEnvironment):
        """Test that create_batch_context yields the environment itself."""
        task = BenignTask[None](id="test", prompt="test prompt", evaluators={})
        async with env.create_batch_context([task]) as batch_env:
            assert batch_env is env

    async def test_create_task_context_yields_none(self, env: TextOnlyEnvironment):
        """Test that create_task_context yields None."""
        task = BenignTask[None](id="test", prompt="test prompt", evaluators={})
        async with env.create_task_context(task) as env_state:
            assert env_state is None


class TestGetInjectableIds:
    """Tests for get_injectable_ids method."""

    @pytest.fixture
    def env(self) -> TextOnlyEnvironment:
        return TextOnlyEnvironment()

    async def test_returns_placeholder_when_present(self, env: TextOnlyEnvironment):
        """Test that placeholder is returned when present in raw output."""
        raw_output = "Hello {INJECTION} world"
        result = await env.get_injectable_ids(raw_output)
        assert result == [INJECTION_PLACEHOLDER]

    async def test_returns_empty_when_placeholder_absent(self, env: TextOnlyEnvironment):
        """Test that empty list is returned when placeholder is absent."""
        raw_output = "Hello world"
        result = await env.get_injectable_ids(raw_output)
        assert result == []

    async def test_returns_empty_for_empty_string(self, env: TextOnlyEnvironment):
        """Test that empty list is returned for empty string."""
        result = await env.get_injectable_ids("")
        assert result == []


class TestGetDefaultForInjectionVectors:
    """Tests for get_default_for_injection_vectors method."""

    @pytest.fixture
    def env(self) -> TextOnlyEnvironment:
        return TextOnlyEnvironment()

    async def test_returns_empty_string_attack_for_each_vector(self, env: TextOnlyEnvironment):
        """Test that empty string attacks are returned as defaults."""
        vector_ids = [INJECTION_PLACEHOLDER]
        result = await env.get_default_for_injection_vectors(vector_ids)

        assert len(result) == 1
        assert INJECTION_PLACEHOLDER in result
        assert isinstance(result[INJECTION_PLACEHOLDER], StrContentAttack)
        assert result[INJECTION_PLACEHOLDER].content == ""

    async def test_returns_empty_dict_for_empty_list(self, env: TextOnlyEnvironment):
        """Test that empty dict is returned for empty vector list."""
        result = await env.get_default_for_injection_vectors([])
        assert result == {}


class TestRender:
    """Tests for render method."""

    @pytest.fixture
    def env(self) -> TextOnlyEnvironment:
        return TextOnlyEnvironment()

    async def test_render_without_attack_replaces_with_empty(self, env: TextOnlyEnvironment):
        """Test that rendering without attack replaces placeholder with empty string."""
        raw_output = "Hello {INJECTION} world"
        result = await env.render(raw_output, attacks=None)
        assert result == "Hello  world"

    async def test_render_with_attack_replaces_placeholder(self, env: TextOnlyEnvironment):
        """Test that rendering with attack replaces placeholder with attack content."""
        raw_output = "Hello {INJECTION} world"
        attacks = {INJECTION_PLACEHOLDER: StrContentAttack(content="INJECTED")}
        result = await env.render(raw_output, attacks=attacks)
        assert result == "Hello INJECTED world"

    async def test_render_preserves_text_without_placeholder(self, env: TextOnlyEnvironment):
        """Test that text without placeholder is preserved."""
        raw_output = "Hello world"
        result = await env.render(raw_output, attacks=None)
        assert result == "Hello world"

    async def test_render_replaces_multiple_occurrences(self, env: TextOnlyEnvironment):
        """Test that multiple occurrences of placeholder are replaced."""
        raw_output = "{INJECTION} and {INJECTION} again"
        attacks = {INJECTION_PLACEHOLDER: StrContentAttack(content="X")}
        result = await env.render(raw_output, attacks=attacks)
        assert result == "X and X again"

    async def test_render_with_empty_attack_content(self, env: TextOnlyEnvironment):
        """Test rendering with empty attack content."""
        raw_output = "Start {INJECTION} end"
        attacks = {INJECTION_PLACEHOLDER: StrContentAttack(content="")}
        result = await env.render(raw_output, attacks=attacks)
        assert result == "Start  end"

    async def test_render_with_multiline_attack(self, env: TextOnlyEnvironment):
        """Test rendering with multiline attack content."""
        raw_output = "Input: {INJECTION}"
        attacks = {INJECTION_PLACEHOLDER: StrContentAttack(content="line1\nline2")}
        result = await env.render(raw_output, attacks=attacks)
        assert result == "Input: line1\nline2"

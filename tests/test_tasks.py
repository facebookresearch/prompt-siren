# Copyright (c) Meta Platforms, Inc. and affiliates.
import pytest
from prompt_siren.tasks import BenignTask, MaliciousTask
from prompt_siren.types import InjectableUserContent
from pydantic_ai import BinaryContent, ModelMessage, UserContent
from pydantic_ai.messages import ModelRequest


class TestBenignTaskValidation:
    """Test BenignTask validation logic."""

    def test_benign_task_empty_string_raises(self):
        """Test that empty string prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt string cannot be empty"):
            BenignTask(id="test", prompt="", evaluators={})

    def test_benign_task_whitespace_only_string_raises(self):
        """Test that whitespace-only string prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt string cannot be empty"):
            BenignTask(id="test", prompt="   ", evaluators={})

    def test_benign_task_whitespace_with_newlines_raises(self):
        """Test that whitespace with newlines raises ValueError."""
        with pytest.raises(ValueError, match="prompt string cannot be empty"):
            BenignTask(id="test", prompt="\n\t  \n", evaluators={})

    def test_benign_task_empty_list_raises(self):
        """Test that empty list prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt list cannot be empty"):
            BenignTask(id="test", prompt=[], evaluators={})

    def test_benign_task_valid_string(self):
        """Test that valid string prompt works."""
        task = BenignTask(id="test", prompt="Valid prompt", evaluators={})
        assert task.prompt == "Valid prompt"
        assert task.id == "test"

    def test_benign_task_valid_string_with_whitespace(self):
        """Test that valid string with leading/trailing whitespace works."""
        task = BenignTask(id="test", prompt="  Valid prompt  ", evaluators={})
        assert task.prompt == "  Valid prompt  "

    def test_benign_task_valid_list(self):
        """Test that valid list prompt works."""

        prompt: str | list[UserContent | InjectableUserContent] = [
            BinaryContent(data=b"abc", media_type="image/png")
        ]
        task = BenignTask(id="test", prompt=prompt, evaluators={})
        assert task.prompt == prompt

    def test_benign_task_with_message_history(self):
        """Test that BenignTask can have message history."""

        history: list[ModelMessage] = [ModelRequest.user_text_prompt("previous")]
        task = BenignTask(
            id="test",
            prompt="Valid prompt",
            evaluators={},
            message_history=history,
        )
        assert task.message_history == history

    def test_benign_task_without_message_history(self):
        """Test that message_history defaults to None."""
        task = BenignTask(id="test", prompt="Valid prompt", evaluators={})
        assert task.message_history is None


class TestMaliciousTaskValidation:
    """Test MaliciousTask validation logic."""

    def test_malicious_task_empty_string_raises(self):
        """Test that empty string goal raises ValueError."""
        with pytest.raises(ValueError, match="goal string cannot be empty"):
            MaliciousTask(id="test", goal="", evaluators={})

    def test_malicious_task_whitespace_only_string_raises(self):
        """Test that whitespace-only string goal raises ValueError."""
        with pytest.raises(ValueError, match="goal string cannot be empty"):
            MaliciousTask(id="test", goal="   ", evaluators={})

    def test_malicious_task_whitespace_with_newlines_raises(self):
        """Test that whitespace with newlines raises ValueError."""
        with pytest.raises(ValueError, match="goal string cannot be empty"):
            MaliciousTask(id="test", goal="\n\t  \n", evaluators={})

    def test_malicious_task_valid_goal(self):
        """Test that valid goal works."""
        task = MaliciousTask(id="test", goal="Valid goal", evaluators={})
        assert task.goal == "Valid goal"
        assert task.id == "test"

    def test_malicious_task_valid_goal_with_whitespace(self):
        """Test that valid goal with leading/trailing whitespace works."""
        task = MaliciousTask(id="test", goal="  Valid goal  ", evaluators={})
        assert task.goal == "  Valid goal  "

    def test_malicious_task_no_message_history(self):
        """Test that MaliciousTask does not have message_history field."""
        task = MaliciousTask(id="test", goal="Valid goal", evaluators={})
        assert not hasattr(task, "message_history")

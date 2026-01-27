# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for DictAttack implementation."""

import json
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from prompt_siren.agents.states import (
    EndState,
    FinishReason,
    InjectableModelRequestState,
    ModelRequestState,
)
from prompt_siren.attacks.dict_attack import (
    create_dict_attack,
    create_dict_attack_from_file,
    DictAttack,
    DictAttackConfig,
    FileAttackConfig,
)
from prompt_siren.attacks.executor import InjectionCheckpoint
from prompt_siren.tasks import TaskCouple
from prompt_siren.types import (
    AttackFile,
    InjectableModelRequestPart,
    InjectableStrContent,
    InjectableUserPromptPart,
    InjectionAttack,
    InjectionAttacksDict,
    StrContentAttack,
)
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelRequestPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.usage import RunUsage

from ..conftest import (
    MockAgent,
)

pytestmark = pytest.mark.anyio


@dataclass
class _TestableAgent(MockAgent):  # Prefix with _ to avoid pytest collection
    """Enhanced MockAgent for testing DictAttack with proper state machine simulation."""

    # Control the agent's behavior for testing
    should_find_injectable: bool = True
    injectable_vector_ids: list[str] | None = None

    def __post_init__(self):
        if self.injectable_vector_ids is None:
            self.injectable_vector_ids = ["vector1", "vector2"]

    def create_initial_request_state(
        self,
        environment,
        env_state,
        user_prompt,
        *,
        message_history=None,
        usage=None,
    ):
        """Create initial request state for testing."""
        run_ctx = RunContext(
            deps=env_state,
            retry=0,
            tool_name=None,
            messages=[],
            model=TestModel(),
            usage=usage or RunUsage(),
        )

        # Create a proper ModelRequest
        model_request = ModelRequest.user_text_prompt("Test prompt")

        return ModelRequestState(
            run_ctx=run_ctx,
            environment=environment,
            model_request=model_request,
            _previous_state=None,
        )

    async def iter(
        self,
        environment,
        env_state,
        user_prompt,
        *,
        message_history=None,
        toolsets,
        usage_limits=None,
        usage=None,
        attacks=None,
        instrument=None,
    ) -> AsyncGenerator:
        """Simulate agent iteration for testing."""
        # Start with initial state
        run_ctx = RunContext(
            deps=env_state,
            retry=0,
            tool_name=None,
            messages=[],
            model=TestModel(),
            usage=usage or RunUsage(),
        )

        # Create a proper ModelRequest
        model_request = ModelRequest.user_text_prompt("Test prompt")

        initial_state = ModelRequestState(
            run_ctx=run_ctx,
            environment=environment,
            model_request=model_request,
            _previous_state=None,
        )
        yield initial_state

        injectable_state = None
        if self.should_find_injectable:
            # Simulate finding an injectable state
            # Create injectable content using InjectableStrContent
            injectable_content = InjectableStrContent(
                content="Test prompt with {INJECT:vector1} injection opportunity",
                default={
                    self.injectable_vector_ids[0]
                    if self.injectable_vector_ids
                    else "vector1": StrContentAttack(content="default")
                },
            )

            injectable_parts: list[ModelRequestPart | InjectableModelRequestPart] = [
                InjectableUserPromptPart(
                    content=[injectable_content],
                )
            ]

            injectable_state = InjectableModelRequestState(
                run_ctx=run_ctx,
                environment=environment,
                injectable_model_request_parts=injectable_parts,
                _previous_state=initial_state,
            )
            yield injectable_state

        # End with final state
        last_state = injectable_state if self.should_find_injectable else initial_state
        assert last_state is not None

        end_state = EndState(
            run_ctx=run_ctx,
            environment=environment,
            finish_reason=FinishReason.AGENT_LOOP_END,
            _previous_state=last_state,
        )
        yield end_state

    async def resume_iter_from_state(
        self,
        *,
        current_state,
        toolsets: Sequence[AbstractToolset],
        usage_limits=None,
        attacks=None,
        instrument=None,
    ) -> AsyncGenerator:
        """Resume iteration from a given state for testing."""
        # For testing, just yield the current state and then end
        yield current_state

        # Create an end state
        end_state = EndState(
            run_ctx=current_state.run_ctx,
            environment=current_state.environment,
            finish_reason=FinishReason.AGENT_LOOP_END,
            _previous_state=current_state,
        )
        yield end_state


@pytest.fixture
def attack_file_path(tmp_path: Path) -> Path:
    """Create a temporary attack file for testing."""
    attack_data = {
        "metadata": {"name": "test-attack"},
        "attacks": {
            "mock_task_benign:mock_task_malicious": {
                "vector1": {"content": "Test attack content", "kind": "str"},
                "vector2": {"content": "Another test attack", "kind": "str"},
            }
        },
    }

    attack_file = tmp_path / "test_attacks.json"
    with open(attack_file, "w") as f:
        json.dump(attack_data, f, indent=2)

    return attack_file


@pytest.fixture
def sample_attacks_dict() -> dict[str, InjectionAttacksDict[StrContentAttack]]:
    """Create a sample attacks dictionary for testing."""
    return {
        "mock_task_benign:mock_task_malicious": {
            "vector1": StrContentAttack(content="Test attack content"),
            "vector2": StrContentAttack(content="Another test attack"),
        }
    }


class TestDictAttackConfig:
    """Test DictAttackConfig validation and behavior."""

    def test_config_validation_with_dict(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test config validation with valid attacks dictionary."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        assert config.attack_name == "test-attack"
        assert "mock_task_benign:mock_task_malicious" in config.attacks_by_task

    def test_config_validation_empty_dict(self):
        """Test config validation with empty attacks dictionary."""
        config = DictAttackConfig(attacks_by_task={}, attack_name="empty-attack")
        assert config.attack_name == "empty-attack"
        assert len(config.attacks_by_task) == 0


class TestFileAttackConfig:
    """Test FileAttackConfig validation and behavior."""

    def test_config_validation_valid_path(self, attack_file_path: Path):
        """Test config validation with valid file path."""
        config = FileAttackConfig(file_path=str(attack_file_path))
        assert config.file_path == str(attack_file_path)

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid file path."""
        # Config creation doesn't validate the file path immediately
        config = FileAttackConfig(file_path="/nonexistent/path.json")
        # Validation happens when create_dict_attack_from_file is called
        with pytest.raises(FileNotFoundError):
            create_dict_attack_from_file(config)

    def test_config_validation_invalid_json(self, tmp_path: Path):
        """Test config validation with invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        config = FileAttackConfig(file_path=str(invalid_file))
        # Validation happens when create_dict_attack_from_file is called
        with pytest.raises(json.JSONDecodeError):
            create_dict_attack_from_file(config)


class TestDictAttackCreation:
    """Test DictAttack creation and basic properties."""

    def test_create_dict_attack_factory(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test DictAttack creation via factory function."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = create_dict_attack(config)

        assert isinstance(attack, DictAttack)
        assert attack.attack_name == "test-attack"

    def test_create_dict_attack_from_file_factory(self, attack_file_path: Path):
        """Test DictAttack creation from file via factory function."""
        config = FileAttackConfig(file_path=str(attack_file_path))
        attack = create_dict_attack_from_file(config)

        assert isinstance(attack, DictAttack)
        assert attack.attack_name == "test-attack"

    def test_dict_attack_has_correct_attacks(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test that DictAttack contains the correct attacks."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        assert attack.attack_name == "test-attack"
        assert "mock_task_benign:mock_task_malicious" in attack._config.attacks_by_task

        loaded_attacks = attack._config.attacks_by_task["mock_task_benign:mock_task_malicious"]
        assert "vector1" in loaded_attacks
        assert loaded_attacks["vector1"].content == "Test attack content"

    def test_dict_attack_from_file_loads_correctly(self, attack_file_path: Path):
        """Test that DictAttack from file loads attack file correctly."""
        config = FileAttackConfig(file_path=str(attack_file_path))
        attack = create_dict_attack_from_file(config)

        assert attack.attack_name == "test-attack"
        assert "mock_task_benign:mock_task_malicious" in attack._config.attacks_by_task

        loaded_attacks = attack._config.attacks_by_task["mock_task_benign:mock_task_malicious"]
        assert "vector1" in loaded_attacks
        assert loaded_attacks["vector1"].content == "Test attack content"

    def test_dict_attack_missing_task(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test DictAttack behavior when task is not in attacks."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        # Task not in attacks should return empty dict
        unknown_task_key = "unknown:task"
        assert unknown_task_key not in attack._config.attacks_by_task


class TestDictAttackExecution:
    """Test DictAttack execution with generate_attack method."""

    def _make_mock_checkpoint(
        self,
        couple: TaskCouple,
        available_vectors: list[str],
        agent_name: str = "test-agent",
    ) -> InjectionCheckpoint:
        """Helper to create a mock checkpoint for testing."""
        return InjectionCheckpoint(
            couple=couple,
            injectable_state=None,
            available_vectors=available_vectors,
            agent_name=agent_name,
            _checkpoint_id="test-checkpoint",
        )

    async def test_dict_attack_execution_with_attacks(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
        mock_task_couple: TaskCouple,
    ):
        """Test DictAttack generates attacks when they're available for the task."""
        from prompt_siren.attacks.simple_attack_base import InjectionContext

        # Create attack instance
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        # Create a mock checkpoint
        checkpoint = self._make_mock_checkpoint(
            couple=mock_task_couple,
            available_vectors=["vector1", "vector2"],
        )

        # Create an injection context from the checkpoint
        context = InjectionContext.from_checkpoint(checkpoint)

        # Generate the attack
        attacks = attack.generate_attack(context)

        # Verify the results
        assert isinstance(attacks, dict)
        assert "vector1" in attacks
        assert "vector2" in attacks
        assert attacks["vector1"].content == "Test attack content"
        assert attacks["vector2"].content == "Another test attack"

    async def test_dict_attack_from_file_execution(
        self, attack_file_path: Path, mock_task_couple: TaskCouple
    ):
        """Test DictAttack loaded from file generates attacks correctly."""
        from prompt_siren.attacks.simple_attack_base import InjectionContext

        # Create attack instance from file
        config = FileAttackConfig(file_path=str(attack_file_path))
        attack = create_dict_attack_from_file(config)

        # Create a mock checkpoint
        checkpoint = self._make_mock_checkpoint(
            couple=mock_task_couple,
            available_vectors=["vector1", "vector2"],
        )

        # Create an injection context from the checkpoint
        context = InjectionContext.from_checkpoint(checkpoint)

        # Generate the attack
        attacks = attack.generate_attack(context)

        # Verify the results
        assert isinstance(attacks, dict)
        assert len(attacks) == 2

    async def test_dict_attack_execution_no_injectable_states(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
        mock_task_couple: TaskCouple,
    ):
        """Test DictAttack returns all available attacks when generating."""
        from prompt_siren.attacks.simple_attack_base import InjectionContext

        # Create attack instance
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        # Create a mock checkpoint with no vectors
        checkpoint = self._make_mock_checkpoint(
            couple=mock_task_couple,
            available_vectors=[],
        )

        # Create an injection context from the checkpoint
        context = InjectionContext.from_checkpoint(checkpoint)

        # Generate the attack - should return available attacks for the couple
        attacks = attack.generate_attack(context)

        assert isinstance(attacks, dict)
        # DictAttack should still return its attacks based on couple ID
        assert len(attacks) == 2

    def test_attack_name_property(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test that attack_name property returns the correct name."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        assert attack.attack_name == "test-attack"

    def test_config_property(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
    ):
        """Test that config property returns the correct config."""
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        assert attack.config == config
        assert attack.config.attack_name == "test-attack"


class TestDictAttackIntegration:
    """Integration tests for DictAttack with attack file formats."""

    def test_attack_file_format_compatibility(self, tmp_path: Path):
        """Test that DictAttack works with AttackFile format."""
        # Create an AttackFile programmatically
        attacks_dict: dict[str, InjectionAttacksDict[InjectionAttack]] = {
            "mock_task_benign:mock_task_malicious": {
                "vector1": StrContentAttack(content="Test attack content"),
                "vector2": StrContentAttack(content="Another test attack"),
            }
        }

        attack_file = AttackFile.from_attacks_dict(attacks_dict, name="integration-test")

        # Save to temporary file
        temp_file = tmp_path / "integration_test.json"
        with open(temp_file, "w") as f:
            json.dump(attack_file.model_dump(), f, indent=2)

        # Load with DictAttack via file config
        config = FileAttackConfig(file_path=str(temp_file))
        attack = create_dict_attack_from_file(config)

        # Verify the attacks were loaded correctly
        assert attack.attack_name == "integration-test"
        assert "mock_task_benign:mock_task_malicious" in attack._config.attacks_by_task

        loaded_attacks = attack._config.attacks_by_task["mock_task_benign:mock_task_malicious"]
        assert "vector1" in loaded_attacks
        assert loaded_attacks["vector1"].content == "Test attack content"

    def test_empty_attack_file(self, tmp_path: Path):
        """Test DictAttack handles empty attack files gracefully."""
        empty_attack_data = {
            "metadata": {"name": "empty-attack"},
            "attacks": {},
        }

        empty_file = tmp_path / "empty_attacks.json"
        with open(empty_file, "w") as f:
            json.dump(empty_attack_data, f, indent=2)

        config = FileAttackConfig(file_path=str(empty_file))
        attack = create_dict_attack_from_file(config)

        assert attack.attack_name == "empty-attack"
        assert len(attack._config.attacks_by_task) == 0

    def test_direct_dict_creation(self):
        """Test creating DictAttack directly with a dictionary."""
        attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]] = {
            "mock_task_benign:mock_task_malicious": {
                "vector1": StrContentAttack(content="Direct dict attack"),
            }
        }

        config = DictAttackConfig(attacks_by_task=attacks_dict, attack_name="direct-dict-test")
        attack = create_dict_attack(config)

        assert attack.attack_name == "direct-dict-test"
        assert "mock_task_benign:mock_task_malicious" in attack._config.attacks_by_task
        assert (
            attack._config.attacks_by_task["mock_task_benign:mock_task_malicious"][
                "vector1"
            ].content
            == "Direct dict attack"
        )

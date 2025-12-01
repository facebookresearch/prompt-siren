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
from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.messages import ModelRequest, ModelRequestPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.usage import RunUsage

from ..conftest import (
    create_mock_environment,
    MockAgent,
    MockAgentConfig,
    MockDatasetConfig,
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
    """Test DictAttack execution with mock agents and environments."""

    async def test_dict_attack_execution_with_attacks(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
        mock_task_couple: TaskCouple,
    ):
        """Test DictAttack execution when attacks are available for the task."""

        benign_task = mock_task_couple.benign
        malicious_task = mock_task_couple.malicious

        # Create attack instance
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        # Create mock environment
        env_config = MockDatasetConfig(name="test-env")
        environment = create_mock_environment(env_config)

        # Use our enhanced testable agent
        agent = _TestableAgent(
            name="test-agent",
            custom_parameter="test",
            _config=MockAgentConfig(name="test-agent"),
            should_find_injectable=True,
            injectable_vector_ids=["vector1", "vector2"],
        )

        # Execute the attack
        async with environment.create_task_context(mock_task_couple) as env_state:
            end_state, used_attacks = await attack.attack(
                agent=agent,
                environment=environment,
                message_history=[],
                env_state=env_state,
                toolsets=[],
                benign_task=benign_task,
                malicious_task=malicious_task,
                usage_limits=UsageLimits(),
            )

        # Verify the results
        assert isinstance(end_state, EndState)
        assert isinstance(used_attacks, dict)

    async def test_dict_attack_from_file_execution(
        self, attack_file_path: Path, mock_task_couple: TaskCouple
    ):
        """Test DictAttack loaded from file executes correctly."""
        benign_task = mock_task_couple.benign
        malicious_task = mock_task_couple.malicious

        # Create attack instance from file
        config = FileAttackConfig(file_path=str(attack_file_path))
        attack = create_dict_attack_from_file(config)

        # Create mock environment
        env_config = MockDatasetConfig(name="test-env")
        environment = create_mock_environment(env_config)

        # Use our enhanced testable agent
        agent = _TestableAgent(
            name="test-agent",
            custom_parameter="test",
            _config=MockAgentConfig(name="test-agent"),
            should_find_injectable=True,
            injectable_vector_ids=["vector1", "vector2"],
        )

        # Execute the attack
        async with environment.create_task_context(mock_task_couple) as env_state:
            end_state, used_attacks = await attack.attack(
                agent=agent,
                environment=environment,
                message_history=[],
                env_state=env_state,
                toolsets=[],
                benign_task=benign_task,
                malicious_task=malicious_task,
                usage_limits=UsageLimits(),
            )

        # Verify the results
        assert isinstance(end_state, EndState)
        assert isinstance(used_attacks, dict)

    async def test_dict_attack_execution_no_injectable_states(
        self,
        sample_attacks_dict: dict[str, InjectionAttacksDict[StrContentAttack]],
        mock_task_couple: TaskCouple,
    ):
        """Test DictAttack execution when agent doesn't find injectable states."""

        benign_task = mock_task_couple.benign
        malicious_task = mock_task_couple.malicious

        # Create attack instance
        config = DictAttackConfig(attacks_by_task=sample_attacks_dict, attack_name="test-attack")
        attack = DictAttack(_config=config)

        # Create mock environment
        env_config = MockDatasetConfig(name="test-env")
        environment = create_mock_environment(env_config)

        # Use agent that doesn't find injectable states
        agent = _TestableAgent(
            name="test-agent",
            custom_parameter="test",
            _config=MockAgentConfig(name="test-agent"),
            should_find_injectable=False,  # No injectable states found
            injectable_vector_ids=[],
        )

        # Execute the attack
        async with environment.create_task_context(mock_task_couple) as env_state:
            end_state, attacks = await attack.attack(
                agent=agent,
                environment=environment,
                message_history=[],
                env_state=env_state,
                toolsets=[],
                benign_task=benign_task,
                malicious_task=malicious_task,
                usage_limits=UsageLimits(),
            )

        assert isinstance(end_state, EndState)
        assert isinstance(attacks, dict)
        assert len(attacks) == 2  # All attacks available for mock_task_benign:mock_task_malicious

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

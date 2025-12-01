# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for run.py execution paths.

Tests both benign and attack execution paths including error handling,
concurrency, persistence, and result aggregation.
"""

import sys
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from unittest.mock import AsyncMock

from prompt_siren.agents.plain import PlainAgent, PlainAgentConfig
from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
)
from prompt_siren.run import (
    run_single_tasks_without_attack,
    run_task_couples_with_attack,
)
from prompt_siren.run_persistence import ExecutionPersistence
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple, TaskResult
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from .conftest import (
    create_mock_benign_task,
    create_mock_task_couple,
    MockAttack,
    MockAttackConfig,
    MockDataset,
    MockEnvironment,
    MockEnvState,
)

pytestmark = pytest.mark.anyio


class TestRunBenignTasks:
    """Tests for run_benign_tasks function."""

    async def test_run_single_benign_task_success(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test running a single benign task successfully."""

        # Create agent with TestModel and task
        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        task = create_mock_benign_task("test_task", {"eval1": 1.0})

        # Run task
        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            usage_limits=None,
            max_concurrency=1,
            persistence=None,
            instrument=False,
        )

        # Verify results
        assert len(results) == 1
        assert results[0].task_id == "test_task"
        assert results[0].results == {"eval1": 1.0}

    async def test_run_multiple_benign_tasks(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test running multiple benign tasks."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        tasks = [
            create_mock_benign_task("task1", {"eval1": 1.0}),
            create_mock_benign_task("task2", {"eval1": 0.8}),
            create_mock_benign_task("task3", {"eval1": 0.5}),
        ]

        results = await run_single_tasks_without_attack(
            tasks=tasks,
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            max_concurrency=2,
            instrument=False,
        )

        assert len(results) == 3
        task_ids = {r.task_id for r in results}
        assert task_ids == {"task1", "task2", "task3"}

    async def test_run_benign_task_with_usage_limits(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test running benign task with usage limits."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        task = create_mock_benign_task("test_task", {"eval1": 1.0})

        usage_limits = UsageLimits(request_limit=10)

        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            usage_limits=usage_limits,
            instrument=False,
        )

        assert len(results) == 1
        assert results[0].task_id == "test_task"

    async def test_run_benign_task_with_persistence(
        self, mock_environment, mock_dataset: MockDataset, tmp_path
    ):
        """Test running benign task with persistence enabled."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        task = create_mock_benign_task("test_task", {"eval1": 1.0})

        # Create persistence
        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(type="mock", config={}),
            agent_config=AgentConfig(type="plain", config={"model": "test"}),
            attack_config=None,
        )

        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            persistence=persistence,
            instrument=False,
        )

        assert len(results) == 1

        # Verify files were created
        output_files = list(persistence.output_dir.glob("*.json"))
        assert len(output_files) == 1
        assert "test_task" in output_files[0].name

    async def test_run_benign_tasks_error_handling(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test error handling when a benign task fails."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        # Create a task that will fail evaluation
        def failing_evaluator(task_result):
            raise RuntimeError("Evaluation failed")

        failing_task = BenignTask(
            id="failing_task",
            prompt="This will fail",
            evaluators={"eval1": failing_evaluator},
        )

        # Should raise ExceptionGroup with the error
        with pytest.raises(ExceptionGroup) as exc_info:
            await run_single_tasks_without_attack(
                tasks=[failing_task],
                agent=agent,
                env=mock_environment,
                system_prompt=None,
                toolsets=mock_dataset.default_toolsets,
                instrument=False,
            )

        # Verify error contains our task
        assert "failing_task" in str(exc_info.value)
        assert len(exc_info.value.exceptions) == 1

    async def test_run_benign_tasks_partial_failure(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that partial failures are collected and raised together."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        def failing_evaluator(task_result):
            raise RuntimeError("Evaluation failed")

        tasks = [
            create_mock_benign_task("task1", {"eval1": 1.0}),
            BenignTask(
                id="failing_task",
                prompt="Fails",
                evaluators={"eval1": failing_evaluator},
            ),
            create_mock_benign_task("task3", {"eval1": 0.5}),
        ]

        with pytest.raises(ExceptionGroup) as exc_info:
            await run_single_tasks_without_attack(
                tasks=tasks,
                agent=agent,
                env=mock_environment,
                system_prompt=None,
                toolsets=mock_dataset.default_toolsets,
                instrument=False,
            )

        # Should have 1 failure
        assert len(exc_info.value.exceptions) == 1
        assert "failing_task" in str(exc_info.value)

    async def test_run_benign_tasks_empty_list(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test running with empty task list."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        results = await run_single_tasks_without_attack(
            tasks=[],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        assert results == []


class TestRunAttackTasks:
    """Tests for run_attack_tasks function."""

    async def test_run_multiple_couples(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test running multiple couples."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        couples = [
            create_mock_task_couple("couple1", {"eval1": 1.0}, {"eval1": 0.5}),
            create_mock_task_couple("couple2", {"eval1": 0.8}, {"eval1": 0.3}),
            create_mock_task_couple("couple3", {"eval1": 0.9}, {"eval1": 0.1}),
        ]

        results = await run_task_couples_with_attack(
            couples=couples,
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack,
            max_concurrency=2,
            instrument=False,
        )

        assert len(results) == 3

        # Verify all couples were executed
        benign_ids = {benign.task_id for benign, _ in results}
        malicious_ids = {malicious.task_id for _, malicious in results}

        assert benign_ids == {
            "couple1_benign",
            "couple2_benign",
            "couple3_benign",
        }
        assert malicious_ids == {
            "couple1_malicious",
            "couple2_malicious",
            "couple3_malicious",
        }

    async def test_run_couple_with_usage_limits(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test running couple with usage limits."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        couple = create_mock_task_couple("test_couple", {"eval1": 1.0}, {"eval1": 0.5})

        usage_limits = UsageLimits(request_limit=10)

        results = await run_task_couples_with_attack(
            couples=[couple],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack,
            usage_limits=usage_limits,
            instrument=False,
        )

        assert len(results) == 1

    async def test_run_couple_with_persistence(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
        tmp_path: Path,
    ):
        """Test running couple with persistence enabled."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        couple = create_mock_task_couple("test_couple", {"eval1": 1.0}, {"eval1": 0.5})

        # Create persistence with attack config
        persistence = ExecutionPersistence.create(
            base_dir=tmp_path,
            dataset_config=DatasetConfig(type="mock", config={}),
            agent_config=AgentConfig(type="plain", config={"model": "test"}),
            attack_config=AttackConfig(type="mock", config={}),
        )

        results = await run_task_couples_with_attack(
            couples=[couple],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack,
            persistence=persistence,
            instrument=False,
        )

        assert len(results) == 1

        # Verify files were created
        output_files = list(persistence.output_dir.glob("*.json"))
        assert len(output_files) == 1
        assert "test_couple" in output_files[0].name

    async def test_run_couples_error_handling(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test error handling when a couple fails."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        def failing_evaluator(task_result):
            raise RuntimeError("Evaluation failed")

        benign_task = create_mock_benign_task("benign", {"eval1": 1.0})
        malicious_task = MaliciousTask(
            id="malicious",
            goal="Fails",
            evaluators={"eval1": failing_evaluator},
        )
        failing_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        with pytest.raises(ExceptionGroup) as exc_info:
            await run_task_couples_with_attack(
                couples=[failing_couple],
                agent=agent,
                env=mock_environment,
                system_prompt=None,
                toolsets=mock_dataset.default_toolsets,
                attack=mock_attack,
                instrument=False,
            )

        assert "benign:malicious" in str(exc_info.value)
        assert len(exc_info.value.exceptions) == 1

    async def test_run_couples_partial_failure(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test that partial failures are collected and raised together."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        def failing_evaluator(task_result):
            raise RuntimeError("Evaluation failed")

        # Create one good couple and one failing couple
        good_couple = create_mock_task_couple("good", {"eval1": 1.0}, {"eval1": 0.5})

        benign_task = create_mock_benign_task("bad_benign", {"eval1": 1.0})
        malicious_task = MaliciousTask(
            id="bad_malicious",
            goal="Fails",
            evaluators={"eval1": failing_evaluator},
        )
        failing_couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        with pytest.raises(ExceptionGroup) as exc_info:
            await run_task_couples_with_attack(
                couples=[good_couple, failing_couple],
                agent=agent,
                env=mock_environment,
                system_prompt=None,
                toolsets=mock_dataset.default_toolsets,
                attack=mock_attack,
                instrument=False,
            )

        # Should have 1 failure
        assert len(exc_info.value.exceptions) == 1
        assert "bad_benign:bad_malicious" in str(exc_info.value)

    async def test_run_couples_empty_list(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test running with empty couple list."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        results = await run_task_couples_with_attack(
            couples=[],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack,
            instrument=False,
        )

        assert results == []


class TestConcurrency:
    """Tests for concurrent execution behavior."""

    async def test_benign_tasks_respect_concurrency_limit(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that benign tasks respect concurrency limits."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        tasks = [create_mock_benign_task(f"task{i}", {"eval1": 1.0}) for i in range(10)]

        # Run with concurrency limit
        results = await run_single_tasks_without_attack(
            tasks=tasks,
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            max_concurrency=3,
            instrument=False,
        )

        assert len(results) == 10

    async def test_couples_respect_concurrency_limit(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
        mock_attack: MockAttack,
    ):
        """Test that couples respect concurrency limits."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        couples = [
            create_mock_task_couple(f"couple{i}", {"eval1": 1.0}, {"eval1": 0.5}) for i in range(10)
        ]

        # Run with concurrency limit
        results = await run_task_couples_with_attack(
            couples=couples,
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack,
            max_concurrency=3,
            instrument=False,
        )

        assert len(results) == 10

    async def test_benign_tasks_unlimited_concurrency(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test benign tasks with unlimited concurrency."""

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))
        tasks = [create_mock_benign_task(f"task{i}", {"eval1": 1.0}) for i in range(5)]

        results = await run_single_tasks_without_attack(
            tasks=tasks,
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            max_concurrency=None,  # Unlimited
            instrument=False,
        )

        assert len(results) == 5


class TestSystemPromptIntegration:
    """Tests for system prompt integration with message history."""

    async def test_benign_task_message_history_ordering_with_system_prompt(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that system prompt + task history are ordered correctly in agent.run()."""
        # Create a custom agent that records message_history
        captured_message_history = []

        class RecordingAgent(PlainAgent):
            async def run(self, *args, **kwargs):
                # Capture the message_history parameter
                captured_message_history.append(kwargs.get("message_history"))
                # Call the original run method
                return await super().run(*args, **kwargs)

        # Create a benign task with existing message_history
        task_history: list[ModelMessage] = [ModelResponse(parts=[TextPart("Previous context")])]

        async def dummy_evaluator(task_result: TaskResult[MockEnvState]) -> float:
            return 1.0

        task = BenignTask(
            id="test_task",
            prompt="User prompt",
            evaluators={"eval1": dummy_evaluator},
            message_history=task_history,
        )

        agent = RecordingAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt="Test system prompt",
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        # Verify message_history was captured
        assert len(captured_message_history) == 1
        message_history = captured_message_history[0]

        # Verify ordering: [SystemPromptPart, ...task.message_history]
        assert len(message_history) == 2
        assert isinstance(message_history[0], ModelRequest)
        assert len(message_history[0].parts) == 1
        assert isinstance(message_history[0].parts[0], SystemPromptPart)
        assert message_history[0].parts[0].content == "Test system prompt"
        assert message_history[1] == task_history[0]

    async def test_benign_task_without_system_prompt(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that no system message is added when system_prompt=None."""
        # Create a custom agent that records message_history
        captured_message_history = []

        class RecordingAgent(PlainAgent):
            async def run(self, *args, **kwargs):
                # Capture the message_history parameter
                captured_message_history.append(kwargs.get("message_history"))
                # Call the original run method
                return await super().run(*args, **kwargs)

        task = create_mock_benign_task("test_task", {"eval1": 1.0})
        agent = RecordingAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        # Verify message_history was captured
        assert len(captured_message_history) == 1
        message_history = captured_message_history[0]

        # Verify no system prompt (empty list)
        assert message_history == []

    async def test_attack_receives_system_prompt_in_message_history(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
    ):
        """Test that attack.attack() receives system prompt in message_history parameter."""

        couple = create_mock_task_couple("test", {"eval1": 1.0}, {"eval1": 0.5})
        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        # Create a mock attack that captures the message_history parameter
        mock_attack_instance = MockAttack(
            attack_name="capture_test",
            custom_parameter=None,
            _config=MockAttackConfig(name="capture_test"),
        )

        # Mock the attack method to capture parameters
        original_attack = mock_attack_instance.attack
        attack_mock = AsyncMock(side_effect=original_attack)
        mock_attack_instance.attack = attack_mock

        await run_task_couples_with_attack(
            couples=[couple],
            agent=agent,
            env=mock_environment,
            system_prompt="Test system prompt",
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack_instance,
            instrument=False,
        )

        # Verify attack.attack was called
        assert attack_mock.call_count == 1
        call_kwargs = attack_mock.call_args.kwargs

        # Extract message_history from the call
        message_history = call_kwargs["message_history"]

        # Verify system prompt is present
        assert len(message_history) >= 1
        assert isinstance(message_history[0], ModelRequest)
        assert len(message_history[0].parts) == 1
        assert isinstance(message_history[0].parts[0], SystemPromptPart)
        assert message_history[0].parts[0].content == "Test system prompt"

    async def test_attack_with_system_prompt_and_task_message_history(
        self,
        mock_environment: MockEnvironment,
        mock_dataset: MockDataset,
    ):
        """Test attack receives both system prompt and task with message_history."""
        # Create benign task with message_history
        task_history: list[ModelMessage] = [ModelResponse(parts=[TextPart("Previous context")])]

        async def dummy_evaluator(task_result: TaskResult[MockEnvState]) -> float:
            return 1.0

        async def dummy_evaluator_malicious(task_result: TaskResult[MockEnvState]) -> float:
            return 0.5

        benign_task = BenignTask(
            id="benign_with_history",
            prompt="Benign prompt",
            evaluators={"eval1": dummy_evaluator},
            message_history=task_history,
        )
        malicious_task = MaliciousTask(
            id="malicious",
            goal="Malicious goal",
            evaluators={"eval1": dummy_evaluator_malicious},
        )
        couple = TaskCouple(benign=benign_task, malicious=malicious_task)

        agent = PlainAgent(PlainAgentConfig(model=TestModel(), model_settings=ModelSettings()))

        # Create a mock attack that captures the message_history parameter
        mock_attack_instance = MockAttack(
            attack_name="capture_test",
            custom_parameter=None,
            _config=MockAttackConfig(name="capture_test"),
        )

        # Mock the attack method to capture parameters
        original_attack = mock_attack_instance.attack
        attack_mock = AsyncMock(side_effect=original_attack)
        mock_attack_instance.attack = attack_mock

        await run_task_couples_with_attack(
            couples=[couple],
            agent=agent,
            env=mock_environment,
            system_prompt="Test system prompt",
            toolsets=mock_dataset.default_toolsets,
            attack=mock_attack_instance,
            instrument=False,
        )

        # Verify attack.attack was called
        assert attack_mock.call_count == 1
        call_kwargs = attack_mock.call_args.kwargs

        # Verify system prompt is in message_history parameter
        message_history = call_kwargs["message_history"]
        assert len(message_history) == 1
        assert isinstance(message_history[0], ModelRequest)
        assert len(message_history[0].parts) == 1
        assert isinstance(message_history[0].parts[0], SystemPromptPart)
        assert message_history[0].parts[0].content == "Test system prompt"

        # Verify benign_task has its message_history (attack is responsible for merging)
        received_benign_task = call_kwargs["benign_task"]
        assert received_benign_task.message_history == task_history


class TestMaliciousTaskCustomPrompt:
    """Tests for MaliciousTask custom prompt feature."""

    @staticmethod
    def _create_prompt_capture_model(captured_prompts: list[str]) -> FunctionModel:
        """Create a FunctionModel that captures user prompts."""

        def capture_prompt(messages: list[ModelMessage], info) -> ModelResponse:
            # Extract the user prompt from the last message
            for part in messages[-1].parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        captured_prompts.append(content)
            return ModelResponse(parts=[TextPart(content="done")])

        return FunctionModel(capture_prompt)

    async def test_malicious_task_with_custom_prompt_in_benign_mode(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that MaliciousTask uses custom prompt field when set."""
        captured_prompts: list[str] = []

        async def dummy_evaluator(task_result: TaskResult[MockEnvState]) -> float:
            return 1.0

        # Create malicious task with custom prompt
        task = MaliciousTask(
            id="test_malicious",
            goal="This is the goal",
            prompt="This is the custom prompt",  # Different from goal
            evaluators={"eval1": dummy_evaluator},
        )

        agent = PlainAgent(
            PlainAgentConfig(
                model=self._create_prompt_capture_model(captured_prompts),
                model_settings=ModelSettings(),
            )
        )

        # Run in benign mode
        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        # Verify the custom prompt was used, not the goal
        assert len(captured_prompts) == 1
        assert captured_prompts[0] == "This is the custom prompt"

        # Verify task completed successfully
        assert len(results) == 1
        assert results[0].task_id == "test_malicious"

    async def test_malicious_task_without_custom_prompt_uses_goal(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that MaliciousTask falls back to goal when prompt is None."""
        captured_prompts: list[str] = []

        async def dummy_evaluator(task_result: TaskResult[MockEnvState]) -> float:
            return 1.0

        # Create malicious task without custom prompt
        task = MaliciousTask(
            id="test_malicious",
            goal="This is the goal",
            prompt=None,  # Explicitly None
            evaluators={"eval1": dummy_evaluator},
        )

        agent = PlainAgent(
            PlainAgentConfig(
                model=self._create_prompt_capture_model(captured_prompts),
                model_settings=ModelSettings(),
            )
        )

        # Run in benign mode
        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        # Verify the goal was used as fallback
        assert len(captured_prompts) == 1
        assert captured_prompts[0] == "This is the goal"

        # Verify task completed successfully
        assert len(results) == 1
        assert results[0].task_id == "test_malicious"

    async def test_malicious_task_with_empty_prompt_uses_goal(
        self, mock_environment: MockEnvironment, mock_dataset: MockDataset
    ):
        """Test that MaliciousTask falls back to goal when prompt is empty string."""
        captured_prompts: list[str] = []

        async def dummy_evaluator(task_result: TaskResult[MockEnvState]) -> float:
            return 1.0

        # Create malicious task with empty prompt
        task = MaliciousTask(
            id="test_malicious",
            goal="This is the goal",
            prompt="",  # Empty string
            evaluators={"eval1": dummy_evaluator},
        )

        agent = PlainAgent(
            PlainAgentConfig(
                model=self._create_prompt_capture_model(captured_prompts),
                model_settings=ModelSettings(),
            )
        )

        # Run in benign mode
        results = await run_single_tasks_without_attack(
            tasks=[task],
            agent=agent,
            env=mock_environment,
            system_prompt=None,
            toolsets=mock_dataset.default_toolsets,
            instrument=False,
        )

        # Verify the goal was used due to the `or` operator
        assert len(captured_prompts) == 1
        assert captured_prompts[0] == "This is the goal"

        # Verify task completed successfully
        assert len(results) == 1
        assert results[0].task_id == "test_malicious"

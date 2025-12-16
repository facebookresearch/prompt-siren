# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for JobPersistence class."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prompt_siren.config.experiment_config import (
    AgentConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job.models import (
    CONFIG_FILENAME,
    ExceptionInfo,
    INDEX_FILENAME,
    JobConfig,
    JobStats,
    RESULT_FILENAME,
    TASK_RESULT_FILENAME,
    TaskRunResult,
)
from prompt_siren.job.persistence import (
    _load_config_yaml,
    _save_config_yaml,
    JobPersistence,
)
from prompt_siren.tasks import (
    BenignTask,
    EvaluationResult,
    MaliciousTask,
    TaskCouple,
    TaskResult,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.usage import RunUsage


@pytest.fixture
def job_config() -> JobConfig:
    """Create a minimal job config for testing."""
    return JobConfig(
        job_name="test_job",
        execution_mode="benign",
        created_at=datetime.now(),
        dataset=DatasetConfig(type="test", config={}),
        agent=AgentConfig(type="plain", config={"model": "test"}),
        attack=None,
        execution=ExecutionConfig(concurrency=1),
        telemetry=TelemetryConfig(trace_console=False),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


@pytest.fixture
def job_persistence(tmp_path: Path, job_config: JobConfig) -> JobPersistence:
    """Create a JobPersistence instance for testing."""
    return JobPersistence.create(tmp_path, job_config)


@pytest.fixture
def mock_task_span() -> MagicMock:
    """Create a mock logfire span for testing."""
    mock_span = MagicMock()
    mock_context = MagicMock()
    mock_context.trace_id = 0x123456789ABCDEF0
    mock_context.span_id = 0xABCDEF01
    mock_span.get_span_context.return_value = mock_context
    return mock_span


class TestJobPersistenceLoad:
    """Tests for JobPersistence.load method."""

    def test_loads_existing_job(self, tmp_path: Path, job_config: JobConfig):
        """Test loading an existing job preserves config."""
        job_dir = tmp_path / "existing_job"
        JobPersistence.create(job_dir, job_config)

        loaded = JobPersistence.load(job_dir)
        assert loaded.job_config.job_name == job_config.job_name

    def test_raises_for_missing_job(self, tmp_path: Path):
        """Test that loading nonexistent job raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Job config not found"):
            JobPersistence.load(tmp_path / "nonexistent")

    def test_does_not_overwrite_existing_config_on_create(
        self, tmp_path: Path, job_config: JobConfig
    ):
        """Test that create() preserves existing config file."""
        job_dir = tmp_path / "existing_job"
        job_dir.mkdir()

        custom_content = "# Custom config\njob_name: custom"
        (job_dir / CONFIG_FILENAME).write_text(custom_content)

        JobPersistence.create(job_dir, job_config)

        assert "custom" in (job_dir / CONFIG_FILENAME).read_text()


class TestGetCompletedRuns:
    """Tests for JobPersistence.get_completed_runs method."""

    def test_returns_only_completed_run_ids(self, job_persistence: JobPersistence):
        """Test that only runs with result.json are returned as completed."""
        # Create completed runs
        run_ids = ["abc11111", "def33333"]
        for run_id in run_ids:
            run_dir = job_persistence.get_task_run_dir("task1", run_id)
            run_dir.mkdir(parents=True)
            result = TaskRunResult(
                task_id="task1",
                run_id=run_id,
                finished_at=datetime.now(),
            )
            (run_dir / TASK_RESULT_FILENAME).write_text(result.model_dump_json())

        # Create incomplete run (directory only, no result.json)
        incomplete_dir = job_persistence.get_task_run_dir("task1", "xyz22222")
        incomplete_dir.mkdir(parents=True)

        completed = job_persistence.get_completed_runs("task1")
        assert set(completed) == set(run_ids)  # incomplete run excluded


class TestGetRunStatus:
    """Tests for JobPersistence.get_run_status method."""

    def test_returns_correct_status_for_each_state(self, job_persistence: JobPersistence):
        """Test that correct status is returned for pending, incomplete, completed, and failed."""
        # Pending (no directory)
        status, _ = job_persistence.get_run_status("task1", "aaa11111")
        assert status == "pending"

        # Incomplete (directory exists, no result)
        run_dir = job_persistence.get_task_run_dir("task1", "bbb22222")
        run_dir.mkdir(parents=True)
        status, _ = job_persistence.get_run_status("task1", "bbb22222")
        assert status == "incomplete"

        # Completed (directory + result without exception)
        run_id_completed = "ccc33333"
        run_dir = job_persistence.get_task_run_dir("task1", run_id_completed)
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task1", run_id=run_id_completed, finished_at=datetime.now(), benign_score=1.0
        )
        (run_dir / TASK_RESULT_FILENAME).write_text(result.model_dump_json())
        status, loaded = job_persistence.get_run_status("task1", run_id_completed)
        assert status == "completed"
        assert loaded is not None
        assert loaded.benign_score == 1.0

        # Failed (directory + result with exception)
        run_id_failed = "ddd44444"
        run_dir = job_persistence.get_task_run_dir("task1", run_id_failed)
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task1",
            run_id=run_id_failed,
            finished_at=datetime.now(),
            exception_info=ExceptionInfo(
                exception_type="RuntimeError",
                exception_message="error",
                exception_traceback="",
                occurred_at=datetime.now(),
            ),
        )
        (run_dir / TASK_RESULT_FILENAME).write_text(result.model_dump_json())
        status, loaded = job_persistence.get_run_status("task1", run_id_failed)
        assert status == "failed"
        assert loaded is not None
        assert loaded.exception_info is not None


class TestSaveTaskRun:
    """Tests for JobPersistence.save_task_run method."""

    def test_saves_task_result_with_correct_score(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that save_task_run persists the evaluation score correctly."""

        async def dummy_eval(task_result: TaskResult) -> float:
            return 0.85

        task = BenignTask(id="task1", prompt="test", evaluators={"eval1": dummy_eval})
        evaluation = EvaluationResult(task_id="task1", results={"eval1": 0.85})
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
        usage = RunUsage()

        run_dir = job_persistence.save_task_run(
            task=task,
            evaluation=evaluation,
            messages=messages,
            usage=usage,
            task_span=mock_task_span,
            started_at=datetime.now(),
        )

        loaded = TaskRunResult.model_validate_json((run_dir / TASK_RESULT_FILENAME).read_text())
        assert loaded.task_id == "task1"
        assert loaded.benign_score == 0.85

    def test_appends_entry_to_index(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that save_task_run appends entry to index.jsonl."""

        async def dummy_eval(task_result: TaskResult) -> float:
            return 1.0

        task = BenignTask(id="task1", prompt="test", evaluators={"eval1": dummy_eval})
        evaluation = EvaluationResult(task_id="task1", results={"eval1": 1.0})
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
        usage = RunUsage()

        job_persistence.save_task_run(
            task=task,
            evaluation=evaluation,
            messages=messages,
            usage=usage,
            task_span=mock_task_span,
            started_at=datetime.now(),
        )

        index_content = (job_persistence.job_dir / INDEX_FILENAME).read_text()
        assert "task1" in index_content


class TestSaveCoupleRun:
    """Tests for JobPersistence.save_couple_run method."""

    def test_saves_both_benign_and_attack_scores(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that couple run saves both benign and attack scores."""

        async def benign_eval(task_result: TaskResult) -> float:
            return 0.9

        async def malicious_eval(task_result: TaskResult) -> float:
            return 0.3

        benign = BenignTask(id="benign1", prompt="benign", evaluators={"eval1": benign_eval})
        malicious = MaliciousTask(id="mal1", goal="goal", evaluators={"eval1": malicious_eval})
        couple = TaskCouple(benign=benign, malicious=malicious)

        benign_eval_result = EvaluationResult(task_id="benign1", results={"eval1": 0.9})
        malicious_eval_result = EvaluationResult(task_id="mal1", results={"eval1": 0.3})
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
        usage = RunUsage()

        run_dir = job_persistence.save_couple_run(
            couple=couple,
            benign_eval=benign_eval_result,
            malicious_eval=malicious_eval_result,
            messages=messages,
            usage=usage,
            task_span=mock_task_span,
            started_at=datetime.now(),
        )

        loaded = TaskRunResult.model_validate_json((run_dir / TASK_RESULT_FILENAME).read_text())
        assert loaded.benign_score == 0.9
        assert loaded.attack_score == 0.3


class TestLoadIndex:
    """Tests for JobPersistence.load_index method."""

    def test_loads_all_saved_entries(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that load_index returns all entries from index.jsonl."""

        async def dummy_eval(task_result: TaskResult) -> float:
            return 1.0

        # Save multiple task runs
        for i in range(3):
            task = BenignTask(id=f"task{i}", prompt="test", evaluators={"eval1": dummy_eval})
            evaluation = EvaluationResult(task_id=f"task{i}", results={"eval1": 1.0})
            messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
            usage = RunUsage()

            job_persistence.save_task_run(
                task=task,
                evaluation=evaluation,
                messages=messages,
                usage=usage,
                task_span=mock_task_span,
                started_at=datetime.now(),
            )

        entries = job_persistence.load_index()
        assert len(entries) == 3


class TestDeleteRun:
    """Tests for JobPersistence.delete_run method."""

    def test_deletes_run_directory_and_returns_true(self, job_persistence: JobPersistence):
        """Test that delete_run removes the run directory."""
        run_id = "abc12345"
        run_dir = job_persistence.get_task_run_dir("task1", run_id)
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text("{}")

        result = job_persistence.delete_run("task1", run_id)
        assert result is True
        assert not run_dir.exists()

    def test_returns_false_for_nonexistent_run(self, job_persistence: JobPersistence):
        """Test that delete_run returns False if run doesn't exist."""
        result = job_persistence.delete_run("nonexistent", "xyz99999")
        assert result is False


class TestUpdateJobResult:
    """Tests for JobPersistence.update_job_result method."""

    def test_updates_result_file_with_stats_and_completion_status(
        self, job_persistence: JobPersistence
    ):
        """Test that update_job_result writes stats and is_complete to result.json."""
        stats = JobStats(
            n_total_tasks=10,
            n_completed_runs=10,
            avg_benign_score=0.85,
        )

        job_persistence.update_job_result(stats, is_complete=True)

        content = (job_persistence.job_dir / RESULT_FILENAME).read_text()
        assert "0.85" in content
        assert '"is_complete": true' in content


class TestRemoveIndexEntriesByPaths:
    """Tests for JobPersistence.remove_index_entries_by_paths method."""

    def test_removes_specified_paths_from_index(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that remove_index_entries_by_paths removes specified entries."""

        async def dummy_eval(task_result: TaskResult) -> float:
            return 1.0

        # Create multiple task runs and track their paths
        run_dirs: dict[str, Path] = {}
        for i in range(3):
            task = BenignTask(id=f"task{i}", prompt="test", evaluators={"eval1": dummy_eval})
            evaluation = EvaluationResult(task_id=f"task{i}", results={"eval1": 1.0})
            messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
            usage = RunUsage()

            run_dir = job_persistence.save_task_run(
                task=task,
                evaluation=evaluation,
                messages=messages,
                usage=usage,
                task_span=mock_task_span,
                started_at=datetime.now(),
            )
            run_dirs[f"task{i}"] = run_dir

        # Verify all 3 entries exist
        entries = job_persistence.load_index()
        assert len(entries) == 3

        # Remove task1's entry using its actual path
        task1_path = run_dirs["task1"].relative_to(job_persistence.job_dir)
        job_persistence.remove_index_entries_by_paths({task1_path})

        # Verify only 2 entries remain
        entries = job_persistence.load_index()
        assert len(entries) == 2
        assert all(entry.task_id != "task1" for entry in entries)
        assert any(entry.task_id == "task0" for entry in entries)
        assert any(entry.task_id == "task2" for entry in entries)

    def test_does_nothing_if_index_doesnt_exist(self, job_persistence: JobPersistence):
        """Test that remove_index_entries_by_paths handles missing index gracefully."""
        # Remove index file if it exists
        index_path = job_persistence.job_dir / INDEX_FILENAME
        if index_path.exists():
            index_path.unlink()

        # Should not raise an error
        job_persistence.remove_index_entries_by_paths({Path("nonexistent/abc12345")})

    def test_handles_empty_paths_set(
        self, job_persistence: JobPersistence, mock_task_span: MagicMock
    ):
        """Test that remove_index_entries_by_paths handles empty paths set."""

        async def dummy_eval(task_result: TaskResult) -> float:
            return 1.0

        task = BenignTask(id="task1", prompt="test", evaluators={"eval1": dummy_eval})
        evaluation = EvaluationResult(task_id="task1", results={"eval1": 1.0})
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart("response")])]
        usage = RunUsage()

        job_persistence.save_task_run(
            task=task,
            evaluation=evaluation,
            messages=messages,
            usage=usage,
            task_span=mock_task_span,
            started_at=datetime.now(),
        )

        # Remove with empty set should keep all entries
        job_persistence.remove_index_entries_by_paths(set())

        entries = job_persistence.load_index()
        assert len(entries) == 1


class TestConfigYamlRoundtrip:
    """Tests for config YAML serialization."""

    def test_config_survives_save_and_load(self, tmp_path: Path, job_config: JobConfig):
        """Test that job config can be saved and loaded without data loss."""
        config_path = tmp_path / "config.yaml"

        _save_config_yaml(config_path, job_config)
        loaded = _load_config_yaml(config_path)

        assert loaded.job_name == job_config.job_name
        assert loaded.execution_mode == job_config.execution_mode
        assert loaded.dataset == job_config.dataset
        assert loaded.agent == job_config.agent

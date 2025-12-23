# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for Job class and related functions."""

import re
from datetime import datetime
from pathlib import Path

import pytest
from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job import Job, JobConfigMismatchError
from prompt_siren.job.job import _apply_resume_overrides
from prompt_siren.job.models import (
    CONFIG_FILENAME,
    ExceptionInfo,
    JobConfig,
    RunIndexEntry,
    TaskRunResult,
)
from prompt_siren.job.persistence import _save_config_yaml


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    """Create a minimal experiment config for testing."""
    return ExperimentConfig(
        name="test_experiment",
        agent=AgentConfig(type="plain", config={"model": "test"}),
        dataset=DatasetConfig(type="mock", config={}),
        execution=ExecutionConfig(concurrency=2),
        telemetry=TelemetryConfig(trace_console=False),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


@pytest.fixture
def experiment_config_with_attack() -> ExperimentConfig:
    """Create an experiment config with attack for testing."""
    return ExperimentConfig(
        name="test_attack_experiment",
        agent=AgentConfig(type="plain", config={"model": "test"}),
        dataset=DatasetConfig(type="mock", config={}),
        attack=AttackConfig(type="template_string", config={"attack_template": "test"}),
        execution=ExecutionConfig(concurrency=4),
        telemetry=TelemetryConfig(trace_console=True),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


class TestJobCreate:
    """Tests for Job.create method."""

    def test_creates_job_in_correct_directory(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that Job.create creates job in the expected location."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test_agent",
        )
        assert job.job_dir == tmp_path / "test_job"
        assert job.job_dir.exists()

    def test_auto_generates_job_name_from_components(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that job name is auto-generated from dataset, agent, and mode."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name=None,
            agent_name="plain:gpt-5",
        )
        # Format: <dataset>_<sanitized-agent-name>_<attack|benign>_<YYYY-MM-DD_HH-MM-SS>
        pattern = r"^mock_plain_gpt-5_benign_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        assert re.match(pattern, job.job_config.job_name)

    def test_requires_agent_name_for_auto_generated_name(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that agent_name is required when job_name is not provided."""
        with pytest.raises(ValueError, match="agent_name is required"):
            Job.create(
                experiment_config=experiment_config,
                execution_mode="benign",
                jobs_dir=tmp_path,
                job_name=None,
                agent_name=None,
            )

    def test_prevents_duplicate_job_names(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that creating a job with existing name raises error."""
        Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="existing_job",
            agent_name="test",
        )

        with pytest.raises(FileExistsError, match="Job directory already exists"):
            Job.create(
                experiment_config=experiment_config,
                execution_mode="benign",
                jobs_dir=tmp_path,
                job_name="existing_job",
                agent_name="test",
            )


class TestJobResume:
    """Tests for Job.resume method."""

    def test_resumes_existing_job(self, experiment_config: ExperimentConfig, tmp_path: Path):
        """Test resuming an existing job loads config."""
        original_job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        resumed_job = Job.resume(job_dir=original_job.job_dir)
        assert resumed_job.job_config.job_name == "test_job"

    def test_raises_for_nonexistent_job(self, tmp_path: Path):
        """Test that resuming nonexistent job raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Job config not found"):
            Job.resume(job_dir=tmp_path / "nonexistent")

    def test_applies_execution_overrides(self, experiment_config: ExperimentConfig, tmp_path: Path):
        """Test that execution-related overrides are applied on resume."""
        original_job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )
        assert original_job.job_config.execution.concurrency == 2

        resumed_job = Job.resume(
            job_dir=original_job.job_dir,
            overrides=["execution.concurrency=8"],
        )
        assert resumed_job.job_config.execution.concurrency == 8

    def test_rejects_immutable_overrides(self, experiment_config: ExperimentConfig, tmp_path: Path):
        """Test that dataset/agent/attack overrides are rejected on resume."""
        original_job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        with pytest.raises(JobConfigMismatchError, match="Cannot modify dataset"):
            Job.resume(job_dir=original_job.job_dir, overrides=["dataset.type=other"])

        with pytest.raises(JobConfigMismatchError, match="Cannot modify agent"):
            Job.resume(job_dir=original_job.job_dir, overrides=["agent.config.model=other"])

    def test_rejects_unknown_overrides(self, experiment_config: ExperimentConfig, tmp_path: Path):
        """Test that unknown override keys are rejected."""
        original_job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        with pytest.raises(JobConfigMismatchError, match="Unknown override key"):
            Job.resume(job_dir=original_job.job_dir, overrides=["unknown.field=value"])


class TestJobCleanupForRetry:
    """Tests for retry cleanup behavior on Job.resume."""

    def test_cancelled_error_deleted_when_specified(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that CancelledError runs are deleted when specified in retry_on_errors."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        # Create a failed run with CancelledError
        run_id = "abc12345"
        run_dir = job.job_dir / "task1" / run_id
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task1",
            run_id=run_id,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            exception_info=ExceptionInfo(
                exception_type="CancelledError",
                exception_message="cancelled",
                exception_traceback="",
                occurred_at=datetime.now(),
            ),
        )
        (run_dir / "result.json").write_text(result.model_dump_json())

        index_entry = RunIndexEntry(
            task_id="task1",
            run_id=run_id,
            timestamp=datetime.now(),
            benign_score=None,
            attack_score=None,
            exception_type="CancelledError",
            path=Path(f"task1/{run_id}"),
        )
        (job.job_dir / "index.jsonl").write_text(index_entry.model_dump_json() + "\n")

        # Resume with CancelledError in retry list
        Job.resume(job_dir=job.job_dir, retry_on_errors=["CancelledError"])

        # Run should be deleted
        assert not run_dir.exists()

    def test_cancelled_error_preserved_when_not_in_retry_list(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that CancelledError runs are preserved when not in retry_on_errors."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        # Create a failed run with CancelledError
        run_id = "abc12345"
        run_dir = job.job_dir / "task1" / run_id
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task1",
            run_id=run_id,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            exception_info=ExceptionInfo(
                exception_type="CancelledError",
                exception_message="cancelled",
                exception_traceback="",
                occurred_at=datetime.now(),
            ),
        )
        (run_dir / "result.json").write_text(result.model_dump_json())

        index_entry = RunIndexEntry(
            task_id="task1",
            run_id=run_id,
            timestamp=datetime.now(),
            benign_score=None,
            attack_score=None,
            exception_type="CancelledError",
            path=Path(f"task1/{run_id}"),
        )
        (job.job_dir / "index.jsonl").write_text(index_entry.model_dump_json() + "\n")

        # Resume with only TimeoutError in retry list (not CancelledError)
        Job.resume(job_dir=job.job_dir, retry_on_errors=["TimeoutError"])

        # CancelledError run should be preserved
        assert run_dir.exists()

    def test_retry_on_errors_deletes_matching_failed_runs(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that retry_on_errors deletes runs with specified error types."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        # Create a failed run with TimeoutError
        run_id = "abc12345"
        run_dir = job.job_dir / "task1" / run_id
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task1",
            run_id=run_id,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            exception_info=ExceptionInfo(
                exception_type="TimeoutError",
                exception_message="timed out",
                exception_traceback="",
                occurred_at=datetime.now(),
            ),
        )
        (run_dir / "result.json").write_text(result.model_dump_json())

        index_entry = RunIndexEntry(
            task_id="task1",
            run_id=run_id,
            timestamp=datetime.now(),
            benign_score=None,
            attack_score=None,
            exception_type="TimeoutError",
            path=Path(f"task1/{run_id}"),
        )
        (job.job_dir / "index.jsonl").write_text(index_entry.model_dump_json() + "\n")

        Job.resume(job_dir=job.job_dir, retry_on_errors=["TimeoutError"])

        assert not run_dir.exists()

    def test_cleanup_deletes_incomplete_runs_only_with_retry_flags(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that incomplete runs are only deleted when retry_on_errors is set."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        incomplete_run = job.job_dir / "task1" / "abc12345"
        incomplete_run.mkdir(parents=True)

        # Without retry flags, incomplete runs are preserved
        Job.resume(job_dir=job.job_dir)
        assert incomplete_run.exists()

        # With retry_on_errors, incomplete runs are also deleted
        Job.resume(job_dir=job.job_dir, retry_on_errors=["TimeoutError"])
        assert not incomplete_run.exists()

    def test_cleanup_updates_index_when_runs_deleted(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that index.jsonl is updated when runs are deleted during retry cleanup."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        # Create failed runs with index entries
        run_ids = ["aaa11111", "bbb22222", "ccc33333"]
        for i in range(3):
            run_id = run_ids[i]
            run_dir = job.job_dir / f"task{i}" / run_id
            run_dir.mkdir(parents=True)
            result = TaskRunResult(
                task_id=f"task{i}",
                run_id=run_id,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                exception_info=ExceptionInfo(
                    exception_type="TimeoutError" if i < 2 else "RuntimeError",
                    exception_message="error",
                    exception_traceback="",
                    occurred_at=datetime.now(),
                ),
            )
            (run_dir / "result.json").write_text(result.model_dump_json())

            index_entry = RunIndexEntry(
                task_id=f"task{i}",
                run_id=run_id,
                timestamp=datetime.now(),
                benign_score=None,
                attack_score=None,
                exception_type="TimeoutError" if i < 2 else "RuntimeError",
                path=Path(f"task{i}/{run_id}"),
            )
            with open(job.job_dir / "index.jsonl", "a") as f:
                f.write(index_entry.model_dump_json() + "\n")

        # Verify index has 3 entries
        resumed_job = Job.resume(job_dir=job.job_dir, retry_on_errors=["TimeoutError"])
        entries = resumed_job.persistence.load_index()

        # Only the RuntimeError entry should remain (task2)
        assert len(entries) == 1
        assert entries[0].task_id == "task2"
        assert entries[0].exception_type == "RuntimeError"

        # Verify the TimeoutError runs were deleted but RuntimeError run remains
        assert not (job.job_dir / "task0" / run_ids[0]).exists()
        assert not (job.job_dir / "task1" / run_ids[1]).exists()
        assert (job.job_dir / "task2" / run_ids[2]).exists()  # Not deleted (RuntimeError)


class TestJobRunCounts:
    """Tests for run counting and filtering methods."""

    def test_get_run_counts_counts_all_runs(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that get_run_counts counts all runs including errored ones."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        # Create 3 runs for task1 (1 success, 2 errors)
        for run_id, exc_type in [
            ("aaa11111", None),
            ("bbb22222", "TimeoutError"),
            ("ccc33333", "RuntimeError"),
        ]:
            run_dir = job.job_dir / "task1" / run_id
            run_dir.mkdir(parents=True)
            result = TaskRunResult(
                task_id="task1",
                run_id=run_id,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                exception_info=ExceptionInfo(
                    exception_type=exc_type,
                    exception_message="error",
                    exception_traceback="",
                    occurred_at=datetime.now(),
                )
                if exc_type
                else None,
            )
            (run_dir / "result.json").write_text(result.model_dump_json())

            index_entry = RunIndexEntry(
                task_id="task1",
                run_id=run_id,
                timestamp=datetime.now(),
                benign_score=1.0 if exc_type is None else None,
                attack_score=None,
                exception_type=exc_type,
                path=Path(f"task1/{run_id}"),
            )
            with open(job.job_dir / "index.jsonl", "a") as f:
                f.write(index_entry.model_dump_json() + "\n")

        # Create 1 run for task2
        run_dir = job.job_dir / "task2" / "ddd44444"
        run_dir.mkdir(parents=True)
        result = TaskRunResult(
            task_id="task2",
            run_id="ddd44444",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        (run_dir / "result.json").write_text(result.model_dump_json())
        index_entry = RunIndexEntry(
            task_id="task2",
            run_id="ddd44444",
            timestamp=datetime.now(),
            benign_score=1.0,
            attack_score=None,
            exception_type=None,
            path=Path("task2/ddd44444"),
        )
        with open(job.job_dir / "index.jsonl", "a") as f:
            f.write(index_entry.model_dump_json() + "\n")

        # All runs should be counted (both successful and errored)
        run_counts = job.persistence.get_run_counts()
        assert run_counts == {"task1": 3, "task2": 1}

    def test_filter_tasks_needing_runs(self, experiment_config: ExperimentConfig, tmp_path: Path):
        """Test that filter_tasks_needing_runs filters correctly."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
            n_runs_per_task=2,
        )

        # Create 2 runs for task1 (at limit), 1 for task2 (needs 1 more), 0 for task3
        for task_id, run_id in [
            ("task1", "aaa11111"),
            ("task1", "bbb22222"),
            ("task2", "ccc33333"),
        ]:
            run_dir = job.job_dir / task_id / run_id
            run_dir.mkdir(parents=True)
            result = TaskRunResult(
                task_id=task_id,
                run_id=run_id,
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            (run_dir / "result.json").write_text(result.model_dump_json())

            index_entry = RunIndexEntry(
                task_id=task_id,
                run_id=run_id,
                timestamp=datetime.now(),
                benign_score=1.0,
                attack_score=None,
                exception_type=None,
                path=Path(f"{task_id}/{run_id}"),
            )
            with open(job.job_dir / "index.jsonl", "a") as f:
                f.write(index_entry.model_dump_json() + "\n")

        # task1 has 2 runs (at limit), task2 has 1 run (needs more), task3 has 0 runs
        result = job.filter_tasks_needing_runs(["task1", "task2", "task3"])
        assert set(result) == {"task2", "task3"}

    def test_filter_warns_when_task_has_more_runs_than_expected(
        self, experiment_config: ExperimentConfig, tmp_path: Path
    ):
        """Test that a warning is emitted when a task has more runs than n_runs_per_task."""
        job = Job.create(
            experiment_config=experiment_config,
            execution_mode="benign",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
            n_runs_per_task=1,
        )

        # Create 3 runs for task1 (more than n_runs_per_task=1)
        for run_id in ["aaa11111", "bbb22222", "ccc33333"]:
            run_dir = job.job_dir / "task1" / run_id
            run_dir.mkdir(parents=True)
            result = TaskRunResult(
                task_id="task1",
                run_id=run_id,
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            (run_dir / "result.json").write_text(result.model_dump_json())

            index_entry = RunIndexEntry(
                task_id="task1",
                run_id=run_id,
                timestamp=datetime.now(),
                benign_score=1.0,
                attack_score=None,
                exception_type=None,
                path=Path(f"task1/{run_id}"),
            )
            with open(job.job_dir / "index.jsonl", "a") as f:
                f.write(index_entry.model_dump_json() + "\n")

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            job.filter_tasks_needing_runs(["task1"])

            assert len(w) == 1
            assert "task1" in str(w[0].message)
            assert "3 runs" in str(w[0].message)
            assert "n_runs_per_task=1" in str(w[0].message)


class TestJobToExperimentConfig:
    """Tests for Job.to_experiment_config method."""

    def test_roundtrip_preserves_config(
        self, experiment_config_with_attack: ExperimentConfig, tmp_path: Path
    ):
        """Test that JobConfig can be converted back to ExperimentConfig."""
        job = Job.create(
            experiment_config=experiment_config_with_attack,
            execution_mode="attack",
            jobs_dir=tmp_path,
            job_name="test_job",
            agent_name="test",
        )

        converted = job.to_experiment_config()
        assert converted.agent.type == experiment_config_with_attack.agent.type
        assert converted.dataset.type == experiment_config_with_attack.dataset.type
        assert converted.attack is not None
        assert converted.attack.type == "template_string"


class TestApplyResumeOverrides:
    """Tests for _apply_resume_overrides function."""

    def test_applies_execution_and_telemetry_overrides(self, tmp_path: Path):
        """Test applying valid execution overrides."""
        config_path = tmp_path / CONFIG_FILENAME
        job_config = JobConfig(
            job_name="test",
            execution_mode="benign",
            created_at=datetime.now(),
            dataset=DatasetConfig(type="test", config={}),
            agent=AgentConfig(type="plain", config={}),
            attack=None,
            execution=ExecutionConfig(concurrency=1),
            telemetry=TelemetryConfig(trace_console=False),
            output=OutputConfig(jobs_dir=Path("jobs")),
        )
        _save_config_yaml(config_path, job_config)

        updated = _apply_resume_overrides(
            job_config,
            ["execution.concurrency=8", "telemetry.trace_console=true"],
            config_path,
        )

        assert updated.execution.concurrency == 8
        assert updated.telemetry.trace_console is True

    def test_rejects_hydra_add_prefix(self, tmp_path: Path):
        """Test that Hydra + prefix is rejected on resume."""
        config_path = tmp_path / CONFIG_FILENAME
        job_config = JobConfig(
            job_name="test",
            execution_mode="benign",
            created_at=datetime.now(),
            dataset=DatasetConfig(type="test", config={}),
            agent=AgentConfig(type="plain", config={}),
            attack=None,
            execution=ExecutionConfig(concurrency=1),
            telemetry=TelemetryConfig(),
            output=OutputConfig(jobs_dir=Path("jobs")),
        )
        _save_config_yaml(config_path, job_config)

        with pytest.raises(JobConfigMismatchError, match="Cannot add new keys"):
            _apply_resume_overrides(job_config, ["+execution.concurrency=4"], config_path)

    def test_rejects_hydra_delete_prefix(self, tmp_path: Path):
        """Test that Hydra ~ prefix is rejected on resume."""
        config_path = tmp_path / CONFIG_FILENAME
        job_config = JobConfig(
            job_name="test",
            execution_mode="benign",
            created_at=datetime.now(),
            dataset=DatasetConfig(type="test", config={}),
            agent=AgentConfig(type="plain", config={}),
            attack=None,
            execution=ExecutionConfig(concurrency=1),
            telemetry=TelemetryConfig(),
            output=OutputConfig(jobs_dir=Path("jobs")),
        )
        _save_config_yaml(config_path, job_config)

        with pytest.raises(JobConfigMismatchError, match="Cannot delete keys"):
            _apply_resume_overrides(job_config, ["~execution.concurrency"], config_path)

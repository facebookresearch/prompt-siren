# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Persistence layer for job-based task execution results."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeVar

import yaml
from filelock import FileLock
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from .models import (
    CONFIG_FILENAME,
    ExceptionInfo,
    INDEX_FILENAME,
    JobConfig,
    JobResult,
    JobStats,
    RESULT_FILENAME,
    RunIndexEntry,
    TASK_EXECUTION_FILENAME,
    TASK_RESULT_FILENAME,
    TaskRunExecution,
    TaskRunResult,
)
from .naming import sanitize_for_filename

if TYPE_CHECKING:
    from logfire import LogfireSpan

    from ..tasks import EvaluationResult, Task, TaskCouple
    from ..types import InjectionAttack

EnvStateT = TypeVar("EnvStateT")
InjectionAttackT = TypeVar("InjectionAttackT", bound="InjectionAttack")


class JobPersistence:
    """Manages persistence for a single job's task executions.

    Directory structure:
        job_dir/
            config.yaml           # Job configuration
            result.json           # Aggregated job results
            index.jsonl           # Index of all task runs
            <task_id>/
                run_001/
                    result.json      # Task run result
                    execution.json   # Full execution data
                run_002/
                    ...
    """

    def __init__(self, job_dir: Path, job_config: JobConfig):
        """Initialize JobPersistence.

        Args:
            job_dir: Directory for this job's data
            job_config: Configuration for this job
        """
        self.job_dir = job_dir
        self.job_config = job_config

    @classmethod
    def create(cls, job_dir: Path, job_config: JobConfig) -> JobPersistence:
        """Create a new JobPersistence and initialize the job directory.

        Args:
            job_dir: Directory for this job's data
            job_config: Configuration for this job

        Returns:
            JobPersistence instance with initialized directory
        """
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save config.yaml if it doesn't exist
        config_path = job_dir / CONFIG_FILENAME
        if not config_path.exists():
            _save_config_yaml(config_path, job_config)

        # Initialize result.json if it doesn't exist
        result_path = job_dir / RESULT_FILENAME
        if not result_path.exists():
            initial_result = JobResult(
                job_name=job_config.job_name,
                started_at=datetime.now(),
                stats=JobStats(
                    n_total_tasks=0,
                    n_runs_per_task=job_config.n_runs_per_task,
                ),
            )
            with open(result_path, "w") as f:
                f.write(initial_result.model_dump_json(indent=2))

        return cls(job_dir, job_config)

    @classmethod
    def load(cls, job_dir: Path) -> JobPersistence:
        """Load an existing JobPersistence from a job directory.

        Args:
            job_dir: Directory containing the job data

        Returns:
            JobPersistence instance

        Raises:
            FileNotFoundError: If config.yaml doesn't exist
        """
        config_path = job_dir / CONFIG_FILENAME
        if not config_path.exists():
            msg = f"Job config not found: {config_path}"
            raise FileNotFoundError(msg)

        job_config = _load_config_yaml(config_path)
        return cls(job_dir, job_config)

    def get_task_run_dir(self, task_id: str, run_index: int) -> Path:
        """Get the directory path for a specific task run.

        Args:
            task_id: Task identifier
            run_index: Run index (1-based for pass@k)

        Returns:
            Path to the task run directory
        """
        task_id_safe = sanitize_for_filename(task_id)
        return self.job_dir / task_id_safe / f"run_{run_index:03d}"

    def get_completed_runs(self, task_id: str) -> list[int]:
        """Get list of completed run indices for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of run indices that have result.json files
        """
        task_id_safe = sanitize_for_filename(task_id)
        task_dir = self.job_dir / task_id_safe

        if not task_dir.exists():
            return []

        completed = []
        for run_dir in task_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                result_path = run_dir / TASK_RESULT_FILENAME
                if result_path.exists():
                    # Extract run index from directory name
                    try:
                        run_index = int(run_dir.name.split("_")[1])
                        completed.append(run_index)
                    except (IndexError, ValueError):
                        continue
        return sorted(completed)

    def get_run_status(self, task_id: str, run_index: int) -> tuple[str, TaskRunResult | None]:
        """Get the status of a specific task run.

        Args:
            task_id: Task identifier
            run_index: Run index

        Returns:
            Tuple of (status, result):
            - ("completed", result): Run completed successfully
            - ("failed", result): Run failed with exception
            - ("incomplete", None): Run directory exists but no result
            - ("pending", None): Run directory doesn't exist
        """
        run_dir = self.get_task_run_dir(task_id, run_index)

        if not run_dir.exists():
            return ("pending", None)

        result_path = run_dir / TASK_RESULT_FILENAME
        if not result_path.exists():
            return ("incomplete", None)

        result = TaskRunResult.model_validate_json(result_path.read_text())
        if result.exception_info is not None:
            return ("failed", result)
        return ("completed", result)

    def save_task_run(
        self,
        task: Task[EnvStateT],
        run_index: int,
        evaluation: EvaluationResult,
        messages: list[ModelMessage],
        usage: RunUsage,
        task_span: LogfireSpan,
        started_at: datetime,
        exception: BaseException | None = None,
        generated_attacks: dict[str, InjectionAttackT] | None = None,
        attack_score: float | None = None,
    ) -> Path:
        """Save a single task run result and execution data.

        Args:
            task: Task that was executed
            run_index: Run index (1-based)
            evaluation: Task evaluation results
            messages: Conversation messages
            usage: Token usage
            task_span: Logfire span for trace context
            started_at: Timestamp when task execution started
            exception: Exception if the run failed
            generated_attacks: Generated attack vectors (for attack mode)
            attack_score: Attack score (for attack mode)

        Returns:
            Path to the run directory
        """
        run_dir = self.get_task_run_dir(task.id, run_index)
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        execution_id = str(uuid.uuid4())[:8]

        # Calculate benign score
        benign_score = (
            sum(evaluation.results.values()) / len(evaluation.results)
            if evaluation.results
            else 0.0
        )

        # Create exception info if needed
        exception_info = ExceptionInfo.from_exception(exception) if exception else None

        # Get trace context
        span_context = task_span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context else None
        span_id = format(span_context.span_id, "016x") if span_context else None

        # Save result.json (lightweight)
        result = TaskRunResult(
            task_id=task.id,
            run_index=run_index,
            started_at=started_at,
            finished_at=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_info=exception_info,
        )
        result_path = run_dir / TASK_RESULT_FILENAME
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save execution.json (heavy)
        # Convert attacks to serializable format
        attacks_dict: dict[str, Any] | None = None
        if generated_attacks:
            from ..types import InjectionAttacksDictTypeAdapter

            attacks_dict = InjectionAttacksDictTypeAdapter.dump_python(generated_attacks)

        execution = TaskRunExecution(
            task_id=task.id,
            run_index=run_index,
            execution_id=execution_id,
            timestamp=now,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages,
            usage=usage,
            attacks=attacks_dict,
        )
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        with open(execution_path, "w") as f:
            f.write(execution.model_dump_json(indent=2))

        # Update index
        self._append_to_index(
            task_id=task.id,
            run_index=run_index,
            timestamp=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_info.exception_type if exception_info else None,
            path=run_dir.relative_to(self.job_dir),
        )

        return run_dir

    def save_couple_run(
        self,
        couple: TaskCouple[EnvStateT],
        run_index: int,
        benign_eval: EvaluationResult,
        malicious_eval: EvaluationResult,
        messages: list[ModelMessage],
        usage: RunUsage,
        task_span: LogfireSpan,
        started_at: datetime,
        exception: BaseException | None = None,
        generated_attacks: dict[str, InjectionAttackT] | None = None,
    ) -> Path:
        """Save a task couple run result and execution data.

        Args:
            couple: Task couple that was executed
            run_index: Run index (1-based)
            benign_eval: Benign task evaluation results
            malicious_eval: Malicious task evaluation results
            messages: Conversation messages
            usage: Token usage
            task_span: Logfire span for trace context
            started_at: Timestamp when task execution started
            exception: Exception if the run failed
            generated_attacks: Generated attack vectors

        Returns:
            Path to the run directory
        """
        run_dir = self.get_task_run_dir(couple.id, run_index)
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        execution_id = str(uuid.uuid4())[:8]

        # Calculate scores
        benign_score = (
            sum(benign_eval.results.values()) / len(benign_eval.results)
            if benign_eval.results
            else 0.0
        )
        attack_score = (
            sum(malicious_eval.results.values()) / len(malicious_eval.results)
            if malicious_eval.results
            else 0.0
        )

        # Create exception info if needed
        exception_info = ExceptionInfo.from_exception(exception) if exception else None

        # Get trace context
        span_context = task_span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context else None
        span_id = format(span_context.span_id, "016x") if span_context else None

        # Save result.json (lightweight)
        result = TaskRunResult(
            task_id=couple.id,
            run_index=run_index,
            started_at=started_at,
            finished_at=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_info=exception_info,
        )
        result_path = run_dir / TASK_RESULT_FILENAME
        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        # Save execution.json (heavy)
        attacks_dict: dict[str, Any] | None = None
        if generated_attacks:
            from ..types import InjectionAttacksDictTypeAdapter

            attacks_dict = InjectionAttacksDictTypeAdapter.dump_python(generated_attacks)

        execution = TaskRunExecution(
            task_id=couple.id,
            run_index=run_index,
            execution_id=execution_id,
            timestamp=now,
            trace_id=trace_id,
            span_id=span_id,
            messages=messages,
            usage=usage,
            attacks=attacks_dict,
        )
        execution_path = run_dir / TASK_EXECUTION_FILENAME
        with open(execution_path, "w") as f:
            f.write(execution.model_dump_json(indent=2))

        # Update index
        self._append_to_index(
            task_id=couple.id,
            run_index=run_index,
            timestamp=now,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_info.exception_type if exception_info else None,
            path=run_dir.relative_to(self.job_dir),
        )

        return run_dir

    def _append_to_index(
        self,
        task_id: str,
        run_index: int,
        timestamp: datetime,
        benign_score: float | None,
        attack_score: float | None,
        exception_type: str | None,
        path: Path,
    ) -> None:
        """Atomically append to job index.jsonl with file locking."""
        index_path = self.job_dir / INDEX_FILENAME
        lock_path = self.job_dir / f"{INDEX_FILENAME}.lock"

        entry = RunIndexEntry(
            task_id=task_id,
            run_index=run_index,
            timestamp=timestamp,
            benign_score=benign_score,
            attack_score=attack_score,
            exception_type=exception_type,
            path=path,
        )

        with FileLock(lock_path):
            with open(index_path, "a") as f:
                f.write(entry.model_dump_json() + "\n")

    def update_job_result(self, stats: JobStats, is_complete: bool = False) -> None:
        """Update the job result with current statistics.

        Args:
            stats: Current job statistics
            is_complete: Whether the job is complete
        """
        result_path = self.job_dir / RESULT_FILENAME

        # Load existing result to preserve original start time
        if result_path.exists():
            existing_result = JobResult.model_validate_json(result_path.read_text())
            started_at = existing_result.started_at
        else:
            started_at = datetime.now()

        result = JobResult(
            job_name=self.job_config.job_name,
            started_at=started_at,
            finished_at=datetime.now() if is_complete else None,
            is_complete=is_complete,
            stats=stats,
        )

        with open(result_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

    def load_index(self) -> list[RunIndexEntry]:
        """Load all entries from the job index.

        Returns:
            List of index entries
        """
        index_path = self.job_dir / INDEX_FILENAME
        if not index_path.exists():
            return []

        entries = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(RunIndexEntry.model_validate_json(line))
        return entries

    def delete_run(self, task_id: str, run_index: int) -> bool:
        """Delete a task run directory.

        Used when retrying failed runs.

        Args:
            task_id: Task identifier
            run_index: Run index

        Returns:
            True if the directory was deleted, False if it didn't exist
        """
        import shutil

        run_dir = self.get_task_run_dir(task_id, run_index)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False

    def remove_index_entries_by_paths(self, paths_to_remove: set[Path]) -> None:
        """Remove entries from the index by their paths.

        Rewrites the entire index file excluding entries with paths in the given set.

        Args:
            paths_to_remove: Set of paths (relative to job_dir) to remove from index
        """
        index_path = self.job_dir / INDEX_FILENAME
        if not index_path.exists():
            return

        lock_path = self.job_dir / f"{INDEX_FILENAME}.lock"

        with FileLock(lock_path):
            # Load all existing entries
            entries = []
            with open(index_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(RunIndexEntry.model_validate_json(line))

            # Filter out entries with paths in paths_to_remove
            filtered_entries = [entry for entry in entries if entry.path not in paths_to_remove]

            # Rewrite the index file
            with open(index_path, "w") as f:
                for entry in filtered_entries:
                    f.write(entry.model_dump_json() + "\n")


def _save_config_yaml(config_path: Path, job_config: JobConfig) -> None:
    """Save job config to YAML file."""
    config_dict = job_config.model_dump(mode="json")

    header = f"# Job: {job_config.job_name}\n"
    header += f"# Created: {job_config.created_at.isoformat()}\n"
    header += f"# Mode: {job_config.execution_mode}\n\n"

    with open(config_path, "w") as f:
        f.write(header)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _load_config_yaml(config_path: Path) -> JobConfig:
    """Load job config from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return JobConfig.model_validate(config_dict)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Job class for managing experiment runs with resume capability."""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from .models import (
    CONFIG_FILENAME,
    JobConfig,
    JobStats,
)
from .naming import generate_job_name
from .persistence import _load_config_yaml, JobPersistence

if TYPE_CHECKING:
    from ..config.experiment_config import ExperimentConfig
    from ..types import ExecutionMode

# Fields that can be overridden on resume
RESUMABLE_OVERRIDE_PREFIXES = ("execution.", "telemetry.", "output.", "usage_limits.")

# Fields that cannot be changed on resume
IMMUTABLE_PREFIXES = ("dataset.", "agent.", "attack.")


class JobConfigMismatchError(Exception):
    """Raised when resume overrides attempt to modify immutable config."""


class Job:
    """Manages a single experiment job with persistence and resume capability.

    A job represents a complete experiment run that can be:
    - Created fresh from an ExperimentConfig
    - Resumed from an existing job directory
    - Retried for specific failed tasks
    """

    def __init__(
        self,
        job_dir: Path,
        job_config: JobConfig,
        is_resuming: bool = False,
    ):
        """Initialize a Job.

        Use Job.create() or Job.resume() instead of calling this directly.
        """
        self.job_dir = job_dir
        self.job_config = job_config
        self.is_resuming = is_resuming
        self._persistence: JobPersistence | None = None

    @classmethod
    def create(
        cls,
        experiment_config: ExperimentConfig,
        execution_mode: ExecutionMode,
        jobs_dir: Path,
        job_name: str | None = None,
        agent_name: str | None = None,
        n_runs_per_task: int = 1,
    ) -> Job:
        """Create a new job from experiment configuration.

        Args:
            experiment_config: Validated experiment configuration
            execution_mode: "benign" or "attack"
            jobs_dir: Base directory for all jobs
            job_name: Custom job name (auto-generated if None)
            agent_name: Agent name for job naming (required if job_name is None)
            n_runs_per_task: Number of runs per task for pass@k

        Returns:
            New Job instance
        """
        # Generate job name if not provided
        if job_name is None:
            if agent_name is None:
                msg = "agent_name is required when job_name is not provided"
                raise ValueError(msg)
            job_name = generate_job_name(
                dataset_type=experiment_config.dataset.type,
                agent_name=agent_name,
                attack_type=experiment_config.attack.type if experiment_config.attack else None,
            )

        job_dir = jobs_dir / job_name

        # Check if job already exists
        if job_dir.exists():
            config_path = job_dir / CONFIG_FILENAME
            if config_path.exists():
                raise FileExistsError(
                    f"Job directory already exists: {job_dir}\n"
                    "Use 'jobs resume' to continue an existing job, or choose a different job name."
                )

        # Create job config from experiment config
        # Exclude jobs_dir from output - it's not needed for resume since job_dir is passed directly
        output_dict = experiment_config.output.model_dump()
        output_dict.pop("jobs_dir", None)

        job_config = JobConfig(
            job_name=job_name,
            execution_mode=execution_mode,
            created_at=datetime.now(),
            n_runs_per_task=n_runs_per_task,
            dataset=experiment_config.dataset.model_dump(),
            agent=experiment_config.agent.model_dump(),
            attack=experiment_config.attack.model_dump() if experiment_config.attack else None,
            execution=experiment_config.execution.model_dump(),
            telemetry=experiment_config.telemetry.model_dump(),
            output=output_dict,
            task_ids=experiment_config.task_ids,
            usage_limits=asdict(experiment_config.usage_limits)
            if experiment_config.usage_limits
            else None,
        )

        # Create job directory and persistence
        job_dir.mkdir(parents=True, exist_ok=True)
        JobPersistence.create(job_dir, job_config)

        return cls(job_dir, job_config, is_resuming=False)

    @classmethod
    def resume(
        cls,
        job_dir: Path,
        overrides: list[str] | None = None,
        retry_on_errors: list[str] | None = None,
    ) -> Job:
        """Resume an existing job from its directory.

        Args:
            job_dir: Path to the job directory
            overrides: Hydra-style overrides for execution-related fields
            retry_on_errors: Error types to retry (delete runs with these exceptions)

        Returns:
            Job instance configured for resumption

        Raises:
            FileNotFoundError: If job directory or config doesn't exist
            JobConfigMismatchError: If overrides attempt to modify immutable config
        """
        config_path = job_dir / CONFIG_FILENAME
        if not config_path.exists():
            msg = f"Job config not found: {config_path}"
            raise FileNotFoundError(msg)

        # Load existing config
        job_config = _load_config_yaml(config_path)

        # Validate and apply overrides
        if overrides:
            job_config = _apply_resume_overrides(job_config, overrides, config_path)

        # Create job instance
        job = cls(job_dir, job_config, is_resuming=True)

        # Handle retry logic
        job._cleanup_for_retry(retry_on_errors)

        return job

    @property
    def persistence(self) -> JobPersistence:
        """Get the persistence instance for this job."""
        if self._persistence is None:
            self._persistence = JobPersistence(self.job_dir, self.job_config)
        return self._persistence

    def get_remaining_runs(self, task_ids: list[str]) -> list[tuple[str, int]]:
        """Get list of (task_id, run_index) pairs that need to be executed.

        Args:
            task_ids: List of all task IDs to run

        Returns:
            List of (task_id, run_index) tuples for runs that haven't completed
        """
        remaining = []
        for task_id in task_ids:
            for run_index in range(1, self.job_config.n_runs_per_task + 1):
                status, _ = self.persistence.get_run_status(task_id, run_index)
                if status in ("pending", "incomplete"):
                    remaining.append((task_id, run_index))
        return remaining

    def get_job_stats(self, task_ids: list[str]) -> JobStats:
        """Calculate current job statistics.

        Args:
            task_ids: List of all task IDs

        Returns:
            JobStats with current counts
        """
        n_total_tasks = len(task_ids)
        n_runs_per_task = self.job_config.n_runs_per_task
        total_runs = n_total_tasks * n_runs_per_task

        completed_runs = 0
        failed_runs = 0
        exception_counts: dict[str, int] = defaultdict(int)

        scores_benign: list[float] = []
        scores_attack: list[float] = []

        for task_id in task_ids:
            for run_index in range(1, n_runs_per_task + 1):
                status, result = self.persistence.get_run_status(task_id, run_index)
                if status == "completed":
                    completed_runs += 1
                    if result and result.benign_score is not None:
                        scores_benign.append(result.benign_score)
                    if result and result.attack_score is not None:
                        scores_attack.append(result.attack_score)
                elif status == "failed":
                    failed_runs += 1
                    if result and result.exception_info:
                        exception_counts[result.exception_info.exception_type] += 1

        remaining_runs = total_runs - completed_runs - failed_runs

        return JobStats(
            n_total_tasks=n_total_tasks,
            n_runs_per_task=n_runs_per_task,
            n_completed_runs=completed_runs,
            n_failed_runs=failed_runs,
            n_remaining_runs=remaining_runs,
            avg_benign_score=sum(scores_benign) / len(scores_benign) if scores_benign else None,
            avg_attack_score=sum(scores_attack) / len(scores_attack) if scores_attack else None,
            exception_counts=dict(exception_counts),
        )

    def _cleanup_for_retry(
        self,
        retry_on_errors: list[str] | None,
    ) -> None:
        """Clean up run directories for retry based on error filtering.

        Args:
            retry_on_errors: Error types to retry (delete runs with these exceptions)
        """
        if not retry_on_errors:
            return

        retry_error_set = set(retry_on_errors)

        # Track paths that are deleted for index cleanup
        deleted_paths: set[Path] = set()

        # Load index to find failed runs
        index_entries = self.persistence.load_index()

        for entry in index_entries:
            if entry.exception_type is not None and entry.exception_type in retry_error_set:
                run_dir = self.job_dir / entry.path
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    deleted_paths.add(entry.path)

        # Also clean up incomplete runs (directories without result.json)
        for task_dir in self.job_dir.iterdir():
            if not task_dir.is_dir():
                continue
            # Skip lock directories (unlikely but possible)
            if task_dir.name.endswith(".lock"):
                continue

            for run_dir in task_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    result_path = run_dir / "result.json"
                    if not result_path.exists():
                        # Incomplete run, delete it
                        shutil.rmtree(run_dir)
                        deleted_paths.add(run_dir.relative_to(self.job_dir))

        # Clean up index entries for all deleted runs
        if deleted_paths:
            self.persistence.remove_index_entries_by_paths(deleted_paths)

    def to_experiment_config(self) -> ExperimentConfig:
        """Convert JobConfig back to ExperimentConfig for execution.

        Returns:
            ExperimentConfig instance
        """
        from pydantic_ai.usage import UsageLimits

        from ..config.experiment_config import (
            AgentConfig,
            AttackConfig,
            DatasetConfig,
            ExecutionConfig,
            ExperimentConfig,
            OutputConfig,
            TelemetryConfig,
        )

        # Provide default for jobs_dir since it's not stored in JobConfig
        # (job_dir is passed directly on resume, so jobs_dir is not needed)
        output_dict = dict(self.job_config.output)
        output_dict.setdefault("jobs_dir", Path("jobs"))

        return ExperimentConfig(
            name=self.job_config.job_name,
            agent=AgentConfig.model_validate(self.job_config.agent),
            dataset=DatasetConfig.model_validate(self.job_config.dataset),
            attack=AttackConfig.model_validate(self.job_config.attack)
            if self.job_config.attack
            else None,
            execution=ExecutionConfig.model_validate(self.job_config.execution),
            telemetry=TelemetryConfig.model_validate(self.job_config.telemetry),
            output=OutputConfig.model_validate(output_dict),
            task_ids=self.job_config.task_ids,
            usage_limits=UsageLimits(**self.job_config.usage_limits)
            if self.job_config.usage_limits
            else None,
        )


def _apply_resume_overrides(
    job_config: JobConfig,
    overrides: list[str],
    config_path: Path,
) -> JobConfig:
    """Apply Hydra-style overrides to a job config on resume.

    Args:
        job_config: Original job configuration
        overrides: List of Hydra-style overrides (e.g., "execution.concurrency=8")
        config_path: Path to config file (for updating)

    Returns:
        Updated JobConfig

    Raises:
        JobConfigMismatchError: If overrides attempt to modify immutable fields
    """
    # Validate overrides don't touch immutable fields
    for override in overrides:
        key = override.split("=")[0].lstrip("+~")
        for prefix in IMMUTABLE_PREFIXES:
            if key.startswith(prefix):
                raise JobConfigMismatchError(
                    f"Cannot modify {prefix.rstrip('.')} configuration on resume. "
                    f"Attempted to change: {key}\n"
                    f"Only these prefixes can be modified: {', '.join(RESUMABLE_OVERRIDE_PREFIXES)}"
                )

        # Warn if override doesn't match any resumable prefix
        is_valid = any(key.startswith(prefix) for prefix in RESUMABLE_OVERRIDE_PREFIXES)
        if not is_valid:
            raise JobConfigMismatchError(
                f"Unknown override key: {key}\n"
                f"Only these prefixes can be modified on resume: {', '.join(RESUMABLE_OVERRIDE_PREFIXES)}"
            )

    # Convert job config to OmegaConf for override handling
    config_dict = job_config.model_dump(mode="json")
    cfg = OmegaConf.create(config_dict)

    # Apply overrides using OmegaConf
    for override in overrides:
        key, value = override.split("=", 1)
        key = key.lstrip("+~")  # Remove Hydra special prefixes
        OmegaConf.update(cfg, key, _parse_value(value))

    # Convert back to JobConfig
    updated_dict = OmegaConf.to_container(cfg, resolve=True)
    updated_config = JobConfig.model_validate(updated_dict)

    # Save updated config back to file
    from .persistence import _save_config_yaml

    _save_config_yaml(config_path, updated_config)

    return updated_config


def _parse_value(value: str) -> str | int | float | bool:
    """Parse a string value to the appropriate type.

    Handles:
    - true/false -> bool
    - integers
    - floats
    - strings (with optional quotes)
    """
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # String (strip quotes if present)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


def list_jobs(jobs_dir: Path) -> list[tuple[Path, JobConfig | None]]:
    """List all jobs in a jobs directory.

    Args:
        jobs_dir: Base directory for jobs

    Returns:
        List of (job_dir, job_config) tuples. job_config is None if config couldn't be loaded.
    """
    if not jobs_dir.exists():
        return []

    jobs = []
    for job_path in jobs_dir.iterdir():
        if not job_path.is_dir():
            continue

        config_path = job_path / CONFIG_FILENAME
        if config_path.exists():
            try:
                job_config = _load_config_yaml(config_path)
                jobs.append((job_path, job_config))
            except Exception:
                jobs.append((job_path, None))

    return sorted(jobs, key=lambda x: x[0].name)

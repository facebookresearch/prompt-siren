# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Job class for managing experiment runs with resume capability."""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from ..config.experiment_config import ExperimentConfig
from ..types import ExecutionMode
from .models import (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    INDEX_LOCK_FILENAME,
    JobConfig,
)
from .naming import generate_job_name
from .persistence import _save_config_yaml, JobPersistence, load_config_yaml

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
    ):
        """Initialize a Job.

        Use Job.create() or Job.resume() instead of calling this directly.
        """
        self.job_dir = job_dir
        self.job_config = job_config
        self.persistence = JobPersistence(self.job_dir, self.job_config)

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

        # Create job config by extending experiment config with job-specific fields
        job_config = JobConfig(
            job_name=job_name,
            execution_mode=execution_mode,
            created_at=datetime.now(),
            n_runs_per_task=n_runs_per_task,
            **experiment_config.model_dump(),
        )

        # Create job directory and persistence
        job_dir.mkdir(parents=True, exist_ok=True)
        JobPersistence.create(job_dir, job_config)

        return cls(job_dir, job_config)

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
            raise FileNotFoundError(f"Job config not found: {config_path}")

        # Load existing config
        job_config = load_config_yaml(config_path)

        # Validate and apply overrides
        if overrides:
            job_config = _apply_resume_overrides(job_config, overrides, config_path)

        # Create job instance
        job = cls(job_dir, job_config)

        # Handle retry logic
        job._cleanup_for_retry(retry_on_errors)

        return job

    def _cleanup_for_retry(
        self,
        retry_on_errors: Sequence[str] | None,
    ) -> None:
        """Clean up run directories for retry based on error filtering.

        Args:
            retry_on_errors: Error types to retry (delete runs with these exceptions)
        """
        if not retry_on_errors:
            return

        retry_error_set = {error.lower() for error in retry_on_errors}

        # Track paths that are deleted for index cleanup
        deleted_paths: set[Path] = set()

        # Load index to find failed runs
        index_entries = self.persistence.load_index()

        for entry in index_entries:
            if entry.exception_type is not None and entry.exception_type.lower() in retry_error_set:
                run_dir = self.job_dir / entry.path
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    deleted_paths.add(entry.path)

        special_files = {CONFIG_FILENAME, INDEX_FILENAME, INDEX_LOCK_FILENAME}
        for task_dir in self.job_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if task_dir.name in special_files or task_dir.name.endswith(".lock"):
                continue

            for run_dir in task_dir.iterdir():
                if run_dir.is_dir():
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

        Since JobConfig extends ExperimentConfig, this extracts just the
        ExperimentConfig fields.

        Returns:
            ExperimentConfig instance
        """
        # Extract only ExperimentConfig fields from JobConfig
        return ExperimentConfig.model_validate(
            self.job_config.model_dump(include=set(ExperimentConfig.model_fields.keys()))
        )

    def filter_tasks_needing_runs(self, task_ids: list[str]) -> list[str]:
        """Filter task IDs to only those needing more runs.

        Args:
            task_ids: List of task IDs to filter

        Returns:
            Task IDs that have fewer runs than n_runs_per_task
        """
        run_counts = self.persistence.get_run_counts()
        n_required = self.job_config.n_runs_per_task

        # Warn about tasks with more runs than expected
        for tid in task_ids:
            count = run_counts.get(tid, 0)
            if count > n_required:
                warnings.warn(
                    f"Task '{tid}' has {count} runs but n_runs_per_task={n_required}",
                    stacklevel=2,
                )

        return [tid for tid in task_ids if run_counts.get(tid, 0) < n_required]


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
    # Validate overrides
    for override in overrides:
        # Reject Hydra add/delete prefixes - only updates allowed on resume
        if override.startswith("+"):
            raise JobConfigMismatchError(
                f"Cannot add new keys on resume: {override}\n"
                "Only updates to existing keys are allowed."
            )
        if override.startswith("~"):
            raise JobConfigMismatchError(
                f"Cannot delete keys on resume: {override}\n"
                "Only updates to existing keys are allowed."
            )

        key = override.split("=")[0]

        # Check for immutable fields
        for prefix in IMMUTABLE_PREFIXES:
            if key.startswith(prefix):
                raise JobConfigMismatchError(
                    f"Cannot modify {prefix.rstrip('.')} configuration on resume. "
                    f"Attempted to change: {key}\n"
                    f"Only these prefixes can be modified: {', '.join(RESUMABLE_OVERRIDE_PREFIXES)}"
                )

        # Check override is in allowed prefixes
        is_valid = any(key.startswith(prefix) for prefix in RESUMABLE_OVERRIDE_PREFIXES)
        if not is_valid:
            raise JobConfigMismatchError(
                f"Unknown override key: {key}\n"
                f"Only these prefixes can be modified on resume: {', '.join(RESUMABLE_OVERRIDE_PREFIXES)}"
            )

    # Convert job config to OmegaConf and apply overrides
    config_dict = job_config.model_dump(mode="json")
    cfg = OmegaConf.create(config_dict)

    # Use OmegaConf's built-in parsing for overrides
    override_cfg = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg, override_cfg)

    # Convert back to JobConfig
    updated_dict = OmegaConf.to_container(merged, resolve=True)
    updated_config = JobConfig.model_validate(updated_dict)

    # Save updated config back to file
    _save_config_yaml(config_path, updated_config)

    return updated_config

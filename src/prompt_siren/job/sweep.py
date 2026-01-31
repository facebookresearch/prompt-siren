# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Sweep management for tracking and resuming Hydra multirun sweeps."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from pydantic import BaseModel, Field

from .models import CONFIG_FILENAME, INDEX_FILENAME

# Directory name for sweep metadata
SWEEPS_DIR = ".sweeps"
SWEEP_LOCK_SUFFIX = ".lock"


class SweepMetadata(BaseModel):
    """Metadata for a Hydra sweep (multirun).

    Tracks all job directories created as part of a single sweep,
    enabling batch resume operations.
    """

    sweep_id: str = Field(description="Sweep identifier (typically timestamp)")
    created_at: datetime = Field(default_factory=datetime.now)
    job_names: list[str] = Field(
        default_factory=list,
        description="Names of jobs in this sweep (relative to jobs_dir)",
    )
    hydra_sweep_dir: str | None = Field(
        default=None,
        description="Path to Hydra's sweep output directory",
    )
    launcher: str | None = Field(
        default=None,
        description="Hydra launcher used (e.g., 'submitit_slurm', 'basic')",
    )
    config_name: str | None = Field(
        default=None,
        description="Original config name used for the sweep",
    )


class SweepRegistry:
    """Registry for tracking sweeps and their associated jobs.

    Maintains a .sweeps/ directory in the jobs_dir with one JSON file
    per sweep, enabling efficient lookup and batch operations.
    """

    def __init__(self, jobs_dir: Path):
        """Initialize SweepRegistry.

        Args:
            jobs_dir: Base directory containing all jobs
        """
        self.jobs_dir = Path(jobs_dir)
        self.sweeps_dir = self.jobs_dir / SWEEPS_DIR

    def _get_sweep_path(self, sweep_id: str) -> Path:
        """Get the path to a sweep metadata file."""
        return self.sweeps_dir / f"{sweep_id}.json"

    def _get_lock_path(self, sweep_id: str) -> Path:
        """Get the path to a sweep lock file."""
        return self.sweeps_dir / f"{sweep_id}{SWEEP_LOCK_SUFFIX}"

    def register_job(
        self,
        sweep_id: str,
        job_name: str,
        hydra_sweep_dir: str | None = None,
        launcher: str | None = None,
        config_name: str | None = None,
    ) -> None:
        """Register a job as part of a sweep.

        Creates the sweep metadata file if it doesn't exist, or appends
        the job to an existing sweep. Uses file locking for concurrent safety.

        Args:
            sweep_id: Sweep identifier (typically timestamp from Hydra)
            job_name: Name of the job directory
            hydra_sweep_dir: Path to Hydra's sweep output directory
            launcher: Hydra launcher used
            config_name: Original config name
        """
        self.sweeps_dir.mkdir(parents=True, exist_ok=True)

        sweep_path = self._get_sweep_path(sweep_id)
        lock_path = self._get_lock_path(sweep_id)

        with FileLock(lock_path):
            if sweep_path.exists():
                # Load existing sweep and append job
                with open(sweep_path) as f:
                    sweep = SweepMetadata.model_validate_json(f.read())
                if job_name not in sweep.job_names:
                    sweep.job_names.append(job_name)
            else:
                # Create new sweep
                sweep = SweepMetadata(
                    sweep_id=sweep_id,
                    job_names=[job_name],
                    hydra_sweep_dir=hydra_sweep_dir,
                    launcher=launcher,
                    config_name=config_name,
                )

            # Save sweep metadata
            with open(sweep_path, "w") as f:
                f.write(sweep.model_dump_json(indent=2))

    def get_sweep(self, sweep_id: str) -> SweepMetadata | None:
        """Get sweep metadata by ID.

        Args:
            sweep_id: Sweep identifier

        Returns:
            SweepMetadata if found, None otherwise
        """
        sweep_path = self._get_sweep_path(sweep_id)
        if not sweep_path.exists():
            return None

        with open(sweep_path) as f:
            return SweepMetadata.model_validate_json(f.read())

    def list_sweeps(self) -> list[SweepMetadata]:
        """List all registered sweeps.

        Returns:
            List of SweepMetadata for all sweeps
        """
        if not self.sweeps_dir.exists():
            return []

        sweeps = [
            sweep
            for sweep_file in self.sweeps_dir.glob("*.json")
            if (sweep := self._load_sweep_file(sweep_file)) is not None
        ]

        return sorted(sweeps, key=lambda s: s.created_at, reverse=True)

    def _load_sweep_file(self, sweep_file: Path) -> SweepMetadata | None:
        """Load a sweep metadata file, returning None on error."""
        try:
            with open(sweep_file) as f:
                return SweepMetadata.model_validate_json(f.read())
        except (json.JSONDecodeError, ValueError):
            return None

    def get_job_dirs(self, sweep_id: str) -> list[Path]:
        """Get all job directories for a sweep.

        Args:
            sweep_id: Sweep identifier

        Returns:
            List of paths to job directories
        """
        sweep = self.get_sweep(sweep_id)
        if sweep is None:
            return []

        return [self.jobs_dir / name for name in sweep.job_names if (self.jobs_dir / name).exists()]

    def find_sweeps_by_pattern(self, pattern: str) -> list[SweepMetadata]:
        """Find sweeps matching a pattern.

        Args:
            pattern: Glob pattern to match sweep IDs

        Returns:
            List of matching SweepMetadata
        """
        if not self.sweeps_dir.exists():
            return []

        sweeps = [
            sweep
            for sweep_file in self.sweeps_dir.glob(f"{pattern}.json")
            if (sweep := self._load_sweep_file(sweep_file)) is not None
        ]

        return sorted(sweeps, key=lambda s: s.created_at, reverse=True)


def find_sweep_jobs_by_pattern(jobs_dir: Path, pattern: str) -> list[Path]:
    """Find job directories matching a pattern.

    This is a fallback for when sweep metadata doesn't exist.
    Searches for job directories whose names match the given pattern.

    Args:
        jobs_dir: Base directory containing jobs
        pattern: Glob pattern to match job directory names

    Returns:
        List of matching job directory paths
    """
    jobs_dir = Path(jobs_dir)

    # If pattern doesn't contain wildcards, treat as sweep timestamp
    if "*" not in pattern and "?" not in pattern:
        # Assume it's a timestamp like "2026-01-31_17-01-04"
        pattern = f"*_{pattern}_*"

    matching_dirs = [
        path
        for path in jobs_dir.glob(pattern)
        if path.is_dir() and (path / CONFIG_FILENAME).exists()
    ]

    return sorted(matching_dirs)


def get_job_status(job_dir: Path) -> dict:
    """Get status summary for a job.

    Args:
        job_dir: Path to job directory

    Returns:
        Dictionary with status information:
        - total_tasks: Number of tasks with any runs
        - completed_tasks: Number of tasks with successful runs
        - failed_tasks: Number of tasks with only failed runs
        - has_failures: Whether any failures exist
        - failure_types: Set of exception types encountered
    """
    from .models import RunIndexEntry

    index_path = job_dir / INDEX_FILENAME
    if not index_path.exists():
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "has_failures": False,
            "failure_types": set(),
        }

    # Load index entries
    entries: list[RunIndexEntry] = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(RunIndexEntry.model_validate_json(line))

    # Group by task_id
    task_runs: dict[str, list[RunIndexEntry]] = {}
    for entry in entries:
        if entry.task_id not in task_runs:
            task_runs[entry.task_id] = []
        task_runs[entry.task_id].append(entry)

    # Analyze tasks
    completed_tasks = 0
    failed_tasks = 0
    failure_types: set[str] = set()

    for runs in task_runs.values():
        has_success = any(r.exception_type is None for r in runs)
        has_failure = any(r.exception_type is not None for r in runs)

        if has_success:
            completed_tasks += 1
        elif has_failure:
            failed_tasks += 1

        for run in runs:
            if run.exception_type:
                failure_types.add(run.exception_type)

    return {
        "total_tasks": len(task_runs),
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "has_failures": len(failure_types) > 0,
        "failure_types": failure_types,
    }

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Execution backends for running experiments locally or on SLURM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .config.experiment_config import SlurmConfig

T = TypeVar("T")


class ExecutionBackend(ABC):
    """Abstract base class for execution backends.

    Backends handle where and how experiment jobs are executed - either
    locally in the current process or submitted to a job scheduler like SLURM.
    """

    @abstractmethod
    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | Any:
        """Submit a single job for execution.

        Args:
            fn: Function to execute
            *args: Positional arguments to pass to fn
            **kwargs: Keyword arguments to pass to fn

        Returns:
            Result of fn (for local) or job handle (for SLURM)
        """
        ...

    @abstractmethod
    def submit_batch(
        self, fn: Callable[..., T], args_list: list[tuple[Any, ...]]
    ) -> list[T] | list[Any]:
        """Submit multiple jobs for execution.

        Args:
            fn: Function to execute for each set of arguments
            args_list: List of argument tuples, one per job

        Returns:
            List of results (for local) or job handles (for SLURM)
        """
        ...

    @abstractmethod
    def wait(self) -> list[Any]:
        """Wait for all submitted jobs to complete.

        Returns:
            List of results from all submitted jobs
        """
        ...

    @property
    @abstractmethod
    def is_async(self) -> bool:
        """Whether this backend runs jobs asynchronously."""
        ...


class LocalBackend(ExecutionBackend):
    """Execute jobs locally in the current process.

    Jobs are executed synchronously - submit() blocks until complete.
    """

    def __init__(self):
        self.results: list[Any] = []

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function immediately and return result."""
        result = fn(*args, **kwargs)
        self.results.append(result)
        return result

    def submit_batch(self, fn: Callable[..., T], args_list: list[tuple[Any, ...]]) -> list[T]:
        """Execute functions sequentially and return results."""
        results = []
        for args in args_list:
            result = fn(*args)
            results.append(result)
            self.results.append(result)
        return results

    def wait(self) -> list[Any]:
        """Return all collected results (already complete)."""
        return self.results

    @property
    def is_async(self) -> bool:
        return False


@dataclass
class SlurmJobInfo:
    """Information about a submitted SLURM job."""

    job_id: str
    job_dir: Path
    log_file: Path


class SlurmBackend(ExecutionBackend):
    """Execute jobs on SLURM via submitit.

    Jobs are submitted asynchronously - submit() returns immediately
    with a job handle. Use wait() to block until all jobs complete.
    """

    def __init__(
        self,
        slurm_config: SlurmConfig,
        log_dir: Path,
        wait_for_completion: bool = True,
    ):
        """Initialize SLURM backend.

        Args:
            slurm_config: SLURM configuration
            log_dir: Directory for SLURM logs
            wait_for_completion: Whether wait() should block for results
        """
        try:
            import submitit
        except ImportError as e:
            raise ImportError(
                "submitit is required for SLURM execution. Install it with: pip install submitit"
            ) from e

        self.slurm_config = slurm_config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.wait_for_completion = wait_for_completion

        # Create executor
        self.executor = submitit.AutoExecutor(folder=str(self.log_dir))
        self._configure_executor()

        self.jobs: list[submitit.Job] = []
        self.job_infos: list[SlurmJobInfo] = []

    def _configure_executor(self) -> None:
        """Configure submitit executor with SLURM parameters."""
        params = {
            "slurm_partition": self.slurm_config.partition,
            "timeout_min": self.slurm_config.time_minutes,
            "gpus_per_node": self.slurm_config.gpus_per_node,
            "cpus_per_task": self.slurm_config.cpus_per_task,
            "mem_gb": self.slurm_config.mem_gb,
        }

        if self.slurm_config.constraint:
            params["slurm_constraint"] = self.slurm_config.constraint

        # Add any additional parameters
        params.update(self.slurm_config.additional_parameters)

        self.executor.update_parameters(**params)

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> SlurmJobInfo:
        """Submit a single job to SLURM.

        Returns:
            SlurmJobInfo with job details
        """
        job = self.executor.submit(fn, *args, **kwargs)
        self.jobs.append(job)

        info = SlurmJobInfo(
            job_id=job.job_id,
            job_dir=Path(job.paths.folder),
            log_file=Path(job.paths.stdout),
        )
        self.job_infos.append(info)
        return info

    def submit_batch(
        self, fn: Callable[..., T], args_list: list[tuple[Any, ...]]
    ) -> list[SlurmJobInfo]:
        """Submit multiple jobs to SLURM as an array job.

        Returns:
            List of SlurmJobInfo for each job
        """
        if not args_list:
            return []

        # submitit's map_array expects positional args unpacked
        # Convert list of tuples to separate lists for each position
        if args_list and isinstance(args_list[0], tuple):
            transposed = list(zip(*args_list, strict=True))
            jobs = self.executor.map_array(fn, *transposed)
        else:
            jobs = self.executor.map_array(fn, args_list)

        self.jobs.extend(jobs)

        infos = []
        for job in jobs:
            info = SlurmJobInfo(
                job_id=job.job_id,
                job_dir=Path(job.paths.folder),
                log_file=Path(job.paths.stdout),
            )
            infos.append(info)
            self.job_infos.append(info)

        return infos

    def wait(self) -> list[Any]:
        """Wait for all submitted jobs to complete.

        Returns:
            List of results from all jobs (if wait_for_completion=True)
            or empty list (if wait_for_completion=False)
        """
        if not self.wait_for_completion:
            return []

        results: list[Any] = []
        for job in self.jobs:
            result = self._get_job_result(job)
            results.append(result)

        return results

    def _get_job_result(self, job: Any) -> Any:
        """Get result from a job, returning exception if it failed."""
        try:
            return job.result()
        except Exception as e:
            return e

    @property
    def is_async(self) -> bool:
        return True

    def get_job_status(self) -> dict[str, list[str]]:
        """Get status of all submitted jobs.

        Returns:
            Dictionary with job_ids grouped by status
        """
        status_map: dict[str, list[str]] = {
            "pending": [],
            "running": [],
            "completed": [],
            "failed": [],
            "unknown": [],
        }

        for job in self.jobs:
            state = job.state
            if state == "PENDING":
                status_map["pending"].append(job.job_id)
            elif state == "RUNNING":
                status_map["running"].append(job.job_id)
            elif state == "COMPLETED":
                status_map["completed"].append(job.job_id)
            elif state in ("FAILED", "CANCELLED", "TIMEOUT"):
                status_map["failed"].append(job.job_id)
            else:
                status_map["unknown"].append(job.job_id)

        return status_map


def create_backend(
    use_slurm: bool,
    slurm_config: SlurmConfig | None = None,
    log_dir: Path | None = None,
    wait_for_completion: bool = True,
) -> ExecutionBackend:
    """Factory function to create an execution backend.

    Args:
        use_slurm: Whether to use SLURM backend
        slurm_config: SLURM configuration (required if use_slurm=True)
        log_dir: Directory for SLURM logs (required if use_slurm=True)
        wait_for_completion: Whether to wait for SLURM jobs to complete

    Returns:
        ExecutionBackend instance
    """
    if use_slurm:
        if slurm_config is None:
            slurm_config = SlurmConfig()
        if log_dir is None:
            raise ValueError("log_dir is required for SLURM backend")
        return SlurmBackend(slurm_config, log_dir, wait_for_completion)
    return LocalBackend()


# ============================================================================
# Wrapper functions for SLURM execution
# ============================================================================


def _run_resume_job(
    job_dir: str,
    overrides: list[str],
    retry_on_errors: list[str] | None,
    execution_mode: str,
) -> dict[str, Any]:
    """Wrapper function for running a resume job on SLURM.

    This function is pickle-able and can be submitted to SLURM.
    It reimports everything needed since it runs in a fresh process.

    Args:
        job_dir: Path to job directory
        overrides: Hydra-style overrides
        retry_on_errors: Error types to retry
        execution_mode: 'benign' or 'attack'

    Returns:
        Dictionary with job results
    """
    import asyncio
    from pathlib import Path

    # Import here to avoid pickle issues
    from prompt_siren.hydra_app import run_attack_experiment, run_benign_experiment
    from prompt_siren.job import Job

    job_path = Path(job_dir)
    job = Job.resume(
        job_dir=job_path,
        overrides=overrides if overrides else None,
        retry_on_errors=retry_on_errors,
    )

    experiment_config = job.to_experiment_config()

    if execution_mode == "benign":
        result = asyncio.run(run_benign_experiment(experiment_config, job=job))
    else:
        result = asyncio.run(run_attack_experiment(experiment_config, job=job))

    return {
        "job_name": job.job_config.job_name,
        "job_dir": str(job_path),
        "result": result,
    }


def _run_experiment_job(
    config_dict: dict[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    """Wrapper function for running a new experiment job on SLURM.

    This function is pickle-able and can be submitted to SLURM.

    Args:
        config_dict: Experiment configuration as dictionary
        execution_mode: 'benign' or 'attack'

    Returns:
        Dictionary with job results
    """
    import asyncio

    # Import here to avoid pickle issues
    from prompt_siren.config.experiment_config import ExperimentConfig
    from prompt_siren.hydra_app import run_attack_experiment, run_benign_experiment

    experiment_config = ExperimentConfig.model_validate(config_dict)

    if execution_mode == "benign":
        result = asyncio.run(run_benign_experiment(experiment_config))
    else:
        result = asyncio.run(run_attack_experiment(experiment_config))

    return {
        "job_name": experiment_config.output.job_name,
        "result": result,
    }

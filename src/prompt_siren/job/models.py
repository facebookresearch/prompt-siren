# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Data models for job management and persistence."""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from ..config.experiment_config import ExperimentConfig
from ..types import ExecutionMode


class ExceptionInfo(BaseModel):
    """Information about an exception that occurred during task execution."""

    exception_type: str
    exception_message: str
    exception_traceback: str
    occurred_at: datetime

    @classmethod
    def from_exception(cls, e: BaseException) -> ExceptionInfo:
        """Create ExceptionInfo from an exception."""
        return cls(
            exception_type=type(e).__name__,
            exception_message=str(e),
            exception_traceback=traceback.format_exc(),
            occurred_at=datetime.now(),
        )


class TaskRunResult(BaseModel):
    """Result for a single task run (one execution of a task).

    A task may be run multiple times for pass@k evaluation.
    """

    task_id: str
    run_id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    benign_score: float | None = None
    attack_score: float | None = None
    exception_info: ExceptionInfo | None = None


class TaskRunExecution(BaseModel):
    """Full execution data for a task run."""

    task_id: str
    run_id: str
    execution_id: str
    timestamp: datetime
    trace_id: str | None = None
    span_id: str | None = None
    messages: list[ModelMessage]
    usage: RunUsage
    attacks: dict[str, Any] | None = None  # Generated attacks


class JobConfig(ExperimentConfig):
    """Snapshot of the experiment configuration for a job.

    Extends ExperimentConfig with job-specific metadata. This is saved as
    config.yaml in the job directory and can be loaded via OmegaConf for
    resume with Hydra-style overrides.
    """

    job_name: str
    execution_mode: ExecutionMode
    created_at: datetime
    n_runs_per_task: int = Field(default=1, ge=1, description="Number of runs per task for pass@k")


class RunIndexEntry(BaseModel):
    """Entry in per-job index.jsonl for fast result lookup."""

    task_id: str
    run_id: str  # 8-char UUID
    timestamp: datetime
    benign_score: float | None
    attack_score: float | None
    exception_type: str | None = None  # None if successful
    path: Path  # Relative path to run directory from job directory


# Constants for file names
CONFIG_FILENAME = "config.yaml"
INDEX_FILENAME = "index.jsonl"
RESULT_FILENAME = "result.json"
INDEX_LOCK_FILENAME = "index.jsonl.lock"
TASK_RESULT_FILENAME = "result.json"
TASK_EXECUTION_FILENAME = "execution.json"

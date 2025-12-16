# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for job data models."""

from datetime import datetime
from pathlib import Path

import pytest
from prompt_siren.config.experiment_config import (
    AgentConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job.models import (
    ExceptionInfo,
    JobConfig,
    TaskRunResult,
)


class TestExceptionInfo:
    """Tests for ExceptionInfo.from_exception classmethod."""

    def test_from_exception_captures_type_and_message(self):
        """Test that exception type and message are captured correctly."""
        try:
            raise ValueError("test error message")
        except ValueError as e:
            info = ExceptionInfo.from_exception(e)
            assert info.exception_type == "ValueError"
            assert info.exception_message == "test error message"

    def test_from_exception_captures_traceback(self):
        """Test that traceback includes exception info."""
        try:
            raise KeyError("missing key")
        except KeyError as e:
            info = ExceptionInfo.from_exception(e)
            assert "KeyError" in info.exception_traceback
            assert "missing key" in info.exception_traceback

    def test_from_exception_with_chained_exception(self):
        """Test handling of chained exceptions preserves the outer exception."""
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise RuntimeError("outer") from ValueError("inner")
        except RuntimeError as e:
            info = ExceptionInfo.from_exception(e)
            assert info.exception_type == "RuntimeError"
            assert info.exception_message == "outer"


class TestTaskRunResult:
    """Tests for TaskRunResult.is_successful property."""

    def test_is_successful_when_completed_without_exception(self):
        """Test is_successful returns True when finished without exception."""
        result = TaskRunResult(
            task_id="task1",
            run_id="abc12345",
            started_at=datetime.now(),
            finished_at=datetime.now(),
            benign_score=1.0,
        )
        assert result.is_successful is True

    def test_is_not_successful_when_exception_occurred(self):
        """Test is_successful returns False when exception occurred."""
        result = TaskRunResult(
            task_id="task1",
            run_id="abc12345",
            started_at=datetime.now(),
            finished_at=datetime.now(),
            benign_score=0.0,
            exception_info=ExceptionInfo(
                exception_type="RuntimeError",
                exception_message="error",
                exception_traceback="traceback",
                occurred_at=datetime.now(),
            ),
        )
        assert result.is_successful is False

    def test_is_not_successful_when_not_finished(self):
        """Test is_successful returns False when task hasn't finished."""
        result = TaskRunResult(
            task_id="task1",
            run_id="abc12345",
            started_at=datetime.now(),
            finished_at=None,
        )
        assert result.is_successful is False


class TestJobConfig:
    """Tests for JobConfig validation."""

    def test_n_runs_per_task_rejects_zero(self):
        """Test that n_runs_per_task must be >= 1."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            JobConfig(
                job_name="test",
                execution_mode="benign",
                created_at=datetime.now(),
                n_runs_per_task=0,
                dataset=DatasetConfig(type="test", config={}),
                agent=AgentConfig(type="plain", config={}),
                attack=None,
                execution=ExecutionConfig(),
                telemetry=TelemetryConfig(),
                output=OutputConfig(jobs_dir=Path("jobs")),
            )

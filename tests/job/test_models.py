# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for job data models."""

from prompt_siren.job.models import ExceptionInfo


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


class TestJobConfig:
    """Tests for JobConfig validation."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""File-based span exporter for human-readable logging.

This module provides a span exporter that writes spans to a log file
in a human-readable format, with immediate flush for crash safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from typing import TextIO

# Context variable for tracking current task ID across async boundaries
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)


class FileSpanExporter(SpanExporter):
    """Exports spans to a human-readable log file with immediate flush.

    This exporter writes spans in a format similar to the console output
    but without colors, suitable for log files that need to be read later.

    The exporter flushes after each write for crash safety - if the program
    crashes, all spans up to that point will be persisted.
    """

    def __init__(
        self,
        file_path: Path,
        min_level: str = "INFO",
    ) -> None:
        """Initialize the file exporter.

        Args:
            file_path: Path to the log file
            min_level: Minimum log level to export (DEBUG, INFO, WARNING, ERROR)
        """
        self.file_path = file_path
        self.min_level = min_level.upper()
        self._file: TextIO | None = None
        self._level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    def _ensure_file_open(self) -> TextIO:
        """Ensure the log file is open, creating it if necessary."""
        if self._file is None or self._file.closed:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.file_path, "a", encoding="utf-8", buffering=1)
        return self._file

    def _format_span(self, span: ReadableSpan, depth: int = 0) -> str:
        """Format a span as a human-readable log line.

        Args:
            span: The span to format
            depth: Nesting depth for indentation

        Returns:
            Formatted log line
        """
        # Get timestamp
        if span.start_time:
            # Convert nanoseconds to datetime
            ts = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
            timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Determine log level from span attributes or default to SPAN
        level = "SPAN"
        if span.attributes:
            level = span.attributes.get("logfire.level_name", "SPAN")
            if isinstance(level, str):
                level = level.upper()

        # Build the message
        indent = "  " * depth
        name = span.name or "unnamed"

        # Get task_id from span attributes or context variable
        task_id = None
        if span.attributes and "task_id" in span.attributes:
            task_id = span.attributes["task_id"]
        else:
            task_id = current_task_id.get()
        task_prefix = f"[{task_id}] " if task_id else ""

        # Include key attributes in the message
        attrs_str = ""
        if span.attributes:
            # Filter to interesting attributes
            interesting_keys = [
                "job_name",
                "task_id",
                "model",
                "score",
                "status",
                "error",
                "logfire.msg",
            ]
            attrs = []
            for key in interesting_keys:
                if key in span.attributes:
                    value = span.attributes[key]
                    attrs.append(f"{key}={value}")

            # Check for logfire.msg which contains the formatted message
            if "logfire.msg" in span.attributes:
                name = span.attributes["logfire.msg"]
            elif attrs:
                attrs_str = f" ({', '.join(attrs)})"

        return f"{timestamp} | {level:<8} | {task_prefix}{indent}{name}{attrs_str}"

    def _should_export(self, span: ReadableSpan) -> bool:
        """Check if a span should be exported based on log level.

        Args:
            span: The span to check

        Returns:
            True if the span should be exported
        """
        # Get span level
        level = "INFO"
        if span.attributes and "logfire.level_name" in span.attributes:
            level = str(span.attributes["logfire.level_name"]).upper()

        # SPAN type spans are always at INFO level
        if level == "SPAN":
            level = "INFO"

        min_priority = self._level_priority.get(self.min_level, 1)
        span_priority = self._level_priority.get(level, 1)

        return span_priority >= min_priority

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to the log file.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult indicating success or failure
        """
        try:
            file = self._ensure_file_open()

            for span in spans:
                if self._should_export(span):
                    # Calculate depth based on parent spans
                    # For now, we use a simple heuristic based on span name patterns
                    depth = 0
                    if span.parent:
                        # Count parent chain depth (limited for performance)
                        depth = min(self._estimate_depth(span), 10)

                    line = self._format_span(span, depth)
                    file.write(line + "\n")
                    file.flush()  # Immediate flush for crash safety

            return SpanExportResult.SUCCESS

        except Exception:
            return SpanExportResult.FAILURE

    def _estimate_depth(self, span: ReadableSpan) -> int:
        """Estimate the nesting depth of a span.

        This is a heuristic based on span naming patterns since we don't
        have easy access to the full parent chain.

        Args:
            span: The span to estimate depth for

        Returns:
            Estimated depth level
        """
        # Use span context or attributes to estimate depth
        # This is a simplified approach
        name = span.name or ""

        # Common patterns for depth estimation
        if "experiment" in name.lower():
            return 0
        if "task" in name.lower():
            return 1
        if "attack" in name.lower():
            return 2
        if "chat" in name.lower() or "agent" in name.lower():
            return 3
        if "tool" in name.lower():
            return 4
        return 2  # Default middle depth

    def shutdown(self) -> None:
        """Shutdown the exporter and close the file."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the file.

        Args:
            timeout_millis: Timeout in milliseconds (unused, flush is immediate)

        Returns:
            True if successful
        """
        if self._file is not None and not self._file.closed:
            self._file.flush()
        return True

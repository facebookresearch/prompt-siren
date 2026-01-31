# Copyright (c) Meta Platforms, Inc. and affiliates.
"""OpenTelemetry setup and configuration using Logfire."""

from __future__ import annotations

import logging
import os
from logging import basicConfig
from pathlib import Path
from typing import Literal

import logfire
from logfire import ConsoleOptions
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .file_exporter import FileSpanExporter
from .pydantic_ai_processor.span_processor import OpenInferenceSpanProcessor

# Type alias for log levels
LogLevel = Literal["trace", "debug", "info", "notice", "warn", "warning", "error", "fatal"]

# Global reference to file exporter for adding job-specific logging
_file_exporter: FileSpanExporter | None = None


def setup_telemetry(
    service_name: str = "prompt-siren",
    otlp_endpoint: str | None = None,
    enable_console_export: bool = True,
    log_level: str = "INFO",
) -> None:
    """Set up OpenTelemetry using Logfire with the specified configuration.

    Args:
        service_name: Name of the service for telemetry
        otlp_endpoint: Optional OTLP endpoint URL (e.g., "http://localhost:6006/v1/traces")
        enable_console_export: Whether to export spans to console
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    """

    # Set environment variables for OTLP export if endpoint provided
    if otlp_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_endpoint

    # Convert log level to lowercase for console options
    level_lower = log_level.lower()
    # Map common log levels to logfire's expected format
    level_map = {"warning": "warn"}
    console_level: LogLevel = level_map.get(level_lower, level_lower)  # type: ignore[assignment]

    # Configure console output
    console_config = (
        ConsoleOptions(
            colors="auto",
            span_style="show-parents",
            include_timestamps=True,
            verbose=False,
            min_log_level=console_level,
        )
        if enable_console_export
        else False
    )

    # Configure Logfire and get the instance
    logfire.configure(
        service_name=service_name,
        send_to_logfire=False,  # Don't send to Logfire's platform
        console=console_config,
        additional_span_processors=[OpenInferenceSpanProcessor()],  # Add our custom processor
        scrubbing=False,
        inspect_arguments=False,
    )

    # Instrument PydanticAI
    logfire.instrument_pydantic_ai(include_binary_content=True)
    basicConfig(handlers=[logfire.LogfireLoggingHandler()], level=log_level.upper())
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)


def add_file_logging(log_file: Path, log_level: str = "INFO") -> None:
    """Add file-based logging for spans.

    This should be called after setup_telemetry() and after the job directory
    is created, to enable logging to a file in the job directory.

    If file logging is already active, this will close the existing file
    and switch to the new one (to support resume to the same job directory).

    Args:
        log_file: Path to the log file (e.g., job_dir / "siren.log")
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    """
    global _file_exporter

    # Close existing file exporter if any (e.g., on resume)
    if _file_exporter is not None:
        _file_exporter.shutdown()

    # Create file exporter
    _file_exporter = FileSpanExporter(file_path=log_file, min_level=log_level)

    # Add the file span processor (synchronous for crash safety)
    tracer_provider = trace.get_tracer_provider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(_file_exporter))  # type: ignore[attr-defined]


def close_file_logging() -> None:
    """Close the file exporter and flush any remaining data."""
    global _file_exporter
    if _file_exporter is not None:
        _file_exporter.shutdown()
        _file_exporter = None

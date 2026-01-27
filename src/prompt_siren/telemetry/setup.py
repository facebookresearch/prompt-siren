# Copyright (c) Meta Platforms, Inc. and affiliates.
"""OpenTelemetry setup and configuration using Logfire."""

import logging
import os
from logging import basicConfig
from pathlib import Path

import logfire
from logfire import ConsoleOptions

from .pydantic_ai_processor.span_processor import OpenInferenceSpanProcessor

# Module-level logger
logger = logging.getLogger(__name__)

# Track file handlers so we can remove them on subsequent calls
_job_file_handler: logging.FileHandler | None = None


def setup_telemetry(
    service_name: str = "prompt-siren",
    otlp_endpoint: str | None = None,
    enable_console_export: bool = True,
) -> None:
    """Set up OpenTelemetry using Logfire with the specified configuration.

    Args:
        service_name: Name of the service for telemetry
        otlp_endpoint: Optional OTLP endpoint URL (e.g., "http://localhost:6006/v1/traces")
        enable_console_export: Whether to export spans to console
    """

    # Set environment variables for OTLP export if endpoint provided
    if otlp_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_endpoint

    # Configure console output
    console_config = (
        ConsoleOptions(
            colors="auto",
            span_style="show-parents",
            include_timestamps=True,
            verbose=False,
            min_log_level="debug",
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
    basicConfig(handlers=[logfire.LogfireLoggingHandler()], level="INFO")
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)


def setup_job_file_logging(
    job_dir: Path,
    log_filename: str = "job.log",
) -> Path:
    """Set up file logging to write logs to the job directory.

    This adds a file handler to the root logger that writes all INFO and above
    logs to a file in the job directory. The file handler is formatted with
    timestamps and log levels for easy debugging.

    Args:
        job_dir: Path to the job directory where logs will be written
        log_filename: Name of the log file (default: "job.log")

    Returns:
        Path to the log file
    """
    global _job_file_handler

    # Remove any existing job file handler
    root_logger = logging.getLogger()
    if _job_file_handler is not None:
        root_logger.removeHandler(_job_file_handler)
        _job_file_handler.close()
        _job_file_handler = None

    # Create file handler for job directory
    log_path = job_dir / log_filename
    _job_file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _job_file_handler.setLevel(logging.INFO)

    # Create formatter with timestamp, level, logger name, and message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _job_file_handler.setFormatter(formatter)

    # Add to root logger
    root_logger.addHandler(_job_file_handler)

    logger.info(f"Job logging initialized: {log_path}")
    return log_path

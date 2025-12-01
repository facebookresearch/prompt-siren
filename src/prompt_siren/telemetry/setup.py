# Copyright (c) Meta Platforms, Inc. and affiliates.
"""OpenTelemetry setup and configuration using Logfire."""

import logging
import os
from logging import basicConfig

import logfire
from logfire import ConsoleOptions

from .pydantic_ai_processor.span_processor import OpenInferenceSpanProcessor


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

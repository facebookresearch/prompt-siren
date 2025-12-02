# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Telemetry module for Siren.

Provides OpenTelemetry instrumentation and conversation logging capabilities.
"""

from .setup import setup_telemetry

__all__ = [
    "setup_telemetry",
]

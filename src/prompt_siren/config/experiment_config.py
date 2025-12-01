# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Experiment configuration schema using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits


class AgentConfig(BaseModel):
    """Configuration for agents.."""

    type: str = Field(description="Agent type identifier")
    config: dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")


class AttackConfig(BaseModel):
    """Configuration for attacks."""

    type: str = Field(description="Attack type identifier")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Attack-specific configuration"
    )


class DatasetConfig(BaseModel):
    """Configuration for datasets."""

    type: str = Field(description="Dataset type identifier")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Dataset-specific configuration"
    )


class SandboxManagerConfig(BaseModel):
    """Configuration for sandbox managers."""

    type: str = Field(description="Sandbox manager type identifier")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Sandbox manager-specific configuration",
    )


class ExecutionConfig(BaseModel):
    """Configuration for experiment execution."""

    concurrency: int = Field(default=1, description="Maximum number of tasks to run concurrently")


class OutputConfig(BaseModel):
    """Configuration for experiment output."""

    trace_dir: Path = Field(default=Path("traces"), description="Directory to store trace data")


class TelemetryConfig(BaseModel):
    """Configuration for observability and telemetry."""

    trace_console: bool = Field(default=False, description="Whether to output traces to console")
    otel_endpoint: str | None = Field(default=None, description="OpenTelemetry OTLP endpoint")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    # Experiment metadata
    name: str = Field(default="experiment", description="Experiment name")

    # Component configurations
    agent: AgentConfig = Field(description="Agent configuration")
    dataset: DatasetConfig = Field(
        description="Dataset configuration (specifies tasks and required environment type)"
    )
    attack: AttackConfig | None = Field(
        default=None,
        description="Attack configuration (optional for benign-only)",
    )
    sandbox_manager: SandboxManagerConfig | None = Field(
        default=None,
        description="Sandbox manager configuration (required for datasets using BashEnvironment, optional for others)",
    )

    # Execution settings
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Execution configuration"
    )
    task_ids: list[str] | None = Field(
        default=None,
        description=(
            "Task IDs to run. None = all tasks appropriate for mode "
            "(all benign tasks in benign mode, all couples in attack mode). "
            "Format: 'benign_id:malicious_id' for couples, 'task_id' for individual tasks."
        ),
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig, description="Telemetry configuration"
    )
    usage_limits: UsageLimits | None = Field(
        default=None,
        description=("Usage limits configuration using PydanticAI's UsageLimits."),
    )

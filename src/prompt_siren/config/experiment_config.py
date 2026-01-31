# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Experiment configuration schema using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
    n_runs_per_task: int = Field(
        default=1,
        ge=1,
        description="Number of runs per task. Use with resume to run tasks multiple times.",
    )


class OutputConfig(BaseModel):
    """Configuration for experiment output."""

    jobs_dir: Path = Field(default=Path("jobs"), description="Directory to store job results")
    job_name: str | None = Field(
        default=None, description="Custom job name (auto-generated if None)"
    )


class TelemetryConfig(BaseModel):
    """Configuration for observability and telemetry."""

    trace_console: bool = Field(default=False, description="Whether to output traces to console")
    otel_endpoint: str | None = Field(default=None, description="OpenTelemetry OTLP endpoint")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level for file and console output",
    )
    file: bool = Field(
        default=True,
        description="Whether to write logs to a file in the job directory",
    )


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission."""

    partition: str = Field(
        default="learnfair",
        description="SLURM partition to submit jobs to",
    )
    time_minutes: int = Field(
        default=240,
        ge=1,
        description="Time limit for jobs in minutes",
    )
    gpus_per_node: int = Field(
        default=0,
        ge=0,
        description="Number of GPUs per node",
    )
    cpus_per_task: int = Field(
        default=4,
        ge=1,
        description="Number of CPUs per task",
    )
    mem_gb: int = Field(
        default=32,
        ge=1,
        description="Memory per job in GB",
    )
    constraint: str | None = Field(
        default=None,
        description="SLURM constraint (e.g., 'volta32gb')",
    )
    additional_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional SLURM parameters passed to submitit",
    )


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
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    usage_limits: UsageLimits | None = Field(
        default=None,
        description=("Usage limits configuration using PydanticAI's UsageLimits."),
    )
    slurm: SlurmConfig = Field(
        default_factory=SlurmConfig,
        description="SLURM configuration for job submission (used with --slurm flag)",
    )

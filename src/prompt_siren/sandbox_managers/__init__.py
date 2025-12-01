# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Sandbox manager module for managing containerized execution environments."""

from .abstract import (
    AbstractSandboxManager,
    ExecOutput,
    ExecTimeoutError,
    Output,
    StderrChunk,
    StdoutChunk,
)
from .image_spec import (
    BuildImageSpec,
    BuildStage,
    ImageSpec,
    ImageTag,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from .registry import (
    create_sandbox_manager,
    get_registered_sandbox_managers,
    get_sandbox_config_class,
    register_sandbox_manager,
)
from .sandbox_state import ContainerID, SandboxState
from .sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    SandboxTaskSetup,
    TaskSetup,
)

__all__ = [
    "AbstractSandboxManager",
    "BuildImageSpec",
    "BuildStage",
    "ContainerID",
    "ContainerSetup",
    "ContainerSpec",
    "ExecOutput",
    "ExecTimeoutError",
    "ImageSpec",
    "ImageTag",
    "MultiStageBuildImageSpec",
    "NetworkConfig",
    "Output",
    "PullImageSpec",
    "SandboxState",
    "SandboxTaskSetup",
    "StderrChunk",
    "StdoutChunk",
    "TaskSetup",
    "create_sandbox_manager",
    "get_registered_sandbox_managers",
    "get_sandbox_config_class",
    "register_sandbox_manager",
]

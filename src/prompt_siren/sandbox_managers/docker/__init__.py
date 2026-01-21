# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Docker-based sandbox manager implementation."""

from .local_client import extract_registry_from_tag
from .manager import (
    create_docker_sandbox_manager,
    DockerSandboxConfig,
    DockerSandboxManager,
)

__all__ = [
    # Sandbox manager
    "DockerSandboxConfig",
    "DockerSandboxManager",
    "create_docker_sandbox_manager",
    # Utilities
    "extract_registry_from_tag",
]

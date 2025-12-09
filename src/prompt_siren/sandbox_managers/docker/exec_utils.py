# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Utilities for executing commands in Docker containers."""

from __future__ import annotations

import shlex
from pathlib import Path

import anyio

from ..abstract import ExecOutput, ExecTimeoutError
from ..sandbox_state import ContainerID
from .plugins import AbstractDockerClient


async def exec_in_container(
    docker: AbstractDockerClient,
    container_id: ContainerID,
    cmd: str | list[str],
    stdin: str | bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    timeout: int | None = None,
    default_timeout: int = 300,
    shell_path: Path | None = None,
) -> ExecOutput:
    """Execute a command in a Docker container.

    Args:
        docker: Abstract Docker client
        container_id: Container ID to execute command in
        cmd: Command to execute (string or list of arguments)
        stdin: Optional stdin data to pass to the command
        cwd: Optional working directory for command execution
        env: Optional environment variables for command execution
        user: Optional user to run command as
        timeout: Optional timeout in seconds
        default_timeout: Default timeout if timeout is None
        shell_path: Optional path to shell executable (defaults to /bin/bash)

    Returns:
        ExecOutput containing stdout/stderr chunks and exit code

    Raises:
        ExecTimeoutError: If command execution exceeds timeout
    """
    # Get container
    container = await docker.get_container(container_id)
    _shell_path = str(shell_path) if shell_path is not None else "/bin/bash"

    # Normalize command to bash -c format
    if isinstance(cmd, list):
        bash_cmd = [_shell_path, "-c", " ".join(shlex.quote(arg) for arg in cmd)]
    else:
        bash_cmd = [_shell_path, "-c", cmd]

    timeout_value = timeout if timeout is not None else default_timeout

    try:
        # Use anyio for timeout (Python 3.10 compatible)
        with anyio.fail_after(timeout_value):
            # Execute command in container
            return await container.exec(
                cmd=bash_cmd,
                stdin=stdin,
                user=user or "",
                environment=env,
                workdir=cwd,
                timeout=timeout_value,
            )

    except TimeoutError as e:
        raise ExecTimeoutError(container_id, cmd, timeout_value) from e

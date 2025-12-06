# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Utilities for executing commands in Docker containers."""

from __future__ import annotations

import shlex
from pathlib import Path

import anyio
from aiohttp import ClientTimeout

try:
    import aiodocker
except ImportError as e:
    raise ImportError(
        "Docker sandbox manager requires the 'docker' optional dependency. "
        "Install with: pip install 'prompt-siren[docker]'"
    ) from e

from ..abstract import ExecOutput, ExecTimeoutError, StderrChunk, StdoutChunk
from ..sandbox_state import ContainerID


async def exec_in_container(
    docker: aiodocker.Docker,
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
        docker: Docker client
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
    container = await docker.containers.get(container_id)
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
            # Create exec instance
            exec_instance = await container.exec(
                cmd=bash_cmd,
                stdout=True,
                stderr=True,
                stdin=stdin is not None,
                user=user or "",
                environment=env,
                workdir=cwd,
            )

            # Start execution
            stream = exec_instance.start(detach=False, timeout=ClientTimeout(total=timeout_value))

            # Write stdin if provided
            if stdin is not None:
                if isinstance(stdin, str):
                    stdin = stdin.encode()
                await stream.write_in(stdin)
                # Signal EOF on stdin without closing stdout/stderr
                # Access transport directly like write_in() does
                assert stream._resp is not None
                assert stream._resp.connection is not None
                transport = stream._resp.connection.transport
                if transport and transport.can_write_eof():
                    transport.write_eof()

            # Read output
            outputs = []
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break
                decoded = msg.data.decode("utf-8", errors="replace")
                if msg.stream == 1:  # stdout
                    outputs.append(StdoutChunk(content=decoded))
                elif msg.stream == 2:  # stderr
                    outputs.append(StderrChunk(content=decoded))

            # Get exit code
            exit_code = (await exec_instance.inspect())["ExitCode"]

            return ExecOutput(outputs=outputs, exit_code=exit_code)

    except TimeoutError as e:
        raise ExecTimeoutError(container_id, cmd, timeout_value) from e

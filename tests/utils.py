# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared test utilities."""

from prompt_siren.sandbox_managers.abstract import ExecOutput, StderrChunk, StdoutChunk


def create_exec_output(
    stdout: str | None = None, stderr: str | None = None, exit_code: int = 0
) -> ExecOutput:
    """Helper to create ExecOutput with the new ADT format.

    Args:
        stdout: Standard output content
        stderr: Standard error content
        exit_code: Exit code of the command

    Returns:
        ExecOutput with the provided content
    """
    outputs = []
    if stdout is not None:
        outputs.append(StdoutChunk(content=stdout))
    if stderr is not None:
        outputs.append(StderrChunk(content=stderr))
    return ExecOutput(outputs=outputs, exit_code=exit_code)

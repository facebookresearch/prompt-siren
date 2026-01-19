"""Common Docker client exceptions."""

from __future__ import annotations


class DockerClientError(Exception):
    """Error executing Docker client command.

    This exception is raised by any Docker client implementation when a Docker
    operation fails.

    Attributes:
        status: HTTP status code from Docker API (if available), e.g. 404 for not found
        stdout: Standard output from the command (if available)
        stderr: Standard error from the command (if available)
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ):
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"{message}\nstdout: {stdout}\nstderr: {stderr}")

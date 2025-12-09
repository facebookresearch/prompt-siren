"""Common Docker client exceptions."""

from __future__ import annotations


class DockerClientError(Exception):
    """Error executing Docker client command.

    This exception is raised by any Docker client implementation when a Docker
    operation fails.
    """

    def __init__(self, message: str, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"{message}\nstdout: {stdout}\nstderr: {stderr}")

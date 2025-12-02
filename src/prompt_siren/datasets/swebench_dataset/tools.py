# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import shlex
from pathlib import Path
from typing import Literal, Protocol

from pydantic_ai import RunContext

from ...environments.bash_env import BashEnvState
from ...sandbox_managers.abstract import ExecOutput


class ExecFn(Protocol):
    async def __call__(
        self,
        cmd: str | list[str],
        timeout: int | None = None,
    ) -> ExecOutput: ...


class ToolExecutionError(Exception):
    """Base exception for tool execution errors."""


class ToolFileNotFoundError(ToolExecutionError):
    """Raised when a file cannot be found."""


class ToolDirectoryNotFoundError(ToolExecutionError):
    """Raised when a directory cannot be found."""


class ToolPermissionError(ToolExecutionError):
    """Raised when a permission error occurs."""


def _create_exec_fn(
    run_ctx: RunContext[BashEnvState],
    cwd: str = "/",
    user: str | None = None,
    env: dict[str, str] | None = None,
    shell_path: Path | None = None,
) -> ExecFn:
    """Create an exec function from a RunContext with specified execution parameters.

    Args:
        run_ctx: The RunContext containing sandbox manager and container ID
        cwd: Working directory for command execution (default: /testbed)
        user: User to run commands as (default: container default)
        env: Environment variables to set (default: SWE-bench standard environment)
        shell_path: Optional path to shell executable for command execution (defaults to /bin/bash)

    Returns:
        An exec function that can execute commands in the container
    """
    # Set default environment variables aligned with mini-SWE-agent
    # These disable progress bars and use simpler pagers for cleaner output
    default_env = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }
    # Merge with any provided env vars (provided env takes precedence)
    merged_env = {**default_env, **(env or {})}

    async def exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
        sandbox_manager = run_ctx.deps.sandbox_manager
        container_id = run_ctx.deps.agent_container_id
        # Default timeout of 60 seconds aligned with mini-SWE-agent
        effective_timeout = timeout if timeout is not None else 60
        return await sandbox_manager.exec(
            container_id,
            cmd,
            stdin=None,
            cwd=cwd,
            env=merged_env,
            user=user,
            timeout=effective_timeout,
            shell_path=shell_path,
        )

    return exec_fn


def _escape_shell_arg(arg: str | int) -> str:
    """Safely escape an argument for shell command interpolation.

    This function uses shlex.quote to escape strings for safe use in shell commands,
    preventing command injection attacks. Integer arguments are converted to strings
    without escaping as they are inherently safe.

    Args:
        arg: The argument to escape (string or integer)

    Returns:
        Safely escaped string suitable for shell command interpolation

    Example:
        >>> _escape_shell_arg("file with spaces.txt")
        "'file with spaces.txt'"
        >>> _escape_shell_arg("file; rm -rf /")
        "'file; rm -rf /'"
        >>> _escape_shell_arg(42)
        '42'
    """
    if isinstance(arg, int):
        return str(arg)
    return shlex.quote(str(arg))


def _truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate output if it exceeds max_chars, showing head and tail.

    Follows mini-swe-agent's approach: when output is too long, show first 5k chars,
    an elision message, and last 5k chars. This prevents wasting tokens on long outputs.

    Args:
        output: The output string to potentially truncate
        max_chars: Maximum characters before truncation (default: 10000)

    Returns:
        Original output if under limit, otherwise truncated output with elision message
    """
    if len(output) <= max_chars:
        return output

    half = max_chars // 2
    elided_count = len(output) - max_chars

    warning = (
        "\n\n[WARNING: Output too long. Showing first and last 5000 characters.]\n"
        f"[... ELIDED {elided_count} CHARACTERS ...]\n\n"
    )

    return output[:half] + warning + output[-half:]


async def bash(
    run_ctx: RunContext[BashEnvState],
    command: str | list[str],
    timeout: int | None = None,
) -> str:
    """Execute a bash command in the container.

    Commands are executed in a subshell. Environment variables and directory changes
    are NOT persistent across commands. The virtual environment at /opt/venv is NOT
    automatically activated - prefix commands with 'source /opt/venv/bin/activate &&'
    when you need Python packages.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        command: Command to execute (string or list of arguments)
        timeout: Optional timeout in seconds

    Returns:
        Combined stdout/stderr output, truncated if over 10,000 characters
    """
    exec_fn = _create_exec_fn(run_ctx)
    output = await exec_fn(command, timeout=timeout)
    combined = output.combined or ""
    return _truncate_output(combined)


async def read_file(
    run_ctx: RunContext[BashEnvState],
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read file contents, optionally within a line range.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        path: Path to the file to read
        start_line: Optional starting line number (1-indexed, inclusive)
        end_line: Optional ending line number (1-indexed, inclusive)

    Returns:
        File contents as a string with line numbers

    Raises:
        ToolFileNotFoundError: If the file does not exist
        ToolExecutionError: If the read operation fails
    """
    exec_fn = _create_exec_fn(run_ctx)
    escaped_path = _escape_shell_arg(path)

    if start_line is not None and end_line is not None:
        # Use sed to extract line range with line numbers
        cmd = f"cat -n {escaped_path} | sed -n '{start_line},{end_line}p'"
    elif start_line is not None:
        # From start_line to end of file
        cmd = f"cat -n {escaped_path} | sed -n '{start_line},$p'"
    elif end_line is not None:
        # From beginning to end_line
        cmd = f"cat -n {escaped_path} | sed -n '1,{end_line}p'"
    else:
        # Read entire file with line numbers
        cmd = f"cat -n {escaped_path}"

    result = await exec_fn(cmd)

    if result.exit_code != 0:
        if "No such file or directory" in (result.stderr or ""):
            raise ToolFileNotFoundError(f"File not found: {path}\nDetails: {result.stderr}")
        raise ToolExecutionError(f"Failed to read file {path}\nDetails: {result.stderr}")

    return result.stdout or ""


async def write_file(
    run_ctx: RunContext[BashEnvState],
    path: str,
    content: str,
    create_dirs: bool = True,
) -> str:
    """Write content to a file, optionally creating parent directories.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        path: Path to the file to write
        content: Content to write to the file
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        Success message

    Raises:
        ToolPermissionError: If there are permission issues
        ToolExecutionError: If the write operation fails
    """
    exec_fn = _create_exec_fn(run_ctx)
    escaped_path = _escape_shell_arg(path)

    # Create parent directories if requested
    if create_dirs:
        # Use Python's os.path.dirname to get parent, then escape it
        parent_dir = os.path.dirname(path)
        escaped_parent_dir = _escape_shell_arg(parent_dir)
        mkdir_result = await exec_fn(f"mkdir -p {escaped_parent_dir}")
        if mkdir_result.exit_code != 0:
            raise ToolExecutionError(
                f"Failed to create parent directories for {path}\nDetails: {mkdir_result.stderr}"
            )

    # Write content to file using cat with heredoc
    # Use a unique delimiter that's unlikely to appear in the content
    delimiter = "EOF_WRITE_FILE_DELIMITER"
    cmd = f"cat > {escaped_path} << '{delimiter}'\n{content}\n{delimiter}"

    result = await exec_fn(cmd)

    if result.exit_code != 0:
        if "Permission denied" in (result.stderr or ""):
            raise ToolPermissionError(
                f"Permission denied writing to {path}\nDetails: {result.stderr}"
            )
        raise ToolExecutionError(f"Failed to write file {path}\nDetails: {result.stderr}")

    return f"Successfully wrote to {path}"


async def list_directory(
    run_ctx: RunContext[BashEnvState],
    path: str = ".",
    recursive: bool = False,
    pattern: str | None = None,
) -> list[str]:
    """List files and directories, optionally filtered by pattern.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        path: Directory path to list (default: current directory)
        recursive: Whether to list recursively
        pattern: Optional glob pattern to filter results (e.g., "*.py")

    Returns:
        List of file/directory paths

    Raises:
        ToolDirectoryNotFoundError: If the directory does not exist
        ToolExecutionError: If the listing operation fails
    """
    exec_fn = _create_exec_fn(run_ctx)
    escaped_path = _escape_shell_arg(path)

    if recursive:
        # Use find for recursive listing
        if pattern:
            escaped_pattern = _escape_shell_arg(pattern)
            cmd = f"find {escaped_path} -name {escaped_pattern}"
        else:
            cmd = f"find {escaped_path}"
    else:
        # Use find with maxdepth for non-recursive listing
        # (find's -name handles patterns properly with escaping, unlike ls globs)
        if pattern:
            escaped_pattern = _escape_shell_arg(pattern)
            cmd = f"find {escaped_path} -maxdepth 1 -name {escaped_pattern}"
        else:
            cmd = f"ls -1 {escaped_path}"

    result = await exec_fn(cmd)

    if result.exit_code != 0:
        # If pattern doesn't match anything, ls returns error, but that's ok - return empty list
        if pattern and "No such file or directory" in (result.stderr or ""):
            return []
        # Otherwise, if we got "No such file or directory", it means the directory doesn't exist
        if "No such file or directory" in (result.stderr or ""):
            raise ToolDirectoryNotFoundError(
                f"Directory not found: {path}\nDetails: {result.stderr}"
            )
        raise ToolExecutionError(f"Failed to list directory {path}\nDetails: {result.stderr}")

    # Parse output into list of paths
    output = result.stdout or ""
    if not output.strip():
        return []

    return [line.strip() for line in output.strip().split("\n") if line.strip()]


async def find_files(
    run_ctx: RunContext[BashEnvState],
    pattern: str,
    path: str = ".",
    max_depth: int | None = None,
    file_type: Literal["f", "d", "l"] | None = None,
) -> list[str]:
    """Find files by name pattern using the find command.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        pattern: Name pattern to match (e.g., "*.py", "*test*")
        path: Directory to search in (default: current directory)
        max_depth: Maximum depth to search (None for unlimited)
        file_type: Filter by type - "f" (file), "d" (directory), "l" (symlink)

    Returns:
        List of matching file paths (empty list if no matches)

    Raises:
        ToolExecutionError: If the find operation fails
    """
    exec_fn = _create_exec_fn(run_ctx)
    escaped_path = _escape_shell_arg(path)
    escaped_pattern = _escape_shell_arg(pattern)

    cmd_parts = ["find", escaped_path]

    if max_depth is not None:
        cmd_parts.extend(["-maxdepth", str(max_depth)])

    if file_type:
        cmd_parts.extend(["-type", file_type])

    cmd_parts.extend(["-name", escaped_pattern])

    cmd = " ".join(cmd_parts)
    result = await exec_fn(cmd)

    if result.exit_code != 0:
        raise ToolExecutionError(
            f"Failed to find files with pattern {pattern}\nDetails: {result.stderr}"
        )

    # Parse output into list of paths
    output = result.stdout or ""
    if not output.strip():
        return []

    return [line.strip() for line in output.strip().split("\n") if line.strip()]


async def search_files(
    run_ctx: RunContext[BashEnvState],
    pattern: str,
    path: str = ".",
    context_lines: int = 0,
    case_sensitive: bool = True,
    max_results: int | None = None,
    file_pattern: str | None = None,
) -> str:
    """Search for text pattern in files using grep.

    Args:
        run_ctx: The RunContext with sandbox manager and container
        pattern: Text pattern to search for (supports regex)
        path: Directory to search in (default: current directory)
        context_lines: Number of context lines to show around matches
        case_sensitive: Whether search is case-sensitive (default: True)
        max_results: Maximum number of result lines to return
        file_pattern: Filter by filename pattern (e.g., "*.py")

    Returns:
        Formatted grep output with file:line:content (empty string if no matches)

    Raises:
        ToolExecutionError: If the grep operation fails
    """
    exec_fn = _create_exec_fn(run_ctx)
    escaped_pattern = _escape_shell_arg(pattern)
    escaped_path = _escape_shell_arg(path)

    # BusyBox grep doesn't support --include, so we use find + grep for file filtering
    if file_pattern:
        escaped_file_pattern = _escape_shell_arg(file_pattern)
        # Use find to filter files by pattern, then exec grep on each
        grep_parts = [
            "grep",
            "-H",
            "-n",
        ]  # -H for filename prefix, -n for line numbers

        if not case_sensitive:
            grep_parts.append("-i")

        if context_lines > 0:
            grep_parts.extend(["-C", str(context_lines)])

        grep_parts.append(escaped_pattern)
        grep_cmd = " ".join(grep_parts)

        # Use find -exec to run grep on matching files
        # The + at the end passes multiple files to grep at once (more efficient)
        cmd = f"find {escaped_path} -name {escaped_file_pattern} -type f -exec {grep_cmd} {{}} +"
    else:
        # Standard recursive grep
        cmd_parts = ["grep", "-r", "-n"]  # recursive, with line numbers

        if not case_sensitive:
            cmd_parts.append("-i")

        if context_lines > 0:
            cmd_parts.extend(["-C", str(context_lines)])

        cmd_parts.extend([escaped_pattern, escaped_path])
        cmd = " ".join(cmd_parts)

    # Add head to limit results if specified
    if max_results is not None:
        cmd = f"{cmd} | head -n {max_results}"

    result = await exec_fn(cmd)

    # grep returns exit code 1 when no matches found, which is not an error
    if result.exit_code not in (0, 1):
        raise ToolExecutionError(
            f"Failed to search files for pattern {pattern}\nDetails: {result.stderr}"
        )

    return result.stdout or ""

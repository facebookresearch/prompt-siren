# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWEBench dataset tools."""

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from prompt_siren.datasets.swebench_dataset.tools import (
    _truncate_output,
    bash,
    find_files,
    list_directory,
    read_file,
    search_files,
    ToolDirectoryNotFoundError,
    ToolExecutionError,
    ToolFileNotFoundError,
    ToolPermissionError,
    write_file,
)
from prompt_siren.environments.bash_env import BashEnvState
from prompt_siren.sandbox_managers.abstract import ExecOutput
from prompt_siren.sandbox_managers.sandbox_state import SandboxState
from prompt_siren.sandbox_managers.sandbox_task_setup import SandboxTaskSetup
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from ...utils import create_exec_output


@dataclass
class MockSandboxManager:
    """Mock sandbox manager for testing."""

    exec_fn: AsyncMock

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence) -> AsyncIterator[None]:
        """Mock setup_batch - not used in tool tests."""
        yield

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        """Mock setup_task - not used in tool tests."""
        yield SandboxState(
            agent_container_id="test-container",
            service_containers={},
            execution_id="",
            network_id=None,
        )

    async def clone_sandbox_state(self, source_state):
        """Mock clone_sandbox - not used in tool tests."""
        return source_state

    async def exec(
        self,
        container_id: str,
        cmd: str | list[str],
        stdin: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        shell_path: object | None = None,
    ) -> ExecOutput:
        """Delegate to the mock exec function."""
        return await self.exec_fn(cmd, timeout)


@pytest.fixture
def mock_context() -> Callable[[AsyncMock], RunContext[BashEnvState]]:
    """Create a mock RunContext for testing tools."""

    def _create_context(exec_fn: AsyncMock) -> RunContext[BashEnvState]:
        sandbox_manager = MockSandboxManager(exec_fn=exec_fn)
        sandbox_state = SandboxState(
            agent_container_id="test-container",
            service_containers={},
            execution_id="",
            network_id=None,
        )
        env_state = BashEnvState(sandbox_state=sandbox_state, sandbox_manager=sandbox_manager)
        return RunContext(
            deps=env_state,
            model=TestModel(),
            usage=RunUsage(),
            messages=[],
        )

    return _create_context


class TestBash:
    """Tests for the bash tool."""

    @pytest.mark.anyio
    async def test_bash_success(self, mock_context):
        """Test bash command execution with successful output."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(stdout="hello world", stderr=None, exit_code=0)
        )
        ctx = mock_context(mock_exec)

        result = await bash(ctx, "echo 'hello world'")
        assert result == "hello world"

    @pytest.mark.anyio
    async def test_bash_with_stderr(self, mock_context):
        """Test bash command that produces stderr."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(stdout="", stderr="error message", exit_code=1)
        )
        ctx = mock_context(mock_exec)

        result = await bash(ctx, "false")
        assert result == "error message"

    @pytest.mark.anyio
    async def test_bash_with_list_command(self, mock_context):
        """Test bash with command as list."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert isinstance(cmd, list)
            assert cmd == ["ls", "-la"]
            return create_exec_output(stdout="files", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))

        result = await bash(ctx, ["ls", "-la"])
        assert result != ""

    @pytest.mark.anyio
    async def test_bash_output_truncation_short(self, mock_context):
        """Test bash command with short output (no truncation)."""
        short_output = "x" * 100
        mock_exec = AsyncMock(
            return_value=create_exec_output(stdout=short_output, stderr=None, exit_code=0)
        )
        ctx = mock_context(mock_exec)

        result = await bash(ctx, "echo test")
        assert result == short_output
        assert "ELIDED" not in result

    @pytest.mark.anyio
    async def test_bash_output_truncation_long(self, mock_context):
        """Test bash command with long output (truncation applied)."""
        # Create output longer than 10000 chars
        long_output = "x" * 15000
        mock_exec = AsyncMock(
            return_value=create_exec_output(stdout=long_output, stderr=None, exit_code=0)
        )
        ctx = mock_context(mock_exec)

        result = await bash(ctx, "cat large_file")
        # Should be truncated
        assert len(result) < len(long_output)
        assert "ELIDED" in result
        assert "WARNING: Output too long" in result
        # Should contain first and last parts
        assert result.startswith("x" * 100)  # First chars
        assert result.endswith("x" * 100)  # Last chars


class TestTruncateOutput:
    """Tests for the _truncate_output helper function."""

    def test_short_output_not_truncated(self):
        """Test that output under limit is not truncated."""
        output = "a" * 100
        result = _truncate_output(output)
        assert result == output

    def test_exact_limit_not_truncated(self):
        """Test that output at exact limit is not truncated."""
        output = "a" * 10000
        result = _truncate_output(output)
        assert result == output
        assert "ELIDED" not in result

    def test_long_output_truncated(self):
        """Test that output over limit is truncated with head and tail."""
        # Create 15000 char output
        output = "a" * 15000
        result = _truncate_output(output)

        # Should be shorter than original
        assert len(result) < len(output)

        # Should contain warning and elision message
        assert "WARNING: Output too long" in result
        assert "ELIDED 5000 CHARACTERS" in result

        # Should start with first 5000 chars
        assert result.startswith("a" * 5000)

        # Should end with last 5000 chars
        assert result.endswith("a" * 5000)

    def test_truncation_custom_limit(self):
        """Test truncation with custom max_chars."""
        output = "b" * 1500
        result = _truncate_output(output, max_chars=1000)

        # Should be truncated
        assert len(result) < len(output)
        assert "ELIDED 500 CHARACTERS" in result

        # Should have 500 chars from start and 500 from end
        assert result.startswith("b" * 500)
        assert result.endswith("b" * 500)

    def test_truncation_preserves_content(self):
        """Test that truncation preserves first and last portions correctly."""
        # Create output with distinctive head, middle, and tail
        # Total needs to be > 10000, with enough middle to ensure some is elided
        head = "A" * 5000  # First 5000 chars
        middle = "B" * 10000  # Middle 10000 chars (will be elided)
        tail = "C" * 5000  # Last 5000 chars
        output = head + middle + tail  # 20000 chars total

        result = _truncate_output(output)

        # Should start with the head (all A's)
        assert result.startswith("A" * 5000)

        # Should end with the tail (all C's)
        assert result.endswith("C" * 5000)

        # The middle section should be mostly elided (some might appear at boundaries)
        # But a large chunk of B's should definitely not be present
        assert "B" * 1000 not in result


class TestReadFile:
    """Tests for the read_file tool."""

    @pytest.mark.anyio
    async def test_read_file_success(self, mock_context):
        """Test reading a file successfully."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="     1\tline 1\n     2\tline 2\n     3\tline 3",
                stderr=None,
                exit_code=0,
            )
        )
        ctx = mock_context(mock_exec)

        content = await read_file(ctx, "/path/to/file.txt")
        assert "line 1" in content
        assert "line 2" in content
        assert "line 3" in content

    @pytest.mark.anyio
    async def test_read_file_with_line_range(self, mock_context):
        """Test reading a file with line range."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "sed -n '2,3p'" in cmd
            return create_exec_output(
                stdout="     2\tline 2\n     3\tline 3",
                stderr=None,
                exit_code=0,
            )

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        content = await read_file(ctx, "/path/to/file.txt", start_line=2, end_line=3)
        assert "line 2" in content
        assert "line 3" in content

    @pytest.mark.anyio
    async def test_read_file_with_start_line_only(self, mock_context):
        """Test reading a file from start line to end."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "sed -n '5,$p'" in cmd
            return create_exec_output(
                stdout="     5\tline 5\n     6\tline 6",
                stderr=None,
                exit_code=0,
            )

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        content = await read_file(ctx, "/path/to/file.txt", start_line=5)
        assert "line 5" in content

    @pytest.mark.anyio
    async def test_read_file_with_end_line_only(self, mock_context):
        """Test reading a file from beginning to end line."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "sed -n '1,10p'" in cmd
            return create_exec_output(stdout="     1\tline 1", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        content = await read_file(ctx, "/path/to/file.txt", end_line=10)
        assert "line 1" in content

    @pytest.mark.anyio
    async def test_read_file_not_found(self, mock_context):
        """Test reading a non-existent file."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="",
                stderr="cat: /path/to/file.txt: No such file or directory",
                exit_code=1,
            )
        )
        ctx = mock_context(mock_exec)

        with pytest.raises(ToolFileNotFoundError, match="File not found"):
            await read_file(ctx, "/path/to/file.txt")

    @pytest.mark.anyio
    async def test_read_file_other_error(self, mock_context):
        """Test reading a file with a generic error."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(stdout="", stderr="some other error", exit_code=1)
        )
        ctx = mock_context(mock_exec)

        with pytest.raises(ToolExecutionError, match="Failed to read file"):
            await read_file(ctx, "/path/to/file.txt")


class TestWriteFile:
    """Tests for the write_file tool."""

    @pytest.mark.anyio
    async def test_write_file_success(self, mock_context):
        """Test writing a file successfully."""
        calls = []

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            calls.append(cmd)
            return create_exec_output(stdout="", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await write_file(ctx, "/path/to/file.txt", "content here")
        assert "Successfully wrote" in result
        # Should have mkdir and cat commands
        assert len(calls) == 2
        assert "mkdir -p" in calls[0]
        assert "cat >" in calls[1]

    @pytest.mark.anyio
    async def test_write_file_without_create_dirs(self, mock_context):
        """Test writing a file without creating directories."""
        calls = []

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            calls.append(cmd)
            return create_exec_output(stdout="", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await write_file(ctx, "/path/to/file.txt", "content", create_dirs=False)
        assert "Successfully wrote" in result
        # Should only have cat command
        assert len(calls) == 1
        assert "cat >" in calls[0]

    @pytest.mark.anyio
    async def test_write_file_mkdir_fails(self, mock_context):
        """Test write_file when mkdir fails."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            if "mkdir" in cmd:
                return create_exec_output(stdout="", stderr="mkdir failed", exit_code=1)
            return create_exec_output(stdout="", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        with pytest.raises(ToolExecutionError, match="Failed to create parent directories"):
            await write_file(ctx, "/path/to/file.txt", "content")

    @pytest.mark.anyio
    async def test_write_file_permission_error(self, mock_context):
        """Test write_file with permission denied."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            if "mkdir" in cmd:
                return create_exec_output(stdout="", stderr=None, exit_code=0)
            return create_exec_output(stdout="", stderr="Permission denied", exit_code=1)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        with pytest.raises(ToolPermissionError, match="Permission denied"):
            await write_file(ctx, "/path/to/file.txt", "content")

    @pytest.mark.anyio
    async def test_write_file_other_error(self, mock_context):
        """Test write_file with a generic error."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            if "mkdir" in cmd:
                return create_exec_output(stdout="", stderr=None, exit_code=0)
            return create_exec_output(stdout="", stderr="some error", exit_code=1)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        with pytest.raises(ToolExecutionError, match="Failed to write file"):
            await write_file(ctx, "/path/to/file.txt", "content")


class TestListDirectory:
    """Tests for the list_directory tool."""

    @pytest.mark.anyio
    async def test_list_directory_success(self, mock_context):
        """Test listing a directory successfully."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="file1.txt\nfile2.txt\nsubdir", stderr=None, exit_code=0
            )
        )
        ctx = mock_context(mock_exec)

        files = await list_directory(ctx, "/path/to/dir")
        assert files == ["file1.txt", "file2.txt", "subdir"]

    @pytest.mark.anyio
    async def test_list_directory_empty(self, mock_context):
        """Test listing an empty directory."""
        mock_exec = AsyncMock(return_value=create_exec_output(stdout="", stderr=None, exit_code=0))
        ctx = mock_context(mock_exec)

        files = await list_directory(ctx, "/path/to/dir")
        assert files == []

    @pytest.mark.anyio
    async def test_list_directory_with_pattern(self, mock_context):
        """Test listing a directory with pattern filter."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "*.py" in cmd
            return create_exec_output(stdout="file1.py\nfile2.py", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await list_directory(ctx, "/path/to/dir", pattern="*.py")
        assert files == ["file1.py", "file2.py"]

    @pytest.mark.anyio
    async def test_list_directory_recursive(self, mock_context):
        """Test listing a directory recursively."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "find" in cmd
            return create_exec_output(
                stdout="/path/to/dir\n/path/to/dir/file1.txt\n/path/to/dir/subdir",
                stderr=None,
                exit_code=0,
            )

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await list_directory(ctx, "/path/to/dir", recursive=True)
        assert len(files) == 3

    @pytest.mark.anyio
    async def test_list_directory_recursive_with_pattern(self, mock_context):
        """Test listing a directory recursively with pattern."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "find" in cmd
            assert "*.py" in cmd
            return create_exec_output(stdout="/path/to/dir/file1.py", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await list_directory(ctx, "/path/to/dir", recursive=True, pattern="*.py")
        assert files == ["/path/to/dir/file1.py"]

    @pytest.mark.anyio
    async def test_list_directory_not_found(self, mock_context):
        """Test listing a non-existent directory."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="",
                stderr="ls: /path/to/dir: No such file or directory",
                exit_code=2,
            )
        )
        ctx = mock_context(mock_exec)

        with pytest.raises(ToolDirectoryNotFoundError, match="Directory not found"):
            await list_directory(ctx, "/path/to/dir")

    @pytest.mark.anyio
    async def test_list_directory_pattern_no_match(self, mock_context):
        """Test listing directory when pattern doesn't match anything."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="",
                stderr="ls: /path/to/dir/*.xyz: No such file or directory",
                exit_code=2,
            )
        )
        ctx = mock_context(mock_exec)

        files = await list_directory(ctx, "/path/to/dir", pattern="*.xyz")
        assert files == []


class TestFindFiles:
    """Tests for the find_files tool."""

    @pytest.mark.anyio
    async def test_find_files_success(self, mock_context):
        """Test finding files successfully."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="./file1.py\n./subdir/file2.py", stderr=None, exit_code=0
            )
        )
        ctx = mock_context(mock_exec)

        files = await find_files(ctx, "*.py")
        assert files == ["./file1.py", "./subdir/file2.py"]

    @pytest.mark.anyio
    async def test_find_files_no_matches(self, mock_context):
        """Test finding files with no matches."""
        mock_exec = AsyncMock(return_value=create_exec_output(stdout="", stderr=None, exit_code=0))
        ctx = mock_context(mock_exec)

        files = await find_files(ctx, "*.xyz")
        assert files == []

    @pytest.mark.anyio
    async def test_find_files_with_max_depth(self, mock_context):
        """Test finding files with max depth."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "-maxdepth 2" in cmd
            return create_exec_output(stdout="./file1.py", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await find_files(ctx, "*.py", max_depth=2)
        assert files == ["./file1.py"]

    @pytest.mark.anyio
    async def test_find_files_with_file_type(self, mock_context):
        """Test finding files with type filter."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "-type f" in cmd
            return create_exec_output(stdout="./file1.py", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await find_files(ctx, "*.py", file_type="f")
        assert files == ["./file1.py"]

    @pytest.mark.anyio
    async def test_find_files_with_path(self, mock_context):
        """Test finding files in a specific path."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "find /custom/path" in cmd
            return create_exec_output(stdout="/custom/path/file.py", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        files = await find_files(ctx, "*.py", path="/custom/path")
        assert files == ["/custom/path/file.py"]

    @pytest.mark.anyio
    async def test_find_files_error(self, mock_context):
        """Test find_files with an error."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="",
                stderr="find: '/path': No such file or directory",
                exit_code=1,
            )
        )
        ctx = mock_context(mock_exec)

        with pytest.raises(ToolExecutionError, match="Failed to find files"):
            await find_files(ctx, "*.py", path="/path")


class TestSearchFiles:
    """Tests for the search_files tool."""

    @pytest.mark.anyio
    async def test_search_files_success(self, mock_context):
        """Test searching files successfully."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="file1.py:10:def function_name():\nfile2.py:25:def function_name():",
                stderr=None,
                exit_code=0,
            )
        )
        ctx = mock_context(mock_exec)

        result = await search_files(ctx, "function_name")
        assert "file1.py:10" in result
        assert "file2.py:25" in result

    @pytest.mark.anyio
    async def test_search_files_no_matches(self, mock_context):
        """Test searching files with no matches."""
        mock_exec = AsyncMock(return_value=create_exec_output(stdout="", stderr=None, exit_code=1))
        ctx = mock_context(mock_exec)

        result = await search_files(ctx, "nonexistent")
        assert result == ""

    @pytest.mark.anyio
    async def test_search_files_case_insensitive(self, mock_context):
        """Test searching files case-insensitively."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "-i" in cmd
            return create_exec_output(stdout="file.py:1:MATCH", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await search_files(ctx, "match", case_sensitive=False)
        assert "MATCH" in result

    @pytest.mark.anyio
    async def test_search_files_with_context(self, mock_context):
        """Test searching files with context lines."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "-C 2" in cmd
            return create_exec_output(
                stdout="file.py-8-before\nfile.py:10:match\nfile.py-12-after",
                stderr=None,
                exit_code=0,
            )

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await search_files(ctx, "match", context_lines=2)
        assert "before" in result
        assert "match" in result
        assert "after" in result

    @pytest.mark.anyio
    async def test_search_files_with_file_pattern(self, mock_context):
        """Test searching files with file pattern filter."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            # Should use find with -name and -exec grep
            assert "find" in cmd
            assert "-name '*.py'" in cmd
            assert "-exec grep" in cmd
            return create_exec_output(stdout="file.py:1:match", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await search_files(ctx, "match", file_pattern="*.py")
        assert "file.py" in result

    @pytest.mark.anyio
    async def test_search_files_with_max_results(self, mock_context):
        """Test searching files with max results limit."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "| head -n 10" in cmd
            return create_exec_output(stdout="file.py:1:match", stderr=None, exit_code=0)

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await search_files(ctx, "match", max_results=10)
        assert "match" in result

    @pytest.mark.anyio
    async def test_search_files_with_path(self, mock_context):
        """Test searching files in a specific path."""

        async def mock_exec_fn(cmd: str | list[str], timeout: int | None = None) -> ExecOutput:
            assert "/custom/path" in cmd
            return create_exec_output(
                stdout="/custom/path/file.py:1:match", stderr=None, exit_code=0
            )

        ctx = mock_context(AsyncMock(side_effect=mock_exec_fn))
        result = await search_files(ctx, "match", path="/custom/path")
        assert "/custom/path/file.py" in result

    @pytest.mark.anyio
    async def test_search_files_error(self, mock_context):
        """Test search_files with an error."""
        mock_exec = AsyncMock(
            return_value=create_exec_output(
                stdout="",
                stderr="grep: /path: No such file or directory",
                exit_code=2,
            )
        )
        ctx = mock_context(mock_exec)

        with pytest.raises(ToolExecutionError, match="Failed to search files"):
            await search_files(ctx, "pattern", path="/path")

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for SWEBench dataset tools with real Docker.

These tests use a real Docker daemon and are slower than unit tests.
Run with: pytest -vx -m docker_integration
Skip with: pytest -vx -m "not docker_integration"
"""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from prompt_siren.datasets.swebench_dataset.tools import (
    bash,
    find_files,
    list_directory,
    read_file,
    search_files,
    ToolDirectoryNotFoundError,
    ToolExecutionError,
    ToolFileNotFoundError,
    write_file,
)
from prompt_siren.environments.bash_env import BashEnvState
from prompt_siren.sandbox_managers.abstract import AbstractSandboxManager
from prompt_siren.sandbox_managers.docker.manager import (
    DockerSandboxConfig,
    DockerSandboxManager,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_state import SandboxState
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    TaskSetup,
)
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
async def sandbox_manager(
    test_image: str,
) -> AsyncIterator[tuple[AbstractSandboxManager, str, TaskSetup]]:
    """Create a sandbox manager for tool testing.

    Returns tuple of (sandbox_manager, test_image, task_setup) for use by dependent fixtures.
    """
    config = DockerSandboxConfig(network_enabled=False)
    manager = DockerSandboxManager(config)

    # Create TaskSetup for API
    container_spec = ContainerSpec(image_spec=PullImageSpec(tag=test_image))
    agent_container = ContainerSetup(name="agent", spec=container_spec)
    task_setup = TaskSetup(
        task_id="tool-test",
        agent_container=agent_container,
        service_containers={},
        network_config=None,
    )

    # Setup batch context once for all tests
    async with manager.setup_batch([task_setup]):
        yield manager, test_image, task_setup


@pytest.fixture(scope="module")
async def container_id(
    sandbox_manager: tuple[AbstractSandboxManager, str, TaskSetup],
) -> AsyncIterator[str]:
    """Create a container for tool testing.

    Returns the container ID for use by dependent fixtures.
    """
    manager, _test_image, task_setup = sandbox_manager

    # setup_task takes task_setup and returns SandboxState
    async with manager.setup_task(task_setup) as sandbox_state:
        yield sandbox_state.agent_container_id


@pytest.fixture(scope="module")
async def run_context(
    sandbox_manager: tuple[AbstractSandboxManager, str, TaskSetup], container_id: str
) -> RunContext[BashEnvState]:
    """Create a RunContext for tool testing.

    Provides a RunContext with BashEnvDeps for executing tools in the sandbox.
    """
    manager, _, _ = sandbox_manager

    # Create BashEnvDeps with SandboxState
    sandbox_state = SandboxState(
        agent_container_id=container_id,
        service_containers={},
        execution_id="",
        network_id=None,
    )
    env_state = BashEnvState(sandbox_state=sandbox_state, sandbox_manager=manager)

    # Create RunContext
    return RunContext(
        deps=env_state,
        model=TestModel(),
        usage=RunUsage(),
        messages=[],
    )


@pytest.fixture
async def test_tmp_path(
    request: pytest.FixtureRequest, run_context: RunContext[BashEnvState]
) -> Path:
    """Generate a unique temporary path for each test and create the directory.

    Returns a unique /tmp path based on the test name and a UUID to ensure isolation.
    The directory is automatically created in the container.
    """
    # Sanitize test name for use in path
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
    # Generate unique path with UUID to avoid conflicts
    unique_id = uuid4().hex[:8]
    path = Path("/tmp") / f"test_{test_name}_{unique_id}"

    # Create the directory in the container
    await bash(run_context, ["mkdir", "-p", str(path)])

    return path


@pytest.mark.docker_integration
class TestSWEBenchToolsIntegration:
    """Integration tests for SWEBench tools using real Docker containers."""

    async def test_bash_tool_success(self, run_context: RunContext[BashEnvState]):
        """Test bash tool with a successful command."""
        result = await bash(run_context, "echo 'Hello from Docker'")
        assert "Hello from Docker" in result

    async def test_bash_tool_failure(self, run_context: RunContext[BashEnvState]):
        """Test bash tool with a failing command."""
        result = await bash(run_context, "ls /nonexistent")
        assert "no such file or directory" in result.lower()

    async def test_bash_tool_with_list_command(self, run_context: RunContext[BashEnvState]):
        """Test bash tool with command as list."""
        result = await bash(run_context, ["echo", "test with spaces"])
        assert "test with spaces" in result

    async def test_write_and_read_file(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test writing a file and then reading it back."""
        # Write a file
        write_result = await write_file(
            run_context, f"{test_tmp_path}/test_file.txt", "Hello, Docker!"
        )
        assert "Successfully wrote" in write_result

        # Read it back
        content = await read_file(run_context, f"{test_tmp_path}/test_file.txt")
        assert "Hello, Docker!" in content

    async def test_read_file_with_line_range(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test reading a file with specific line range."""
        # Create a multi-line file
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        await write_file(run_context, f"{test_tmp_path}/multiline.txt", content)

        # Read lines 2-4
        partial_content = await read_file(
            run_context,
            f"{test_tmp_path}/multiline.txt",
            start_line=2,
            end_line=4,
        )
        assert "Line 2" in partial_content
        assert "Line 3" in partial_content
        assert "Line 4" in partial_content
        assert "Line 1" not in partial_content
        assert "Line 5" not in partial_content

    async def test_read_nonexistent_file(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test reading a file that doesn't exist."""
        with pytest.raises(ToolFileNotFoundError, match="File not found"):
            await read_file(run_context, f"{test_tmp_path}/nonexistent_file.txt")

    async def test_write_file_creates_directories(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that write_file creates parent directories."""
        # Write to a nested path
        await write_file(run_context, f"{test_tmp_path}/nested/dir/test.txt", "content")

        # Verify file exists
        content = await read_file(run_context, f"{test_tmp_path}/nested/dir/test.txt")
        assert "content" in content

    async def test_list_directory(self, run_context: RunContext[BashEnvState], test_tmp_path: Path):
        """Test listing directory contents."""
        # Create some files
        await write_file(run_context, f"{test_tmp_path}/test_dir/file1.txt", "content1")
        await write_file(run_context, f"{test_tmp_path}/test_dir/file2.txt", "content2")
        await write_file(run_context, f"{test_tmp_path}/test_dir/file3.log", "content3")

        # List directory
        files = await list_directory(run_context, f"{test_tmp_path}/test_dir")
        assert len(files) == 3
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert "file3.log" in files

    async def test_list_directory_with_pattern(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test listing directory with pattern filter."""
        # Create mixed files
        await write_file(run_context, f"{test_tmp_path}/pattern_dir/test1.txt", "content1")
        await write_file(run_context, f"{test_tmp_path}/pattern_dir/test2.txt", "content2")
        await write_file(run_context, f"{test_tmp_path}/pattern_dir/test3.log", "content3")

        # List only .txt files
        files = await list_directory(run_context, f"{test_tmp_path}/pattern_dir", pattern="*.txt")
        assert len(files) == 2
        assert any("test1.txt" in f for f in files)
        assert any("test2.txt" in f for f in files)
        assert not any("test3.log" in f for f in files)

    async def test_list_directory_recursive(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test listing directory recursively."""
        # Create nested structure
        await write_file(run_context, f"{test_tmp_path}/recursive/file1.txt", "content1")
        await write_file(
            run_context,
            f"{test_tmp_path}/recursive/subdir/file2.txt",
            "content2",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/recursive/subdir/nested/file3.txt",
            "content3",
        )

        # List recursively
        files = await list_directory(run_context, f"{test_tmp_path}/recursive", recursive=True)
        assert len(files) >= 4  # At least the directories and files
        # Check that we found files at different levels
        txt_files = [f for f in files if f.endswith(".txt")]
        assert len(txt_files) == 3

    async def test_list_nonexistent_directory(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test listing a directory that doesn't exist."""
        with pytest.raises(ToolDirectoryNotFoundError, match="Directory not found"):
            await list_directory(run_context, f"{test_tmp_path}/nonexistent_directory")

    async def test_find_files(self, run_context: RunContext[BashEnvState], test_tmp_path: Path):
        """Test finding files by pattern."""
        # Create test files
        await write_file(run_context, f"{test_tmp_path}/find_test/file1.py", "python code")
        await write_file(run_context, f"{test_tmp_path}/find_test/file2.py", "more python")
        await write_file(run_context, f"{test_tmp_path}/find_test/file3.txt", "text file")
        await write_file(
            run_context,
            f"{test_tmp_path}/find_test/subdir/file4.py",
            "nested python",
        )

        # Find all .py files
        py_files = await find_files(run_context, "*.py", path=f"{test_tmp_path}/find_test")
        assert len(py_files) == 3
        assert all(f.endswith(".py") for f in py_files)

    async def test_find_files_with_max_depth(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test finding files with depth limit."""
        # Create nested structure
        await write_file(run_context, f"{test_tmp_path}/depth_test/file1.txt", "content1")
        await write_file(
            run_context,
            f"{test_tmp_path}/depth_test/subdir/file2.txt",
            "content2",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/depth_test/subdir/nested/file3.txt",
            "content3",
        )

        # Find with max_depth=2 (should not find file3.txt)
        files = await find_files(
            run_context,
            "*.txt",
            path=f"{test_tmp_path}/depth_test",
            max_depth=2,
        )
        assert len(files) == 2
        assert any("file1.txt" in f for f in files)
        assert any("file2.txt" in f for f in files)
        assert not any("file3.txt" in f for f in files)

    async def test_find_files_with_type_filter(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test finding only files (not directories)."""
        # Create files and directories
        await write_file(run_context, f"{test_tmp_path}/type_test/file.txt", "content")
        await write_file(
            run_context,
            f"{test_tmp_path}/type_test/subdir/file2.txt",
            "content2",
        )

        # Find only files
        files = await find_files(run_context, "*", path=f"{test_tmp_path}/type_test", file_type="f")
        # Should not include directories
        assert all("file" in f.lower() or "txt" in f for f in files)

    async def test_search_files(self, run_context: RunContext[BashEnvState], test_tmp_path: Path):
        """Test searching for text in files."""
        # Create files with searchable content
        await write_file(
            run_context,
            f"{test_tmp_path}/search_test/file1.txt",
            "This contains SEARCH_TERM in it",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/search_test/file2.txt",
            "This has SEARCH_TERM too",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/search_test/file3.txt",
            "This doesn't have the term",
        )

        # Search for the term
        results = await search_files(
            run_context, "SEARCH_TERM", path=f"{test_tmp_path}/search_test"
        )
        assert "SEARCH_TERM" in results
        assert "file1.txt" in results or f"{test_tmp_path}/search_test/file1.txt" in results
        assert "file2.txt" in results or f"{test_tmp_path}/search_test/file2.txt" in results
        assert "file3.txt" not in results

    async def test_search_files_case_insensitive(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test case-insensitive search."""
        await write_file(
            run_context,
            f"{test_tmp_path}/case_test/file.txt",
            "This has MiXeD CaSe text",
        )

        # Search case-insensitively
        results = await search_files(
            run_context,
            "mixed case",
            path=f"{test_tmp_path}/case_test",
            case_sensitive=False,
        )
        assert "MiXeD CaSe" in results

    async def test_search_files_with_context(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test searching with context lines."""
        # Create file with multiple lines
        content = "Line 1\nLine 2\nLINE WITH MATCH\nLine 4\nLine 5"
        await write_file(run_context, f"{test_tmp_path}/context_test/file.txt", content)

        # Search with context
        results = await search_files(
            run_context,
            "MATCH",
            path=f"{test_tmp_path}/context_test",
            context_lines=1,
        )
        # Should include surrounding lines
        assert "Line 2" in results or "Line 4" in results

    async def test_search_files_with_file_pattern(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test searching only in files matching pattern."""
        await write_file(
            run_context,
            f"{test_tmp_path}/pattern_search/file.py",
            "def function(): pass",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/pattern_search/file.txt",
            "def function in text",
        )

        # Search only in .py files
        results = await search_files(
            run_context,
            "def function",
            path=f"{test_tmp_path}/pattern_search",
            file_pattern="*.py",
        )
        assert "file.py" in results
        assert "file.txt" not in results

    async def test_end_to_end_workflow(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """End-to-end test: create files, search, read, list."""
        # Setup: Create a small project structure
        await write_file(
            run_context,
            f"{test_tmp_path}/project/src/main.py",
            "def main():\n    print('Hello')",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/project/src/utils.py",
            "def helper():\n    return True",
        )
        await write_file(
            run_context,
            f"{test_tmp_path}/project/tests/test_main.py",
            "def test_main():\n    assert True",
        )
        await write_file(run_context, f"{test_tmp_path}/project/README.md", "# My Project")

        # List all files
        all_files = await list_directory(run_context, f"{test_tmp_path}/project", recursive=True)
        py_files = [f for f in all_files if f.endswith(".py")]
        assert len(py_files) == 3

        # Find Python files
        found_py = await find_files(run_context, "*.py", path=f"{test_tmp_path}/project")
        assert len(found_py) == 3

        # Search for "def" in Python files
        search_results = await search_files(
            run_context,
            "def ",
            path=f"{test_tmp_path}/project",
            file_pattern="*.py",
        )
        assert "main.py" in search_results
        assert "utils.py" in search_results
        assert "test_main.py" in search_results

        # Read a specific file
        main_content = await read_file(run_context, f"{test_tmp_path}/project/src/main.py")
        assert "def main" in main_content
        assert "print('Hello')" in main_content


@pytest.mark.docker_integration
class TestSWEBenchToolsSecurity:
    """Security tests for SWE-bench tools to verify proper escaping and injection prevention.

    These tests verify that file paths, patterns, and other user inputs are properly
    escaped to prevent command injection vulnerabilities. Each test uses a unique
    directory path to avoid conflicts since all tests share the same container.

    All tests verify that:
    - Filenames with spaces, quotes, and special characters work correctly
    - Command injection attempts (semicolons, pipes, command substitution) are safely escaped
    - Pattern breakouts in find/grep are prevented
    - The shell features (pipes, redirections) continue to work as needed
    """

    # ==================== read_file Tests ====================

    async def test_read_file_spaces_in_filename(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that filenames with spaces are handled correctly."""
        # Create file with spaces using bash (which handles escaping properly)
        await bash(
            run_context,
            [
                "sh",
                "-c",
                f"echo 'test content' > '{test_tmp_path}/my test file.txt'",
            ],
        )

        # Verify file was created
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id,
            ["test", "-f", f"{test_tmp_path}/my test file.txt"],
        )
        assert check.exit_code == 0, "Setup failed: file with spaces not created"

        content = await read_file(run_context, f"{test_tmp_path}/my test file.txt")
        assert "test content" in content

    async def test_read_file_single_quotes_in_filename(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that filenames with single quotes are handled correctly."""
        await bash(
            run_context,
            [
                "sh",
                "-c",
                f"echo 'test content' > \"{test_tmp_path}/file'with'quotes.txt\"",
            ],
        )

        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id,
            ["test", "-f", f"{test_tmp_path}/file'with'quotes.txt"],
        )
        assert check.exit_code == 0, "Setup failed"

        content = await read_file(run_context, f"{test_tmp_path}/file'with'quotes.txt")
        assert "test content" in content

    async def test_read_file_double_quotes_in_filename(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that filenames with double quotes are handled correctly."""
        await bash(
            run_context,
            [
                "sh",
                "-c",
                f"""echo 'test content' > '{test_tmp_path}/file"with"quotes.txt'""",
            ],
        )

        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id,
            ["test", "-f", f'{test_tmp_path}/file"with"quotes.txt'],
        )
        assert check.exit_code == 0, "Setup failed"

        content = await read_file(run_context, f'{test_tmp_path}/file"with"quotes.txt')
        assert "test content" in content

    async def test_read_file_command_injection_semicolon(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command injection via semicolon is prevented."""
        await bash(run_context, ["touch", f"{test_tmp_path}/safe.txt"])

        # Attempt command injection that would create a canary file
        malicious_path = f"{test_tmp_path}/safe.txt; touch {test_tmp_path}/canary.txt #"

        # This should fail gracefully without executing the injection
        with pytest.raises(ToolFileNotFoundError):
            await read_file(run_context, malicious_path)

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection occurred - canary file exists!"

    async def test_read_file_command_substitution_dollar(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command substitution via $() is prevented."""
        # Attempt command substitution that would create a canary
        malicious_path = f"{test_tmp_path}/$(touch {test_tmp_path}/canary.txt).txt"

        # Should fail without executing the substitution
        with pytest.raises(ToolFileNotFoundError):
            await read_file(run_context, malicious_path)

        # Verify substitution didn't execute
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command substitution occurred - canary file exists!"

    async def test_read_file_command_substitution_backtick(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command substitution via backticks is prevented."""
        # Attempt command substitution using backticks
        malicious_path = f"{test_tmp_path}/`touch {test_tmp_path}/canary.txt`.txt"

        # Should fail without executing the substitution
        with pytest.raises(ToolFileNotFoundError):
            await read_file(run_context, malicious_path)

        # Verify substitution didn't execute
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command substitution occurred - canary file exists!"

    # ==================== write_file Tests ====================

    async def test_write_file_spaces_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that file paths with spaces are handled correctly."""
        # This should work but currently fails
        result = await write_file(run_context, f"{test_tmp_path}/my test file.txt", "test content")
        assert "Successfully wrote" in result

        # Verify file was created correctly
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id,
            ["test", "-f", f"{test_tmp_path}/my test file.txt"],
        )
        assert check.exit_code == 0, "File with spaces was not created"

        # Verify content
        content = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["cat", f"{test_tmp_path}/my test file.txt"]
        )
        assert content.exit_code == 0
        assert "test content" in (content.stdout or "")

    async def test_write_file_quotes_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that file paths with quotes are handled correctly."""
        # Test with single quotes
        result = await write_file(
            run_context, f"{test_tmp_path}/file'with'quotes.txt", "test content"
        )
        assert "Successfully wrote" in result

        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id,
            ["test", "-f", f"{test_tmp_path}/file'with'quotes.txt"],
        )
        assert check.exit_code == 0, "File with single quotes was not created"

    async def test_write_file_command_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command injection in file path is prevented."""
        # Attempt injection that would create a canary file
        malicious_path = f"{test_tmp_path}/file.txt; touch {test_tmp_path}/canary.txt #"

        # With proper escaping, the function should succeed safely
        # The semicolon and command become literal parts of the filename
        result = await write_file(run_context, malicious_path, "content")
        assert "Successfully wrote" in result

        # Verify injection didn't happen - canary file should NOT exist
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection occurred - canary file exists!"

    async def test_write_file_content_with_special_chars(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that content with special characters is handled safely (heredoc should protect this)."""
        # Content with special characters - heredoc should handle this correctly
        special_content = (
            "Line with $var\nLine with `cmd`\nLine with $(subshell)\nLine with 'quotes'"
        )

        result = await write_file(run_context, f"{test_tmp_path}/special.txt", special_content)
        assert "Successfully wrote" in result

        # Verify content is written literally (not interpreted)
        content = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["cat", f"{test_tmp_path}/special.txt"]
        )
        assert content.exit_code == 0
        # These should be literal, not executed
        assert "$var" in (content.stdout or "")
        assert "`cmd`" in (content.stdout or "")
        assert "$(subshell)" in (content.stdout or "")

    async def test_write_file_dirname_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command injection in dirname is prevented.

        write_file uses $(dirname {path}), which means the path is evaluated twice.
        This is particularly dangerous.
        """
        # Path that attempts injection through the dirname call
        # Note: We now use os.path.dirname in Python to avoid shell evaluation
        malicious_path = f"{test_tmp_path}/$(touch {test_tmp_path}/canary.txt)file.txt"

        # With proper escaping and Python dirname, the function should succeed safely
        # The command substitution becomes a literal part of the filename
        result = await write_file(run_context, malicious_path, "content")
        assert "Successfully wrote" in result

        # Verify injection didn't happen - canary file should NOT exist
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Canary file exists!"

    async def test_write_file_deeply_nested_special_chars(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test deeply nested paths with special characters."""
        # Path with spaces in nested directories
        nested_path = f"{test_tmp_path}/my dir/sub dir/file with spaces.txt"

        result = await write_file(run_context, nested_path, "content", create_dirs=True)
        assert "Successfully wrote" in result

        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", nested_path]
        )
        assert check.exit_code == 0, "Nested file with spaces not created"

    async def test_write_file_no_create_dirs_with_special_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test write_file with create_dirs=False and special characters."""
        # Create parent directory manually
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/existing dir"])

        # This should work even though parent has spaces (and we're not creating it)
        with pytest.raises(ToolExecutionError):
            # Should fail because we're trying to write without creating the parent
            await write_file(
                run_context,
                f"{test_tmp_path}/nonexistent/file.txt",
                "content",
                create_dirs=False,
            )

    # ==================== list_directory Tests ====================

    async def test_list_directory_spaces_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that directory paths with spaces are handled correctly."""
        # Create directory with spaces
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/my directory"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my directory/file1.txt"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my directory/file2.txt"])

        # Should list contents correctly
        files = await list_directory(run_context, f"{test_tmp_path}/my directory")
        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files

    async def test_list_directory_quotes_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that directory paths with quotes are handled correctly."""
        # Create directory with single quotes
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/dir'with'quotes"])
        await bash(run_context, ["touch", f"{test_tmp_path}/dir'with'quotes/file.txt"])

        files = await list_directory(run_context, f"{test_tmp_path}/dir'with'quotes")
        assert "file.txt" in files

    async def test_list_directory_command_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command injection in directory path is prevented."""
        # Attempt injection
        malicious_path = f"{test_tmp_path}; touch {test_tmp_path}/canary.txt #"

        # Should fail without executing injection
        with pytest.raises(ToolDirectoryNotFoundError):
            await list_directory(run_context, malicious_path)

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection occurred - canary file exists!"

    async def test_list_directory_pattern_quote_breakout(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that patterns cannot break out of quotes."""
        await bash(run_context, ["touch", f"{test_tmp_path}/test.txt"])
        await bash(run_context, ["touch", f"{test_tmp_path}/test.py"])

        # Pattern that tries to break out and match everything
        # Note: This is more about incorrect results than injection
        malicious_pattern = "*.txt' -o -name '*"

        # Should only match *.txt files, not everything
        files = await list_directory(run_context, f"{test_tmp_path}", pattern=malicious_pattern)

        # The breakout shouldn't work - we should get an error or no results
        # (exact behavior depends on implementation)
        assert "test.py" not in files or len(files) == 0

    async def test_list_directory_pattern_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that pattern cannot be used for command injection."""
        await bash(run_context, ["touch", f"{test_tmp_path}/test.txt"])

        # Pattern that attempts command injection
        malicious_pattern = f"*.txt'; touch {test_tmp_path}/canary.txt; echo '"

        # Should fail or not execute injection
        try:
            await list_directory(run_context, f"{test_tmp_path}", pattern=malicious_pattern)
        except (ToolDirectoryNotFoundError, ToolExecutionError):
            pass  # Expected to fail

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection via pattern occurred!"

    async def test_list_directory_recursive_special_chars(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test recursive listing of directories with special characters."""
        # Create nested structure with spaces
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/my dir/sub dir"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my dir/file1.txt"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my dir/sub dir/file2.txt"])

        # Should list recursively
        files = await list_directory(run_context, f"{test_tmp_path}/my dir", recursive=True)

        # Should find both files
        txt_files = [f for f in files if "file" in f and f.endswith(".txt")]
        assert len(txt_files) >= 2

    # ==================== find_files Tests ====================

    async def test_find_files_spaces_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that search paths with spaces are handled correctly."""
        # Create directory with spaces
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/my directory"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my directory/test.py"])
        await bash(run_context, ["touch", f"{test_tmp_path}/my directory/test.txt"])

        # Should find files in directory with spaces
        py_files = await find_files(run_context, "*.py", path=f"{test_tmp_path}/my directory")
        assert len(py_files) == 1
        assert any("test.py" in f for f in py_files)

    async def test_find_files_command_injection_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that command injection in path is prevented."""
        # Attempt injection via path
        malicious_path = f"{test_tmp_path}; touch {test_tmp_path}/canary.txt #"

        # Should fail without executing injection
        with pytest.raises(ToolExecutionError):
            await find_files(run_context, "*.txt", path=malicious_path)

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection via path occurred!"

    async def test_find_files_pattern_breakout(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that patterns cannot break out of quotes to access unintended files."""
        await bash(run_context, ["touch", f"{test_tmp_path}/test.py"])
        await bash(run_context, ["touch", f"{test_tmp_path}/secret.txt"])

        # Pattern that tries to break out and match all files
        malicious_pattern = "*.py' -o -name '*.txt"

        # Should only find .py files, not break out to find .txt files
        files = await find_files(run_context, malicious_pattern, path=f"{test_tmp_path}")

        # If properly escaped, this should either fail or not find secret.txt
        # The exact behavior depends on how the breakout is handled
        if len(files) > 0:
            assert not any("secret.txt" in f for f in files), "Pattern breakout succeeded!"

    async def test_find_files_pattern_exec_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that patterns cannot inject -exec to run commands."""
        await bash(run_context, ["touch", f"{test_tmp_path}/test.txt"])

        # Pattern that tries to inject -exec to create a canary file
        malicious_pattern = f"*.txt' -o -exec touch {test_tmp_path}/canary.txt ';' -o -name '"

        # Should fail without executing the injection
        try:
            await find_files(run_context, malicious_pattern, path=f"{test_tmp_path}")
        except ToolExecutionError:
            pass  # Expected to fail

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection via -exec occurred!"

    async def test_find_files_quotes_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that paths with quotes are handled correctly."""
        # Create directory with single quotes
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/dir'with'quotes"])
        await bash(run_context, ["touch", f"{test_tmp_path}/dir'with'quotes/test.py"])

        # Should find files in directory with quotes
        files = await find_files(run_context, "*.py", path=f"{test_tmp_path}/dir'with'quotes")
        assert len(files) == 1
        assert any("test.py" in f for f in files)

    async def test_find_files_combined_path_pattern_attack(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that combined attacks on both path and pattern are prevented."""
        await bash(run_context, ["touch", f"{test_tmp_path}/test.txt"])

        # Attack both parameters
        malicious_path = f"{test_tmp_path}; touch {test_tmp_path}/canary1.txt #"
        malicious_pattern = f"*.txt' -exec touch {test_tmp_path}/canary2.txt ';' -name '"

        # Should fail without executing either injection
        with pytest.raises(ToolExecutionError):
            await find_files(run_context, malicious_pattern, path=malicious_path)

        # Verify neither injection happened
        check1 = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary1.txt"]
        )
        check2 = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary2.txt"]
        )
        assert check1.exit_code != 0, "Path injection occurred!"
        assert check2.exit_code != 0, "Pattern injection occurred!"

    # ==================== search_files Tests ====================

    async def test_search_files_spaces_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that search paths with spaces are handled correctly."""
        # Create directory with spaces
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/my directory"])
        await bash(
            run_context,
            [
                "sh",
                "-c",
                f"echo 'SEARCH_TERM' > '{test_tmp_path}/my directory/test.txt'",
            ],
        )

        # Should find the search term in directory with spaces
        results = await search_files(
            run_context, "SEARCH_TERM", path=f"{test_tmp_path}/my directory"
        )
        assert "SEARCH_TERM" in results
        assert "test.txt" in results

    async def test_search_files_pattern_breakout(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that search patterns cannot break out of quotes to search unintended locations."""
        await bash(
            run_context,
            ["sh", "-c", f"echo 'FINDME' > {test_tmp_path}/test.txt"],
        )

        # Pattern that tries to break out and search a different file
        # This attempts to close the quote and add a new path
        malicious_pattern = "FINDME' /etc/passwd '"

        # Should only search for the literal pattern, not break out
        results = await search_files(run_context, malicious_pattern, path=f"{test_tmp_path}")

        # Should not find /etc/passwd content
        assert "root:" not in results, "Pattern breakout allowed searching /etc/passwd!"

    async def test_search_files_pattern_injection(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that patterns cannot inject commands."""
        await bash(run_context, ["sh", "-c", f"echo 'test' > {test_tmp_path}/test.txt"])

        # Pattern that tries to inject a command
        malicious_pattern = f"test'; touch {test_tmp_path}/canary.txt; echo '"

        # Should fail or not execute injection
        try:
            await search_files(run_context, malicious_pattern, path=f"{test_tmp_path}")
        except ToolExecutionError:
            pass  # May fail, which is fine

        # Verify injection didn't happen
        check = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary.txt"]
        )
        assert check.exit_code != 0, "Command injection via pattern occurred!"

    async def test_search_files_file_pattern_breakout(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that file_pattern cannot break out to inject find options or commands."""
        await bash(
            run_context,
            ["sh", "-c", f"echo 'FINDME' > {test_tmp_path}/test.py"],
        )
        await bash(
            run_context,
            ["sh", "-c", f"echo 'SECRET' > {test_tmp_path}/secret.txt"],
        )

        # file_pattern that tries to break out and execute a command or find all files
        malicious_file_pattern = "*.py' -o -name '*.txt"

        # Should only search .py files, not break out to search .txt files
        results = await search_files(
            run_context,
            "FINDME",
            path=f"{test_tmp_path}",
            file_pattern=malicious_file_pattern,
        )

        # Should find FINDME but not SECRET (if properly escaped)
        if "FINDME" in results:
            # If it found anything, it shouldn't have found the secret
            assert "SECRET" not in results, "file_pattern breakout succeeded!"

    async def test_search_files_quotes_in_path(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that paths with quotes are handled correctly."""
        # Create directory with quotes
        await bash(run_context, ["mkdir", "-p", f"{test_tmp_path}/dir'with'quotes"])
        await bash(
            run_context,
            [
                "sh",
                "-c",
                f"echo 'FINDME' > \"{test_tmp_path}/dir'with'quotes/test.txt\"",
            ],
        )

        # Should find the term in directory with quotes
        results = await search_files(run_context, "FINDME", path=f"{test_tmp_path}/dir'with'quotes")
        assert "FINDME" in results
        assert "test.txt" in results

    async def test_search_files_combined_attack(
        self, run_context: RunContext[BashEnvState], test_tmp_path: Path
    ):
        """Test that combined attacks on multiple parameters are prevented."""
        await bash(run_context, ["sh", "-c", f"echo 'test' > {test_tmp_path}/test.py"])

        # Attack multiple parameters simultaneously
        malicious_path = f"{test_tmp_path}; touch {test_tmp_path}/canary1.txt #"
        malicious_pattern = f"test'; touch {test_tmp_path}/canary2.txt; echo '"
        malicious_file_pattern = f"*.py' -exec touch {test_tmp_path}/canary3.txt ';' -name '"

        # With proper escaping, this should either fail safely or return empty results
        # without executing any of the injections
        try:
            _ = await search_files(
                run_context,
                malicious_pattern,
                path=malicious_path,
                file_pattern=malicious_file_pattern,
            )
            # If it succeeds, it should return empty or safe results
        except ToolExecutionError:
            # It's also acceptable to fail with an error (directory not found, etc.)
            pass

        # Verify none of the injections happened (most important check!)
        check1 = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary1.txt"]
        )
        check2 = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary2.txt"]
        )
        check3 = await run_context.deps.sandbox_manager.exec(
            run_context.deps.agent_container_id, ["test", "-f", f"{test_tmp_path}/canary3.txt"]
        )
        assert check1.exit_code != 0, "Path injection occurred!"
        assert check2.exit_code != 0, "Pattern injection occurred!"
        assert check3.exit_code != 0, "file_pattern injection occurred!"

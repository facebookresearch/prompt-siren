# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for CLI commands."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from prompt_siren.cli import main
from prompt_siren.job import Job, JobConfig
from prompt_siren.job.persistence import _save_config_yaml


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_job_config() -> JobConfig:
    """Create a mock job config for testing resume."""
    return JobConfig(
        job_name="test_job",
        execution_mode="benign",
        created_at=datetime.now(),
        dataset={"type": "mock", "config": {}},
        agent={"type": "plain", "config": {"model": "test"}},
        attack=None,
        execution={"concurrency": 1},
        telemetry={"trace_console": False},
        output={"job_name": "test_job"},
    )


@pytest.fixture
def mock_attack_job_config() -> JobConfig:
    """Create a mock job config with attack for testing resume."""
    return JobConfig(
        job_name="test_attack_job",
        execution_mode="attack",
        created_at=datetime.now(),
        dataset={"type": "mock", "config": {}},
        agent={"type": "plain", "config": {"model": "test"}},
        attack={"type": "template_string", "config": {"attack_template": "test"}},
        execution={"concurrency": 1},
        telemetry={"trace_console": False},
        output={"job_name": "test_attack_job"},
    )


class TestJobsResumeCommand:
    """Tests for the 'jobs resume' CLI command."""

    def test_resume_passes_job_to_benign_experiment(
        self, cli_runner: CliRunner, mock_job_config: JobConfig, tmp_path: Path
    ):
        """Test that 'jobs resume' passes the Job instance to run_benign_experiment."""
        # Create a job directory with config
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        _save_config_yaml(job_dir / "config.yaml", mock_job_config)

        # Track the job parameter passed to run_benign_experiment
        captured_job = None

        async def mock_run_benign(experiment_config, job=None):
            nonlocal captured_job
            captured_job = job
            return {}

        # Patch at the source module where it's defined
        with patch("prompt_siren.hydra_app.run_benign_experiment", mock_run_benign):
            result = cli_runner.invoke(main, ["jobs", "resume", "-p", str(job_dir)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert captured_job is not None, "Job was not passed to run_benign_experiment"
        assert captured_job.job_config.job_name == "test_job"
        assert captured_job.is_resuming is True

    def test_resume_passes_job_to_attack_experiment(
        self, cli_runner: CliRunner, mock_attack_job_config: JobConfig, tmp_path: Path
    ):
        """Test that 'jobs resume' passes the Job instance to run_attack_experiment."""
        # Create a job directory with config
        job_dir = tmp_path / "test_attack_job"
        job_dir.mkdir()
        _save_config_yaml(job_dir / "config.yaml", mock_attack_job_config)

        # Track the job parameter passed to run_attack_experiment
        captured_job = None

        async def mock_run_attack(experiment_config, job=None):
            nonlocal captured_job
            captured_job = job
            return {}

        # Patch at the source module where it's defined
        with patch("prompt_siren.hydra_app.run_attack_experiment", mock_run_attack):
            result = cli_runner.invoke(main, ["jobs", "resume", "-p", str(job_dir)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert captured_job is not None, "Job was not passed to run_attack_experiment"
        assert captured_job.job_config.job_name == "test_attack_job"
        assert captured_job.is_resuming is True

    def test_resume_nonexistent_job_fails(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that resuming a nonexistent job fails with appropriate error."""
        nonexistent_path = tmp_path / "nonexistent"
        # Click's Path(exists=True) will catch this before our code runs
        result = cli_runner.invoke(main, ["jobs", "resume", "-p", str(nonexistent_path)])

        assert result.exit_code != 0

    def test_resume_with_retry_on_error_flag(
        self, cli_runner: CliRunner, mock_job_config: JobConfig, tmp_path: Path
    ):
        """Test that --retry-on-error flag is passed to Job.resume."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        _save_config_yaml(job_dir / "config.yaml", mock_job_config)

        with patch("prompt_siren.hydra_app.run_benign_experiment", AsyncMock(return_value={})):
            with patch.object(Job, "resume", wraps=Job.resume) as mock_resume:
                result = cli_runner.invoke(
                    main,
                    ["jobs", "resume", "-p", str(job_dir), "--retry-on-error", "TimeoutError"],
                )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        mock_resume.assert_called_once()
        call_kwargs = mock_resume.call_args.kwargs
        assert call_kwargs.get("retry_on_errors") == ["TimeoutError"]

    def test_resume_with_include_failed_flag(
        self, cli_runner: CliRunner, mock_job_config: JobConfig, tmp_path: Path
    ):
        """Test that --include-failed flag is passed to Job.resume."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        _save_config_yaml(job_dir / "config.yaml", mock_job_config)

        with patch("prompt_siren.hydra_app.run_benign_experiment", AsyncMock(return_value={})):
            with patch.object(Job, "resume", wraps=Job.resume) as mock_resume:
                result = cli_runner.invoke(
                    main, ["jobs", "resume", "-p", str(job_dir), "--include-failed"]
                )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        mock_resume.assert_called_once()
        call_kwargs = mock_resume.call_args.kwargs
        assert call_kwargs.get("include_failed") is True

    def test_resume_with_execution_override(
        self, cli_runner: CliRunner, mock_job_config: JobConfig, tmp_path: Path
    ):
        """Test that execution overrides are applied on resume."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        _save_config_yaml(job_dir / "config.yaml", mock_job_config)

        captured_job = None

        async def mock_run_benign(experiment_config, job=None):
            nonlocal captured_job
            captured_job = job
            return {}

        with patch("prompt_siren.hydra_app.run_benign_experiment", mock_run_benign):
            result = cli_runner.invoke(
                main, ["jobs", "resume", "-p", str(job_dir), "execution.concurrency=8"]
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert captured_job is not None
        assert captured_job.job_config.execution["concurrency"] == 8


class TestJobsStartCommand:
    """Tests for the 'jobs start' CLI command."""

    def test_start_benign_invokes_hydra(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that 'jobs start benign' invokes the Hydra run flow."""
        with patch("prompt_siren.cli._run_hydra") as mock_run_hydra:
            cli_runner.invoke(
                main,
                [
                    "jobs",
                    "start",
                    "benign",
                    f"--jobs-dir={tmp_path}",
                    "+dataset=agentdojo-workspace",
                ],
            )

        # The command should invoke _run_hydra
        mock_run_hydra.assert_called_once()
        call_kwargs = mock_run_hydra.call_args.kwargs
        assert call_kwargs["execution_mode"] == "benign"

    def test_start_attack_invokes_hydra(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that 'jobs start attack' invokes the Hydra run flow."""
        with patch("prompt_siren.cli._run_hydra") as mock_run_hydra:
            cli_runner.invoke(
                main,
                [
                    "jobs",
                    "start",
                    "attack",
                    f"--jobs-dir={tmp_path}",
                    "+dataset=agentdojo-workspace",
                    "+attack=template_string",
                ],
            )

        mock_run_hydra.assert_called_once()
        call_kwargs = mock_run_hydra.call_args.kwargs
        assert call_kwargs["execution_mode"] == "attack"

    def test_start_with_custom_job_name(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that --job-name is passed through to the job."""
        with patch("prompt_siren.cli._run_hydra") as mock_run_hydra:
            cli_runner.invoke(
                main,
                [
                    "jobs",
                    "start",
                    "benign",
                    "--job-name=my-custom-job",
                    f"--jobs-dir={tmp_path}",
                    "+dataset=agentdojo-workspace",
                ],
            )

        mock_run_hydra.assert_called_once()
        # Check that job_name override was added
        call_kwargs = mock_run_hydra.call_args.kwargs
        overrides = call_kwargs["overrides"]
        assert any("output.job_name=my-custom-job" in o for o in overrides)


class TestRunCommandAliases:
    """Tests for 'run' command aliases."""

    def test_run_benign_is_alias_for_jobs_start_benign(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        """Test that 'run benign' invokes the same flow as 'jobs start benign'."""
        with patch("prompt_siren.cli._run_hydra") as mock_run_hydra:
            cli_runner.invoke(
                main,
                ["run", "benign", f"--jobs-dir={tmp_path}", "+dataset=agentdojo-workspace"],
            )

        mock_run_hydra.assert_called_once()
        call_kwargs = mock_run_hydra.call_args.kwargs
        assert call_kwargs["execution_mode"] == "benign"

    def test_run_attack_is_alias_for_jobs_start_attack(
        self, cli_runner: CliRunner, tmp_path: Path
    ):
        """Test that 'run attack' invokes the same flow as 'jobs start attack'."""
        with patch("prompt_siren.cli._run_hydra") as mock_run_hydra:
            cli_runner.invoke(
                main,
                [
                    "run",
                    "attack",
                    f"--jobs-dir={tmp_path}",
                    "+dataset=agentdojo-workspace",
                    "+attack=template_string",
                ],
            )

        mock_run_hydra.assert_called_once()
        call_kwargs = mock_run_hydra.call_args.kwargs
        assert call_kwargs["execution_mode"] == "attack"


class TestResultsCommand:
    """Tests for the 'results' CLI command."""

    def test_results_with_empty_directory(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that results command handles empty directory gracefully."""
        result = cli_runner.invoke(main, ["results", f"--jobs-dir={tmp_path}"])

        assert result.exit_code == 0
        assert "No results found" in result.output


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_export_creates_files(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that 'config export' creates configuration files."""
        output_dir = tmp_path / "config"
        result = cli_runner.invoke(main, ["config", "export", str(output_dir)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Check that config files were created
        assert (output_dir / "config.yaml").exists()

    def test_config_validate_with_invalid_config(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that 'config validate' fails with invalid configuration."""
        # Create invalid config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("invalid: yaml: content")

        result = cli_runner.invoke(
            main, ["config", "validate", f"--config-dir={config_dir}", "benign"]
        )

        # Should fail validation
        assert result.exit_code != 0

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for sweep tracking and resume functionality."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from prompt_siren.config.experiment_config import (
    AgentConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job.models import (
    CONFIG_FILENAME,
    INDEX_FILENAME,
    JobConfig,
)
from prompt_siren.job.sweep import (
    find_sweep_jobs_by_pattern,
    get_job_status,
    SweepMetadata,
    SweepRegistry,
)


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    """Create a temporary jobs directory."""
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    return jobs


@pytest.fixture
def job_config() -> JobConfig:
    """Create a minimal job config for testing."""
    return JobConfig(
        job_name="test_job",
        execution_mode="attack",
        created_at=datetime.now(),
        dataset=DatasetConfig(type="agentdojo", config={"suite_name": "workspace"}),
        agent=AgentConfig(type="plain", config={"model": "test"}),
        attack=None,
        execution=ExecutionConfig(concurrency=1),
        telemetry=TelemetryConfig(trace_console=False),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


def create_mock_job_dir(
    jobs_dir: Path,
    job_name: str,
    job_config: JobConfig,
    index_entries: list[dict] | None = None,
) -> Path:
    """Create a mock job directory with config and optional index."""
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True)

    # Write config.yaml
    config = job_config.model_copy(update={"job_name": job_name})
    config_path = job_dir / CONFIG_FILENAME
    config_path.write_text(config.model_dump_json())

    # Write index.jsonl if provided
    if index_entries:
        index_path = job_dir / INDEX_FILENAME
        with open(index_path, "w") as f:
            for entry in index_entries:
                f.write(json.dumps(entry) + "\n")

    return job_dir


class TestSweepMetadata:
    """Tests for SweepMetadata model."""

    def test_create_sweep_metadata(self):
        """Test creating a SweepMetadata instance."""
        sweep = SweepMetadata(
            sweep_id="2026-01-31_17-01-04",
            job_names=["job1", "job2"],
            launcher="submitit_slurm",
        )
        assert sweep.sweep_id == "2026-01-31_17-01-04"
        assert len(sweep.job_names) == 2
        assert sweep.launcher == "submitit_slurm"

    def test_sweep_metadata_defaults(self):
        """Test SweepMetadata default values."""
        sweep = SweepMetadata(sweep_id="test")
        assert sweep.job_names == []
        assert sweep.hydra_sweep_dir is None
        assert sweep.launcher is None
        assert sweep.config_name is None

    def test_sweep_metadata_serialization(self):
        """Test SweepMetadata JSON serialization."""
        sweep = SweepMetadata(
            sweep_id="2026-01-31_17-01-04",
            job_names=["job1", "job2"],
        )
        json_str = sweep.model_dump_json()
        loaded = SweepMetadata.model_validate_json(json_str)
        assert loaded.sweep_id == sweep.sweep_id
        assert loaded.job_names == sweep.job_names


class TestSweepRegistry:
    """Tests for SweepRegistry class."""

    def test_register_job_creates_sweep(self, jobs_dir: Path):
        """Test registering a job creates a new sweep."""
        registry = SweepRegistry(jobs_dir)
        registry.register_job(
            sweep_id="2026-01-31_17-01-04",
            job_name="job1",
            launcher="submitit_slurm",
        )

        sweep = registry.get_sweep("2026-01-31_17-01-04")
        assert sweep is not None
        assert sweep.sweep_id == "2026-01-31_17-01-04"
        assert "job1" in sweep.job_names
        assert sweep.launcher == "submitit_slurm"

    def test_register_job_appends_to_existing_sweep(self, jobs_dir: Path):
        """Test registering additional jobs appends to existing sweep."""
        registry = SweepRegistry(jobs_dir)

        registry.register_job(sweep_id="test-sweep", job_name="job1")
        registry.register_job(sweep_id="test-sweep", job_name="job2")
        registry.register_job(sweep_id="test-sweep", job_name="job3")

        sweep = registry.get_sweep("test-sweep")
        assert sweep is not None
        assert len(sweep.job_names) == 3
        assert "job1" in sweep.job_names
        assert "job2" in sweep.job_names
        assert "job3" in sweep.job_names

    def test_register_job_no_duplicates(self, jobs_dir: Path):
        """Test registering the same job twice doesn't create duplicates."""
        registry = SweepRegistry(jobs_dir)

        registry.register_job(sweep_id="test-sweep", job_name="job1")
        registry.register_job(sweep_id="test-sweep", job_name="job1")

        sweep = registry.get_sweep("test-sweep")
        assert sweep is not None
        assert sweep.job_names.count("job1") == 1

    def test_get_sweep_nonexistent(self, jobs_dir: Path):
        """Test getting a non-existent sweep returns None."""
        registry = SweepRegistry(jobs_dir)
        assert registry.get_sweep("nonexistent") is None

    def test_list_sweeps(self, jobs_dir: Path):
        """Test listing all sweeps."""
        registry = SweepRegistry(jobs_dir)

        registry.register_job(sweep_id="sweep1", job_name="job1")
        registry.register_job(sweep_id="sweep2", job_name="job2")

        sweeps = registry.list_sweeps()
        assert len(sweeps) == 2
        sweep_ids = {s.sweep_id for s in sweeps}
        assert "sweep1" in sweep_ids
        assert "sweep2" in sweep_ids

    def test_get_job_dirs(self, jobs_dir: Path, job_config: JobConfig):
        """Test getting job directories for a sweep."""
        registry = SweepRegistry(jobs_dir)

        # Create actual job directories
        create_mock_job_dir(jobs_dir, "job1", job_config)
        create_mock_job_dir(jobs_dir, "job2", job_config)

        # Register them in a sweep
        registry.register_job(sweep_id="test-sweep", job_name="job1")
        registry.register_job(sweep_id="test-sweep", job_name="job2")
        registry.register_job(sweep_id="test-sweep", job_name="job3_nonexistent")

        # Should only return existing directories
        job_dirs = registry.get_job_dirs("test-sweep")
        assert len(job_dirs) == 2
        assert jobs_dir / "job1" in job_dirs
        assert jobs_dir / "job2" in job_dirs

    def test_find_sweeps_by_pattern(self, jobs_dir: Path):
        """Test finding sweeps by pattern."""
        registry = SweepRegistry(jobs_dir)

        registry.register_job(sweep_id="2026-01-31_17-01-04", job_name="job1")
        registry.register_job(sweep_id="2026-01-31_18-30-00", job_name="job2")
        registry.register_job(sweep_id="2026-02-01_10-00-00", job_name="job3")

        # Find by date pattern
        sweeps = registry.find_sweeps_by_pattern("2026-01-31*")
        assert len(sweeps) == 2


class TestFindSweepJobsByPattern:
    """Tests for find_sweep_jobs_by_pattern function."""

    def test_find_by_timestamp(self, jobs_dir: Path, job_config: JobConfig):
        """Test finding jobs by timestamp pattern."""
        # Create job directories with timestamps
        create_mock_job_dir(jobs_dir, "dataset_agent_2026-01-31_17-01-04_run0", job_config)
        create_mock_job_dir(jobs_dir, "dataset_agent_2026-01-31_17-01-04_run1", job_config)
        create_mock_job_dir(jobs_dir, "dataset_agent_2026-01-31_18-00-00_run0", job_config)

        # Find by timestamp
        job_dirs = find_sweep_jobs_by_pattern(jobs_dir, "2026-01-31_17-01-04")
        assert len(job_dirs) == 2

    def test_find_by_glob_pattern(self, jobs_dir: Path, job_config: JobConfig):
        """Test finding jobs by glob pattern."""
        create_mock_job_dir(jobs_dir, "agentdojo_gpt-4o_attack_run0", job_config)
        create_mock_job_dir(jobs_dir, "agentdojo_gpt-4o_attack_run1", job_config)
        create_mock_job_dir(jobs_dir, "agentdojo_gpt-4-turbo_attack_run0", job_config)

        # Find by pattern
        job_dirs = find_sweep_jobs_by_pattern(jobs_dir, "*_gpt-4o_*")
        assert len(job_dirs) == 2

    def test_excludes_directories_without_config(self, jobs_dir: Path, job_config: JobConfig):
        """Test that directories without config.yaml are excluded."""
        create_mock_job_dir(jobs_dir, "valid_job_2026-01-31_17-01-04_run0", job_config)

        # Create a directory without config
        invalid_dir = jobs_dir / "invalid_job_2026-01-31_17-01-04_run1"
        invalid_dir.mkdir()

        job_dirs = find_sweep_jobs_by_pattern(jobs_dir, "2026-01-31_17-01-04")
        assert len(job_dirs) == 1


class TestGetJobStatus:
    """Tests for get_job_status function."""

    def test_empty_job(self, jobs_dir: Path, job_config: JobConfig):
        """Test status of job with no runs."""
        job_dir = create_mock_job_dir(jobs_dir, "empty_job", job_config)

        status = get_job_status(job_dir)
        assert status["total_tasks"] == 0
        assert status["completed_tasks"] == 0
        assert status["failed_tasks"] == 0
        assert status["has_failures"] is False

    def test_job_with_successful_runs(self, jobs_dir: Path, job_config: JobConfig):
        """Test status of job with successful runs."""
        entries = [
            {
                "task_id": "task1",
                "run_id": "abc12345",
                "timestamp": datetime.now().isoformat(),
                "benign_score": 1.0,
                "attack_score": 0.0,
                "exception_type": None,
                "path": "task1/abc12345",
            },
            {
                "task_id": "task2",
                "run_id": "def67890",
                "timestamp": datetime.now().isoformat(),
                "benign_score": 1.0,
                "attack_score": 0.0,
                "exception_type": None,
                "path": "task2/def67890",
            },
        ]
        job_dir = create_mock_job_dir(jobs_dir, "successful_job", job_config, entries)

        status = get_job_status(job_dir)
        assert status["total_tasks"] == 2
        assert status["completed_tasks"] == 2
        assert status["failed_tasks"] == 0
        assert status["has_failures"] is False

    def test_job_with_failed_runs(self, jobs_dir: Path, job_config: JobConfig):
        """Test status of job with failed runs."""
        entries = [
            {
                "task_id": "task1",
                "run_id": "abc12345",
                "timestamp": datetime.now().isoformat(),
                "benign_score": None,
                "attack_score": None,
                "exception_type": "ModelHTTPError",
                "path": "task1/abc12345",
            },
            {
                "task_id": "task2",
                "run_id": "def67890",
                "timestamp": datetime.now().isoformat(),
                "benign_score": None,
                "attack_score": None,
                "exception_type": "TimeoutError",
                "path": "task2/def67890",
            },
        ]
        job_dir = create_mock_job_dir(jobs_dir, "failed_job", job_config, entries)

        status = get_job_status(job_dir)
        assert status["total_tasks"] == 2
        assert status["completed_tasks"] == 0
        assert status["failed_tasks"] == 2
        assert status["has_failures"] is True
        assert "ModelHTTPError" in status["failure_types"]
        assert "TimeoutError" in status["failure_types"]

    def test_job_with_mixed_runs(self, jobs_dir: Path, job_config: JobConfig):
        """Test status of job with both successful and failed runs."""
        entries = [
            {
                "task_id": "task1",
                "run_id": "abc12345",
                "timestamp": datetime.now().isoformat(),
                "benign_score": 1.0,
                "attack_score": 0.0,
                "exception_type": None,
                "path": "task1/abc12345",
            },
            {
                "task_id": "task2",
                "run_id": "def67890",
                "timestamp": datetime.now().isoformat(),
                "benign_score": None,
                "attack_score": None,
                "exception_type": "ModelHTTPError",
                "path": "task2/def67890",
            },
        ]
        job_dir = create_mock_job_dir(jobs_dir, "mixed_job", job_config, entries)

        status = get_job_status(job_dir)
        assert status["total_tasks"] == 2
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1
        assert status["has_failures"] is True

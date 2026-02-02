# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for results aggregation.

These tests cover the aggregate_results function and related utilities
for computing pass@k metrics across job directories.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job.models import CONFIG_FILENAME, INDEX_FILENAME, JobConfig, RunIndexEntry
from prompt_siren.results import (
    _group_by_task,
    _read_all_jobs,
    aggregate_results,
    estimate_pass_at_k,
    GroupBy,
)


def _create_job_config(
    job_name: str = "test_job",
    dataset: str = "testdataset",
    dataset_config: dict | None = None,
    agent_type: str = "plain",
    agent_name: str = "agent:test-model",
    attack_type: str | None = None,
    attack_config: dict | None = None,
) -> JobConfig:
    """Create a JobConfig for testing."""
    return JobConfig(
        job_name=job_name,
        execution_mode="benign" if attack_type is None else "attack",
        created_at=datetime.now(),
        dataset=DatasetConfig(type=dataset, config=dataset_config or {}),
        agent=AgentConfig(type=agent_type, config={"model": agent_name}),
        attack=AttackConfig(type=attack_type, config=attack_config or {}) if attack_type else None,
        execution=ExecutionConfig(concurrency=1),
        telemetry=TelemetryConfig(trace_console=False),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


def _save_job_config(job_dir: Path, job_config: JobConfig) -> None:
    """Save job config to YAML file."""
    config_path = job_dir / CONFIG_FILENAME
    config_dict = job_config.model_dump(mode="json")

    header = f"# Job: {job_config.job_name}\n"
    header += f"# Created: {job_config.created_at.isoformat()}\n"
    header += f"# Mode: {job_config.execution_mode}\n\n"

    with open(config_path, "w") as f:
        f.write(header)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _add_index_entry(
    job_dir: Path,
    task_id: str,
    timestamp: str,
    benign_score: float,
    attack_score: float | None,
    run_id: str | None = None,
) -> None:
    """Helper to add an entry to a job's index.jsonl."""
    index_file = job_dir / INDEX_FILENAME

    # Parse timestamp string to datetime
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    entry = RunIndexEntry(
        task_id=task_id,
        run_id=run_id or f"run{timestamp[-4:]}",
        timestamp=dt,
        benign_score=benign_score,
        attack_score=attack_score,
        exception_type=None,
        path=Path(f"{task_id}/{run_id or f'run{timestamp[-4:]}'}"),
    )

    with index_file.open("a") as f:
        f.write(entry.model_dump_json() + "\n")


def _create_job_with_entries(
    jobs_dir: Path,
    job_name: str,
    entries: list[tuple[str, str, float, float | None]],
    dataset: str = "testdataset",
    dataset_config: dict | None = None,
    agent_type: str = "agent",
    agent_name: str = "agent:test-model",
    attack_type: str = "attack",
    attack_config: dict | None = None,
) -> Path:
    """Create a job directory with config and index entries.

    Args:
        jobs_dir: Parent directory for jobs
        job_name: Name of the job
        entries: List of (task_id, timestamp, benign_score, attack_score) tuples
        dataset: Dataset type
        dataset_config: Dataset configuration
        agent_type: Agent type
        agent_name: Agent name (model)
        attack_type: Attack type
        attack_config: Attack configuration

    Returns:
        Path to the job directory
    """
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True)

    job_config = _create_job_config(
        job_name=job_name,
        dataset=dataset,
        dataset_config=dataset_config,
        agent_type=agent_type,
        agent_name=agent_name,
        attack_type=attack_type,
        attack_config=attack_config,
    )
    _save_job_config(job_dir, job_config)

    for task_id, timestamp, benign, attack in entries:
        _add_index_entry(job_dir, task_id, timestamp, benign, attack)

    return job_dir


@pytest.fixture
def jobs_dir_basic(tmp_path: Path) -> Path:
    """Create a jobs directory with sample results."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job_1",
        entries=[
            ("task1", "20240101_120000", 1.0, 0.0),
            ("task2", "20240102_130000", 0.0, 0.0),
        ],
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:azure:gpt-4",
        attack_type="no-attack",
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_multiple_timestamps(tmp_path: Path) -> Path:
    """Create jobs directory with multiple timestamps for same task."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.1),
            ("task1", "20240103_120000", 2.0, 0.2),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_pass_at_k(tmp_path: Path) -> Path:
    """Create jobs directory for pass@k testing with varying scores."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.5),
            ("task1", "20240103_120000", 0.5, 1.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_exact_k_samples(tmp_path: Path) -> Path:
    """Create jobs directory with exactly k samples per task."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            # Task 1: Has one success (benign=1.0) among 3 runs
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            # Task 2: Has no success among 3 runs
            ("task2", "20240101_120000", 0.0, 0.0),
            ("task2", "20240102_120000", 0.0, 0.0),
            ("task2", "20240103_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_estimator_samples(tmp_path: Path) -> Path:
    """Create jobs directory with more than k samples (uses estimator)."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create 5 samples for task1, with 2 successes
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            ("task1", "20240104_120000", 1.0, 0.0),
            ("task1", "20240105_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_estimator_samples_multiple_tasks(tmp_path: Path) -> Path:
    """Create jobs directory with more than k samples (uses estimator) for multiple tasks."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            # Task 1: 5 samples, 2 successes
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            ("task1", "20240104_120000", 1.0, 0.0),
            ("task1", "20240105_120000", 0.0, 0.0),
            # Task 2: 3 samples, no successes
            ("task2", "20240101_120000", 0.0, 0.0),
            ("task2", "20240102_120000", 0.0, 0.0),
            ("task2", "20240103_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_estimator_samples_multiples_of_all_grouped_fields(tmp_path: Path) -> Path:
    """Create jobs directory with multiple datasets, agents, and attacks."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    job_counter = 0
    for dataset in ["dataset1", "dataset2"]:
        for agent in ["agent1", "agent2"]:
            for attack in ["attack1", "attack2"]:
                job_counter += 1
                entries = []
                # Task 1: 5 samples, 2 successes
                for i, benign in enumerate([0.0, 1.0, 0.0, 1.0, 0.0]):
                    entries.append(
                        (
                            "task1",
                            f"2024010{i + 1}_120000",
                            benign,
                            0.0,
                        )
                    )

                # Task 2: 3 samples
                if dataset == "dataset1":
                    # For dataset1, task2 has one attack success
                    for i, (benign, attack_score) in enumerate(
                        [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
                    ):
                        entries.append(
                            (
                                "task2",
                                f"2024010{i + 1}_120000",
                                benign,
                                attack_score,
                            )
                        )
                else:
                    # For dataset2, task2 has no successes
                    for i, (benign, attack_score) in enumerate(
                        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
                    ):
                        entries.append(
                            (
                                "task2",
                                f"2024010{i + 1}_120000",
                                benign,
                                attack_score,
                            )
                        )

                _create_job_with_entries(
                    jobs_dir,
                    job_name=f"job_{dataset}_{agent}_{attack}_{job_counter}",
                    entries=entries,
                    dataset=dataset,
                    agent_type=agent,
                    agent_name=f"{agent}:test-model",
                    attack_type=attack,
                )

    return jobs_dir


@pytest.fixture
def jobs_dir_insufficient_samples(tmp_path: Path) -> Path:
    """Create jobs directory with fewer than k samples."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create only 2 samples
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 1.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_all_successes(tmp_path: Path) -> Path:
    """Create jobs directory where all samples succeed."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create 5 samples, all with benign_score=1.0
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 1.0, 1.0),
            ("task1", "20240102_120000", 1.0, 1.0),
            ("task1", "20240103_120000", 1.0, 1.0),
            ("task1", "20240104_120000", 1.0, 1.0),
            ("task1", "20240105_120000", 1.0, 1.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_no_successes(tmp_path: Path) -> Path:
    """Create jobs directory where no samples succeed."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create 5 samples, all with benign_score=0.0
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 0.0, 0.0),
            ("task1", "20240102_120000", 0.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            ("task1", "20240104_120000", 0.0, 0.0),
            ("task1", "20240105_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_metadata_test(tmp_path: Path) -> Path:
    """Create jobs directory for testing metadata columns."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            # 3 samples for task1
            ("task1", "20240101_120000", 1.0, 0.0),
            ("task1", "20240102_120000", 0.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            # 5 samples for task2
            ("task2", "20240101_120000", 1.0, 1.0),
            ("task2", "20240102_120000", 1.0, 1.0),
            ("task2", "20240103_120000", 0.0, 1.0),
            ("task2", "20240104_120000", 0.0, 1.0),
            ("task2", "20240105_120000", 0.0, 1.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_metadata_pass_at_k(tmp_path: Path) -> Path:
    """Create jobs directory for testing metadata columns with pass@k."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create exactly 3 samples for both tasks
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 1.0, 0.0),
            ("task1", "20240102_120000", 0.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            ("task2", "20240101_120000", 1.0, 0.0),
            ("task2", "20240102_120000", 0.0, 0.0),
            ("task2", "20240103_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


@pytest.fixture
def jobs_dir_multiple_k_values(tmp_path: Path) -> Path:
    """Create jobs directory for testing with multiple k values."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # Create 5 samples for task1 with 2 successes
    _create_job_with_entries(
        jobs_dir,
        job_name="test_job",
        entries=[
            ("task1", "20240101_120000", 1.0, 0.0),
            ("task1", "20240102_120000", 1.0, 0.0),
            ("task1", "20240103_120000", 0.0, 0.0),
            ("task1", "20240104_120000", 0.0, 0.0),
            ("task1", "20240105_120000", 0.0, 0.0),
        ],
    )

    return jobs_dir


class TestAggregatorBasic:
    """Test basic aggregation functionality."""

    def test_read_all_jobs(self, jobs_dir_basic: Path) -> None:
        """Test reading from job directories."""
        all_rows = _read_all_jobs(jobs_dir_basic)
        df = pd.DataFrame(all_rows)

        assert len(df) == 2  # Two rows (one per index entry)
        assert list(df["dataset"].unique()) == ["agentdojo"]
        assert list(df["agent_type"].unique()) == ["plain-agent"]
        assert list(df["attack_type"].unique()) == ["no-attack"]
        assert list(df["agent_name"].unique()) == ["plain:azure:gpt-4"]
        assert set(df["task_id"]) == {"task1", "task2"}

    def test_aggregator_average_metrics(self, jobs_dir_basic: Path) -> None:
        """Test metric averaging."""
        # Default is group_by=GroupBy.ALL which averages across tasks
        df = aggregate_results(jobs_dir_basic, group_by=GroupBy.ALL)

        # Should have one row after aggregating by config
        assert len(df) == 1

        # Average of 1.0 and 0.0 = 0.5
        assert df.iloc[0]["benign_pass@1"] == pytest.approx(0.5)
        # Average of 0.0 and 0.0 = 0.0
        assert df.iloc[0]["attack_pass@1"] == pytest.approx(0.0)


class TestAggregatorMultipleTimestamps:
    """Test aggregation with multiple timestamps for same task."""

    def test_averaging_multiple_timestamps(self, jobs_dir_multiple_timestamps: Path) -> None:
        """Test averaging when same task has multiple timestamps."""
        # Raw data has 3 rows
        all_rows = _read_all_jobs(jobs_dir_multiple_timestamps)
        df = pd.DataFrame(all_rows)
        assert len(df) == 3
        assert all(df["task_id"] == "task1")

        # Aggregate by task (average timestamps)
        df_by_task = _group_by_task(df)
        assert len(df_by_task) == 1  # One unique task

        # Check averaged values
        assert df_by_task.iloc[0]["task_id"] == "task1"
        assert df_by_task.iloc[0]["benign_pass@1"] == pytest.approx(1.0)  # (0 + 1 + 2) / 3
        assert df_by_task.iloc[0]["attack_pass@1"] == pytest.approx(0.1)  # (0 + 0.1 + 0.2) / 3


class TestAggregatorEmptyAndNonexistent:
    """Test aggregation with empty or nonexistent directories."""

    def test_aggregator_empty_directory(self, tmp_path: Path) -> None:
        """Test aggregation with empty directory (no jobs)."""
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()

        # Should return empty DataFrame (not raise error)
        df = aggregate_results(jobs_dir)
        assert df.empty

    def test_aggregator_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test aggregation with nonexistent directory."""
        jobs_dir = tmp_path / "nonexistent"

        # Should return empty DataFrame (not raise error)
        df = aggregate_results(jobs_dir)
        assert df.empty


class TestAggregatorGrouping:
    """Test aggregate_results function with different grouping options."""

    def test_grouping_by_all(self, jobs_dir_basic: Path) -> None:
        """Test group_by=GroupBy.ALL."""
        df_all = aggregate_results(jobs_dir_basic, group_by=GroupBy.ALL)
        assert len(df_all) == 1  # One configuration
        assert "benign_pass@1" in df_all.columns
        assert "attack_pass@1" in df_all.columns
        assert "dataset" in df_all.columns
        assert "agent_type" in df_all.columns

    def test_grouping_by_dataset(self, jobs_dir_basic: Path) -> None:
        """Test group_by=GroupBy.DATASET."""
        df_env = aggregate_results(jobs_dir_basic, group_by=GroupBy.DATASET)
        assert len(df_env) == 1  # One dataset type
        assert "dataset" in df_env.columns
        assert "benign_pass@1" in df_env.columns

    def test_grouping_by_agent(self, jobs_dir_basic: Path) -> None:
        """Test group_by=GroupBy.AGENT."""
        df_agent = aggregate_results(jobs_dir_basic, group_by=GroupBy.AGENT)
        assert len(df_agent) == 1  # One agent type
        assert "agent_type" in df_agent.columns

    def test_grouping_by_agent_name(self, jobs_dir_basic: Path) -> None:
        """Test group_by=GroupBy.AGENT_NAME."""
        df_agent_name = aggregate_results(jobs_dir_basic, group_by=GroupBy.AGENT_NAME)
        assert len(df_agent_name) == 1  # One agent name
        assert "agent_name" in df_agent_name.columns


class TestPassAtK:
    """Tests for pass@k metric computation."""

    def test_pass_at_1_averaging(self, jobs_dir_pass_at_k: Path) -> None:
        """Test pass@1 metric (default behavior - averaging)."""
        df = aggregate_results(jobs_dir_pass_at_k, group_by=GroupBy.ALL, k=1)

        assert len(df) == 1
        assert "benign_pass@1" in df.columns
        assert "attack_pass@1" in df.columns

        # Should average: (0.0 + 1.0 + 0.5) / 3 = 0.5
        assert df.iloc[0]["benign_pass@1"] == pytest.approx(0.5)
        # Should average: (0.0 + 0.5 + 1.0) / 3 = 0.5
        assert df.iloc[0]["attack_pass@1"] == pytest.approx(0.5)

    def test_pass_at_k_exact_k_samples(self, jobs_dir_exact_k_samples: Path) -> None:
        """Test pass@k when we have exactly k samples."""
        df = aggregate_results(jobs_dir_exact_k_samples, group_by=GroupBy.ALL, k=3)

        assert len(df) == 1
        assert "benign_pass@3" in df.columns
        assert "attack_pass@3" in df.columns

        # Average across tasks: (1.0 + 0.0) / 2 = 0.5
        assert df.iloc[0]["benign_pass@3"] == pytest.approx(0.5)
        # Average across tasks: (0.0 + 0.0) / 2 = 0.0
        assert df.iloc[0]["attack_pass@3"] == pytest.approx(0.0)

    def test_pass_at_k_with_estimator(self, jobs_dir_estimator_samples: Path) -> None:
        """Test pass@k when we have more than k samples (uses estimator)."""
        df = aggregate_results(jobs_dir_estimator_samples, group_by=GroupBy.ALL, k=3)

        assert len(df) == 1
        assert "benign_pass@3" in df.columns

        # With n=5, c=2, k=3, the estimator should give 0.9 for task1
        benign_pass_k = df.iloc[0]["benign_pass@3"]
        assert benign_pass_k == pytest.approx(0.9)

        # Verify against direct calculation
        expected = estimate_pass_at_k(5, [2], 3)[0]
        assert benign_pass_k == pytest.approx(expected)

    def test_pass_at_k_with_estimator_and_multiple_tasks(
        self, jobs_dir_estimator_samples_multiple_tasks: Path
    ) -> None:
        """Test pass@k when we have more than k samples (uses estimator)."""
        df = aggregate_results(jobs_dir_estimator_samples_multiple_tasks, group_by=GroupBy.ALL, k=3)

        assert len(df) == 1
        assert "benign_pass@3" in df.columns

        benign_pass_k = df.iloc[0]["benign_pass@3"]
        # 0.9 for task1 and 0.0 for task2, averaging to 0.45
        assert benign_pass_k == pytest.approx(0.45)

    def test_pass_at_k_insufficient_samples_error(
        self, jobs_dir_insufficient_samples: Path
    ) -> None:
        """Test that pass@k raises error when we have fewer than k samples."""
        with pytest.raises(ValueError, match="has only 2 samples but k=3"):
            aggregate_results(jobs_dir_insufficient_samples, group_by=GroupBy.ALL, k=3)

    def test_pass_at_k_all_successes(self, jobs_dir_all_successes: Path) -> None:
        """Test pass@k when all samples succeed."""
        df = aggregate_results(jobs_dir_all_successes, group_by=GroupBy.ALL, k=3)

        assert len(df) == 1
        # With all successes, pass@k should be 1.0
        assert df.iloc[0]["benign_pass@3"] == pytest.approx(1.0)
        assert df.iloc[0]["attack_pass@3"] == pytest.approx(1.0)

    def test_pass_at_k_no_successes(self, jobs_dir_no_successes: Path) -> None:
        """Test pass@k when no samples succeed."""
        df = aggregate_results(jobs_dir_no_successes, group_by=GroupBy.ALL, k=3)

        assert len(df) == 1
        # With no successes, pass@k should be 0.0
        assert df.iloc[0]["benign_pass@3"] == pytest.approx(0.0)
        assert df.iloc[0]["attack_pass@3"] == pytest.approx(0.0)


class TestEstimatePassAtK:
    """Tests for the estimate_pass_at_k function."""

    def test_basic_cases(self) -> None:
        """Test estimate_pass_at_k with basic cases."""
        # Test with n=k (should return 1.0 if c > 0, else 0.0)
        result = estimate_pass_at_k(3, [2], 3)
        assert result[0] == pytest.approx(1.0)

        result = estimate_pass_at_k(3, [0], 3)
        assert result[0] == pytest.approx(0.0)

        # Test with n > k
        result = estimate_pass_at_k(5, [2], 3)
        # Should be between 0 and 1
        assert 0.0 < result[0] < 1.0

        # Test with all correct (c = n)
        result = estimate_pass_at_k(5, [5], 3)
        assert result[0] == pytest.approx(1.0)

        # Test with no correct
        result = estimate_pass_at_k(5, [0], 3)
        assert result[0] == pytest.approx(0.0)

    def test_multiple_tasks(self) -> None:
        """Test estimate_pass_at_k with multiple tasks."""
        result = estimate_pass_at_k(5, [3, 2, 0], 3)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)  # c=3, n=5, k=3 -> high probability
        assert 0.0 < result[1] < 1.0  # c=2, n=5, k=3 -> medium probability
        assert result[2] == pytest.approx(0.0)  # c=0, n=5, k=3 -> zero probability

    def test_different_n_per_task(self) -> None:
        """Test estimate_pass_at_k with different n for each task."""
        result = estimate_pass_at_k([5, 3, 10], [2, 2, 5], 3)
        assert len(result) == 3
        # Each should be calculated with its own n
        assert all(0.0 <= r <= 1.0 for r in result)


class TestMetadataColumns:
    """Tests for metadata columns in results."""

    def test_metadata_columns_in_results(self, jobs_dir_metadata_test: Path) -> None:
        """Test that n_samples, n_tasks, and avg_n_samples columns are included."""
        df = aggregate_results(jobs_dir_metadata_test, group_by=GroupBy.ALL, k=1)

        assert "n_tasks" in df.columns
        assert "avg_n_samples" in df.columns

        # Should have 2 tasks
        assert df.iloc[0]["n_tasks"] == 2
        # Average samples: (3 + 5) / 2 = 4.0
        assert df.iloc[0]["avg_n_samples"] == pytest.approx(4.0)

    def test_metadata_columns_with_pass_at_k(self, jobs_dir_metadata_pass_at_k: Path) -> None:
        """Test metadata columns with pass@k metric."""
        df = aggregate_results(jobs_dir_metadata_pass_at_k, group_by=GroupBy.ALL, k=3)

        assert "n_tasks" in df.columns
        assert "avg_n_samples" in df.columns

        # Should have 2 tasks
        assert df.iloc[0]["n_tasks"] == 2
        # All tasks have exactly 3 samples
        assert df.iloc[0]["avg_n_samples"] == pytest.approx(3.0)


class TestMultipleKValues:
    """Tests for multiple k values."""

    def test_aggregate_results_with_multiple_k_values(
        self, jobs_dir_multiple_k_values: Path
    ) -> None:
        """Test aggregate_results with multiple k values."""
        df = aggregate_results(jobs_dir_multiple_k_values, group_by=GroupBy.ALL, k=[1, 3, 5])

        # Should have 3 rows, one for each k value
        assert len(df) == 3

        # Check k column exists and has correct values
        assert "k" in df.columns
        assert set(df["k"]) == {1, 3, 5}

        # Each row should have appropriate pass@k columns
        for _, row in df.iterrows():
            k_val = int(row["k"])
            assert f"benign_pass@{k_val}" in row
            assert f"attack_pass@{k_val}" in row

        # All should have same metadata
        assert all(df["n_tasks"] == 1)
        assert all(df["avg_n_samples"] == 5.0)

        # Verify k=1 is averaging
        k1_row = df[df["k"] == 1].iloc[0]
        assert k1_row["benign_pass@1"] == pytest.approx(0.4)  # (1+1+0+0+0)/5

        # Verify k=5 with n=5, c=2: should be 1.0 (since n-c=3 < k=5)
        k5_row = df[df["k"] == 5].iloc[0]
        assert k5_row["benign_pass@5"] == pytest.approx(1.0)

        # Verify k=3 uses estimator (n=5, c=2, k=3)
        k3_row = df[df["k"] == 3].iloc[0]
        assert 0.0 < k3_row["benign_pass@3"] < 1.0  # Should use estimator


class TestGroupingWithMultipleConfigurations:
    """Tests for grouping behavior with multiple environments, agents, and attacks."""

    def test_grouping_with_multiple_configurations(
        self, jobs_dir_estimator_samples_multiples_of_all_grouped_fields: Path
    ) -> None:
        """Test grouping behavior with multiple environments, agents, and attacks."""
        # Test group_by=GroupBy.ALL - should have one row per combination
        df_all = aggregate_results(
            jobs_dir_estimator_samples_multiples_of_all_grouped_fields,
            group_by=GroupBy.ALL,
            k=3,
        )
        # 2 datasets x 2 agents x 2 attacks = 8 configurations
        assert len(df_all) == 8
        assert set(df_all["dataset"]) == {"dataset1", "dataset2"}
        assert set(df_all["agent_type"]) == {"agent1", "agent2"}
        assert set(df_all["attack_type"]) == {"attack1", "attack2"}

        # All should have same benign pass@k values (same data per config)
        for _, row in df_all.iterrows():
            # With n=5, c=2, k=3 for task1 and n=3, c=0, k=3 for task2
            # Average: (0.9 + 0.0) / 2 = 0.45
            assert row["benign_pass@3"] == pytest.approx(0.45)
            assert row["n_tasks"] == 2
            assert row["avg_n_samples"] == pytest.approx(4.0)  # (5 + 3) / 2
            if row["dataset"] == "dataset1":
                # For dataset1, task2 has c=1 out of n=3
                assert row["attack_pass@3"] == pytest.approx(0.5)
            elif row["dataset"] == "dataset2":
                # For dataset2, task2 has c=0 out of n=3
                assert row["attack_pass@3"] == pytest.approx(0.0)

    def test_grouping_by_dataset(
        self, jobs_dir_estimator_samples_multiples_of_all_grouped_fields: Path
    ) -> None:
        """Test group_by=GroupBy.DATASET aggregates across agents and attacks."""
        df = aggregate_results(
            jobs_dir_estimator_samples_multiples_of_all_grouped_fields,
            group_by=GroupBy.DATASET,
            k=3,
        )

        # Should have one row per dataset (2 datasets)
        assert len(df) == 2

        # Should include dataset but not agent_type or attack_type
        assert "dataset" in df.columns
        assert "agent_type" not in df.columns
        assert "attack_type" not in df.columns
        assert "agent_name" not in df.columns

        # Check datasets are present
        assert set(df["dataset"]) == {"dataset1", "dataset2"}

        # Each dataset has 4 configurations (2 agents x 2 attacks)
        for _, row in df.iterrows():
            assert row["benign_pass@3"] == pytest.approx(0.45)
            if row["dataset"] == "dataset1":
                assert row["attack_pass@3"] == pytest.approx(0.5)
            elif row["dataset"] == "dataset2":
                assert row["attack_pass@3"] == pytest.approx(0.0)

            # n_tasks is the count of configurations in this group (2 agents x 2 attacks x 2 tasks = 8)
            assert row["n_tasks"] == 8
            assert row["avg_n_samples"] == pytest.approx(4.0)

    def test_grouping_by_agent(
        self, jobs_dir_estimator_samples_multiples_of_all_grouped_fields: Path
    ) -> None:
        """Test group_by=GroupBy.AGENT aggregates across datasets and attacks."""
        df = aggregate_results(
            jobs_dir_estimator_samples_multiples_of_all_grouped_fields,
            group_by=GroupBy.AGENT,
            k=3,
        )

        # Should have one row per agent (2 agents)
        assert len(df) == 2

        # Should include agent_type but not dataset or attack_type
        assert "agent_type" in df.columns
        assert "dataset" not in df.columns
        assert "attack_type" not in df.columns
        assert "agent_name" not in df.columns

        # Check agents are present
        assert set(df["agent_type"]) == {"agent1", "agent2"}

        # Each agent has 4 configurations (2 datasets x 2 attacks)
        for _, row in df.iterrows():
            assert row["benign_pass@3"] == pytest.approx(0.45)
            assert row["attack_pass@3"] == pytest.approx(0.25)
            # n_tasks is the count of configurations in this group (2 datasets x 2 attacks x 2 tasks = 8)
            assert row["n_tasks"] == 8
            assert row["avg_n_samples"] == pytest.approx(4.0)

    def test_grouping_by_attack(
        self, jobs_dir_estimator_samples_multiples_of_all_grouped_fields: Path
    ) -> None:
        """Test group_by=GroupBy.ATTACK aggregates across datasets and agents."""
        df = aggregate_results(
            jobs_dir_estimator_samples_multiples_of_all_grouped_fields,
            group_by=GroupBy.ATTACK,
            k=3,
        )

        # Should have one row per attack (2 attacks)
        assert len(df) == 2

        # Should include attack_type but not dataset or agent_type
        assert "attack_type" in df.columns
        assert "dataset" not in df.columns
        assert "agent_type" not in df.columns
        assert "agent_name" not in df.columns

        # Check attacks are present
        assert set(df["attack_type"]) == {"attack1", "attack2"}

        # Each attack has 4 configurations (2 datasets x 2 agents)
        for _, row in df.iterrows():
            assert row["benign_pass@3"] == pytest.approx(0.45)
            assert row["attack_pass@3"] == pytest.approx(0.25)
            # n_tasks is the count of configurations in this group (2 datasets x 2 agents x 2 tasks = 8)
            assert row["n_tasks"] == 8
            assert row["avg_n_samples"] == pytest.approx(4.0)

    def test_grouping_by_agent_name(
        self, jobs_dir_estimator_samples_multiples_of_all_grouped_fields: Path
    ) -> None:
        """Test group_by=GroupBy.AGENT_NAME aggregates across datasets, agents, and attacks."""
        df = aggregate_results(
            jobs_dir_estimator_samples_multiples_of_all_grouped_fields,
            group_by=GroupBy.AGENT_NAME,
            k=3,
        )

        # Should have one row per agent_name (2 agent names: agent1:test-model, agent2:test-model)
        assert len(df) == 2

        # Should include agent_name but not other grouping columns
        assert "agent_name" in df.columns
        assert "dataset" not in df.columns
        assert "agent_type" not in df.columns
        assert "attack_type" not in df.columns

        # Check agent names are present
        assert set(df["agent_name"]) == {"agent1:test-model", "agent2:test-model"}

        # Aggregates all 8 configurations with identical data
        for _, row in df.iterrows():
            assert row["benign_pass@3"] == pytest.approx(0.45)
            assert row["attack_pass@3"] == pytest.approx(0.25)
            # n_tasks is the count of all configurations (2 datasets x 2 attacks x 2 tasks = 8)
            assert row["n_tasks"] == 8
            assert row["avg_n_samples"] == pytest.approx(4.0)


class TestGroupingByDatasetSuite:
    """Tests for dataset suite grouping."""

    def test_grouping_by_dataset_suite(self, tmp_path: Path) -> None:
        """Test group_by=GroupBy.DATASET_SUITE aggregates across dataset suites.

        This tests that dataset_suite grouping:
        1. Acts like groupby="all" but replaces dataset with dataset_suite
        2. Properly handles task name clashes across dataset suites
        3. Aggregates across dataset values that share the same suite
        """
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()

        # Create job for agentdojo workspace suite
        _create_job_with_entries(
            jobs_dir,
            job_name="job_workspace",
            entries=[
                ("task1", "20240101_120000", 1.0, 0.0),
                ("task2", "20240101_120001", 0.8, 0.2),
            ],
            dataset="agentdojo",
            dataset_config={"suite_name": "workspace"},
            agent_type="plain-agent",
            agent_name="plain:gpt-4",
            attack_type="no-attack",
        )

        # Create job for agentdojo banking suite (has task1 with different scores)
        _create_job_with_entries(
            jobs_dir,
            job_name="job_banking",
            entries=[
                ("task1", "20240101_120002", 0.6, 0.4),
                ("task3", "20240101_120003", 0.4, 0.6),
            ],
            dataset="agentdojo",
            dataset_config={"suite_name": "banking"},
            agent_type="plain-agent",
            agent_name="plain:gpt-4",
            attack_type="no-attack",
        )

        # Test groupby=DATASET - should show one row for all suites
        df_dataset = aggregate_results(jobs_dir, group_by=GroupBy.DATASET, k=1)
        # Since all have the same dataset="agentdojo", should be grouped together
        assert len(df_dataset) == 1
        assert "dataset" in df_dataset.columns
        assert df_dataset.iloc[0]["dataset"] == "agentdojo"
        # Average across all 4 tasks
        assert df_dataset.iloc[0]["benign_pass@1"] == pytest.approx((1.0 + 0.8 + 0.6 + 0.4) / 4)
        assert df_dataset.iloc[0]["n_tasks"] == 4

        # Test groupby="all" - should show one row for the whole dataset (as above)
        df_all = aggregate_results(jobs_dir, group_by=GroupBy.ALL, k=1)
        assert len(df_all) == 1  # One config (same dataset, agent, attack)
        assert "dataset" in df_all.columns

        # Test groupby=DATASET_SUITE - should aggregate by suite
        df_suite = aggregate_results(jobs_dir, group_by=GroupBy.DATASET_SUITE, k=1)
        assert len(df_suite) == 2  # Two suites (workspace and banking)
        assert "dataset_suite" in df_suite.columns
        assert "dataset" not in df_suite.columns  # dataset replaced by dataset_suite
        assert set(df_suite["dataset_suite"]) == {"workspace", "banking"}

        # Check workspace results (average of task1=1.0 and task2=0.8)
        workspace_row = df_suite[df_suite["dataset_suite"] == "workspace"].iloc[0]
        assert workspace_row["benign_pass@1"] == pytest.approx(0.9)
        assert workspace_row["benign_pass@1"] == pytest.approx((1.0 + 0.8) / 2)
        assert workspace_row["attack_pass@1"] == pytest.approx((0.0 + 0.2) / 2)
        assert workspace_row["n_tasks"] == 2

        # Check banking results (average of task1=0.6 and task3=0.4)
        banking_row = df_suite[df_suite["dataset_suite"] == "banking"].iloc[0]
        assert banking_row["benign_pass@1"] == pytest.approx(0.5)
        assert banking_row["benign_pass@1"] == pytest.approx((0.6 + 0.4) / 2)
        assert banking_row["attack_pass@1"] == pytest.approx((0.4 + 0.6) / 2)
        assert banking_row["n_tasks"] == 2

    def test_dataset_suite_grouping_with_multiple_agents(self, tmp_path: Path) -> None:
        """Verify that dataset_suite grouping still respects other config dimensions."""
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()

        # Create job for workspace suite with gpt-4
        _create_job_with_entries(
            jobs_dir,
            job_name="job_workspace_gpt4",
            entries=[
                ("task1", "20240101_120000", 1.0, 0.0),
                ("task2", "20240101_120001", 0.8, 0.2),
            ],
            dataset="agentdojo",
            dataset_config={"suite_name": "workspace"},
            agent_type="plain-agent",
            agent_name="plain:gpt-4",
            attack_type="no-attack",
        )

        # Create job for banking suite with gpt-4
        _create_job_with_entries(
            jobs_dir,
            job_name="job_banking_gpt4",
            entries=[
                ("task1", "20240101_120002", 0.6, 0.4),
                ("task3", "20240101_120003", 0.4, 0.6),
            ],
            dataset="agentdojo",
            dataset_config={"suite_name": "banking"},
            agent_type="plain-agent",
            agent_name="plain:gpt-4",
            attack_type="no-attack",
        )

        # Add a second agent configuration for workspace
        _create_job_with_entries(
            jobs_dir,
            job_name="job_workspace_gpt35",
            entries=[
                ("task1", "20240101_120004", 0.5, 0.5),
                ("task3", "20240101_120004", 0.0, 0.0),
            ],
            dataset="agentdojo",
            dataset_config={"suite_name": "workspace"},
            agent_type="plain-agent",
            agent_name="plain:gpt-3.5",  # Different agent
            attack_type="no-attack",
        )

        df_suite = aggregate_results(jobs_dir, group_by=GroupBy.DATASET_SUITE, k=1)
        # Should have 3 rows (2 for workspace with different agents, 1 for banking)
        assert len(df_suite) == 3
        assert all(df_suite["dataset_suite"].isin(["workspace", "banking"]))

        # Check workspace with gpt-4 (average of task1=1.0 and task2=0.8)
        workspace_gpt4_row = df_suite[
            (df_suite["dataset_suite"] == "workspace") & (df_suite["agent_name"] == "plain:gpt-4")
        ].iloc[0]
        assert workspace_gpt4_row["benign_pass@1"] == pytest.approx(0.9)
        assert workspace_gpt4_row["attack_pass@1"] == pytest.approx((0.0 + 0.2) / 2)
        assert workspace_gpt4_row["n_tasks"] == 2

        # Check banking with gpt-4
        banking_row = df_suite[
            (df_suite["dataset_suite"] == "banking") & (df_suite["agent_name"] == "plain:gpt-4")
        ].iloc[0]
        assert banking_row["benign_pass@1"] == pytest.approx(0.5)
        assert banking_row["attack_pass@1"] == pytest.approx((0.4 + 0.6) / 2)
        assert banking_row["n_tasks"] == 2

        # Check workspace with gpt-3.5
        other_agent_workspace_row = df_suite[
            (df_suite["dataset_suite"] == "workspace") & (df_suite["agent_name"] == "plain:gpt-3.5")
        ].iloc[0]
        assert other_agent_workspace_row["benign_pass@1"] == pytest.approx(0.25)
        assert other_agent_workspace_row["attack_pass@1"] == pytest.approx(0.25)
        assert other_agent_workspace_row["n_tasks"] == 2

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for results aggregation."""

import json
from pathlib import Path

import pandas as pd
import pytest
from prompt_siren.results import (
    _group_by_task,
    _read_index,
    aggregate_results,
    estimate_pass_at_k,
    GroupBy,
)


def _add_index_entry(
    output_dir: Path,
    task_id: str,
    timestamp: str,
    dataset: str,
    dataset_config: dict[str, str] | None,
    agent_type: str,
    agent_name: str,
    attack_type: str | None,
    config_hash: str,
    benign_score: float,
    attack_score: float | None,
    attack_config: dict[str, str] | None = None,
) -> None:
    """Helper to add an entry to index.jsonl."""
    index_file = output_dir / "index.jsonl"

    entry = {
        "execution_id": "test_exec",
        "task_id": task_id,
        "timestamp": timestamp,
        "dataset": dataset,
        "dataset_config": dataset_config or {},
        "agent_type": agent_type,
        "agent_name": agent_name,
        "attack_type": attack_type,
        "attack_config": attack_config,
        "config_hash": config_hash,
        "benign_score": benign_score,
        "attack_score": attack_score,
        "path": str(output_dir / "dummy.json"),
    }

    with index_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory with sample results."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create index entries
    _add_index_entry(
        output_dir,
        task_id="task1",
        timestamp="20240101_120000",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:azure:gpt-4",
        attack_type="no-attack",
        config_hash="abc123",
        benign_score=1.0,
        attack_score=0.0,
        attack_config={},
    )
    _add_index_entry(
        output_dir,
        task_id="task2",
        timestamp="20240102_130000",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:azure:gpt-4",
        attack_type="no-attack",
        config_hash="abc123",
        benign_score=0.0,
        attack_score=0.0,
        attack_config={},
    )

    return output_dir


@pytest.fixture
def output_dir_multiple_timestamps(tmp_path: Path) -> Path:
    """Create output directory with multiple timestamps for same task."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create multiple results for same task
    for i, timestamp in enumerate(["20240101_120000", "20240102_120000", "20240103_120000"]):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=timestamp,
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=float(i),
            attack_score=float(i) * 0.1,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_pass_at_k(tmp_path: Path) -> Path:
    """Create output directory for pass@k testing with varying scores."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create multiple results for same task with different scores
    for i, (benign, attack) in enumerate([(0.0, 0.0), (1.0, 0.5), (0.5, 1.0)]):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=attack,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_exact_k_samples(tmp_path: Path) -> Path:
    """Create output directory with exactly k samples per task."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Task 1: Has one success (benign=1.0) among 3 runs
    for i, (benign, attack) in enumerate([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=attack,
            attack_config={},
        )

    # Task 2: Has no success among 3 runs
    for i, (benign, attack) in enumerate([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]):
        _add_index_entry(
            output_dir,
            task_id="task2",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=attack,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_estimator_samples(tmp_path: Path) -> Path:
    """Create output directory with more than k samples (uses estimator)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 5 samples for task1, with 2 successes
    for i, benign in enumerate([0.0, 1.0, 0.0, 1.0, 0.0]):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=0.0,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_estimator_samples_multiple_tasks(tmp_path: Path) -> Path:
    """Create output directory with more than k samples (uses estimator)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 5 samples for task1, with 2 successes
    for i, benign in enumerate([0.0, 1.0, 0.0, 1.0, 0.0]):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=0.0,
            attack_config={},
        )

    # Task 2: Has no success among 3 runs
    for i, (benign, attack) in enumerate([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]):
        _add_index_entry(
            output_dir,
            task_id="task2",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=benign,
            attack_score=attack,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_estimator_samples_multiples_of_all_grouped_fields(
    tmp_path: Path,
) -> Path:
    """Create output directory with more than k samples (uses estimator)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    for dataset in ["dataset1", "dataset2"]:
        for agent in ["agent1", "agent2"]:
            for attack in ["attack1", "attack2"]:
                # Create 5 samples for task1, with 2 successes
                for i, benign in enumerate([0.0, 1.0, 0.0, 1.0, 0.0]):
                    _add_index_entry(
                        output_dir,
                        task_id="task1",
                        timestamp=f"2024010{i}_120000",
                        dataset=dataset,
                        dataset_config=None,
                        agent_type=agent,
                        agent_name=f"{agent}:test-model",
                        attack_type=attack,
                        config_hash="hash",
                        benign_score=benign,
                        attack_score=0.0,
                        attack_config={},
                    )

                if dataset == "dataset1":
                    # Task 2: Has no success among 3 runs
                    for i, (benign, attack_score) in enumerate(
                        [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
                    ):
                        _add_index_entry(
                            output_dir,
                            task_id="task2",
                            timestamp=f"2024010{i}_120000",
                            dataset=dataset,
                            dataset_config=None,
                            agent_type=agent,
                            agent_name=f"{agent}:test-model",
                            attack_type=attack,
                            config_hash="hash",
                            benign_score=benign,
                            attack_score=attack_score,
                            attack_config={},
                        )
                elif dataset == "dataset2":
                    # Task 2: Has no success among 3 runs
                    for i, (benign, attack_score) in enumerate(
                        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
                    ):
                        _add_index_entry(
                            output_dir,
                            task_id="task2",
                            timestamp=f"2024010{i}_120000",
                            dataset=dataset,
                            dataset_config=None,
                            agent_type=agent,
                            agent_name=f"{agent}:test-model",
                            attack_type=attack,
                            config_hash="hash",
                            benign_score=benign,
                            attack_score=attack_score,
                            attack_config={},
                        )

    return output_dir


@pytest.fixture
def output_dir_insufficient_samples(tmp_path: Path) -> Path:
    """Create output directory with fewer than k samples."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create only 2 samples
    for i in range(2):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=1.0,
            attack_score=0.0,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_all_successes(tmp_path: Path) -> Path:
    """Create output directory where all samples succeed."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 5 samples, all with benign_score=1.0
    for i in range(5):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=1.0,
            attack_score=1.0,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_no_successes(tmp_path: Path) -> Path:
    """Create output directory where no samples succeed."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 5 samples, all with benign_score=0.0
    for i in range(5):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=0.0,
            attack_score=0.0,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_metadata_test(tmp_path: Path) -> Path:
    """Create output directory for testing metadata columns."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 3 samples for task1 and 5 samples for task2
    for i in range(3):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=1.0 if i == 0 else 0.0,
            attack_score=0.0,
            attack_config={},
        )

    for i in range(5):
        _add_index_entry(
            output_dir,
            task_id="task2",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=1.0 if i < 2 else 0.0,
            attack_score=1.0,
            attack_config={},
        )

    return output_dir


@pytest.fixture
def output_dir_metadata_pass_at_k(tmp_path: Path) -> Path:
    """Create output directory for testing metadata columns with pass@k."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create exactly 3 samples for both tasks
    for task_id in ["task1", "task2"]:
        for i in range(3):
            _add_index_entry(
                output_dir,
                task_id=task_id,
                timestamp=f"2024010{i}_120000",
                dataset="testdataset",
                dataset_config=None,
                agent_type="agent",
                agent_name="agent:test-model",
                attack_type="attack",
                config_hash="hash",
                benign_score=1.0 if i == 0 else 0.0,
                attack_score=0.0,
                attack_config={},
            )

    return output_dir


@pytest.fixture
def output_dir_multiple_k_values(tmp_path: Path) -> Path:
    """Create output directory for testing with multiple k values."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 5 samples for task1
    for i in range(5):
        _add_index_entry(
            output_dir,
            task_id="task1",
            timestamp=f"2024010{i}_120000",
            dataset="testdataset",
            dataset_config=None,
            agent_type="agent",
            agent_name="agent:test-model",
            attack_type="attack",
            config_hash="hash",
            benign_score=1.0 if i < 2 else 0.0,
            attack_score=0.0,
            attack_config={},
        )

    return output_dir


def test_aggregator_basic(temp_output_dir: Path) -> None:
    """Test basic aggregation functionality."""
    # Read from index
    all_rows = _read_index(temp_output_dir)
    df = pd.DataFrame(all_rows)

    assert len(df) == 2  # Two rows (one per index entry)
    assert list(df["dataset"].unique()) == ["agentdojo"]
    assert list(df["agent_type"].unique()) == ["plain-agent"]
    assert list(df["attack_type"].unique()) == ["no-attack"]
    assert list(df["agent_name"].unique()) == ["plain:azure:gpt-4"]
    assert list(df["config_hash"].unique()) == ["abc123"]
    assert set(df["task_id"]) == {"task1", "task2"}


def test_aggregator_average_metrics(temp_output_dir: Path) -> None:
    """Test metric averaging."""
    # Default is group_by=GroupBy.ALL which averages across tasks
    df = aggregate_results(temp_output_dir, group_by=GroupBy.ALL)

    # Should have one row after aggregating by config
    assert len(df) == 1

    # Average of 1.0 and 0.0 = 0.5
    assert df.iloc[0]["benign_pass@1"] == pytest.approx(0.5)
    # Average of 0.0 and 0.0 = 0.0
    assert df.iloc[0]["attack_pass@1"] == pytest.approx(0.0)


def test_aggregator_multiple_timestamps_same_task(output_dir_multiple_timestamps: Path) -> None:
    """Test averaging when same task has multiple timestamps."""
    # Raw data has 3 rows
    all_rows = _read_index(output_dir_multiple_timestamps)
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


def test_aggregator_empty_directory(tmp_path: Path) -> None:
    """Test aggregation with empty directory (no index file)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Index file does not exist"):
        aggregate_results(output_dir)


def test_aggregator_nonexistent_directory(tmp_path: Path) -> None:
    """Test aggregation with nonexistent directory."""
    output_dir = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError, match="Index file does not exist"):
        aggregate_results(output_dir)


def test_aggregate_results_with_grouping(temp_output_dir: Path) -> None:
    """Test aggregate_results function with different grouping options."""
    # Default grouping by GroupBy.ALL - averages across tasks for each config
    df_all = aggregate_results(temp_output_dir, group_by=GroupBy.ALL)
    assert len(df_all) == 1  # One configuration
    assert "benign_pass@1" in df_all.columns
    assert "attack_pass@1" in df_all.columns
    assert "dataset" in df_all.columns
    assert "agent_type" in df_all.columns

    # Group by environment
    df_env = aggregate_results(temp_output_dir, group_by=GroupBy.DATASET)
    assert len(df_env) == 1  # One environment type
    assert "dataset" in df_env.columns
    assert "benign_pass@1" in df_env.columns

    # Group by agent
    df_agent = aggregate_results(temp_output_dir, group_by=GroupBy.AGENT)
    assert len(df_agent) == 1  # One agent type
    assert "agent_type" in df_agent.columns

    # Group by agent_name
    df_agent_name = aggregate_results(temp_output_dir, group_by=GroupBy.AGENT_NAME)
    assert len(df_agent_name) == 1  # One agent name
    assert "agent_name" in df_agent_name.columns


def test_pass_at_1_averaging(output_dir_pass_at_k: Path) -> None:
    """Test pass@1 metric (default behavior - averaging)."""
    # Test with k=1 (default - averaging)
    df = aggregate_results(output_dir_pass_at_k, group_by=GroupBy.ALL, k=1)

    assert len(df) == 1
    assert "benign_pass@1" in df.columns
    assert "attack_pass@1" in df.columns

    # Should average: (0.0 + 1.0 + 0.5) / 3 = 0.5
    assert df.iloc[0]["benign_pass@1"] == pytest.approx(0.5)
    # Should average: (0.0 + 0.5 + 1.0) / 3 = 0.5
    assert df.iloc[0]["attack_pass@1"] == pytest.approx(0.5)


def test_pass_at_k_exact_k_samples(output_dir_exact_k_samples: Path) -> None:
    """Test pass@k when we have exactly k samples."""
    # Test with k=3 (exactly 3 samples per task)
    df = aggregate_results(output_dir_exact_k_samples, group_by=GroupBy.ALL, k=3)

    assert len(df) == 1
    assert "benign_pass@3" in df.columns
    assert "attack_pass@3" in df.columns

    # Average across tasks: (1.0 + 0.0) / 2 = 0.5
    assert df.iloc[0]["benign_pass@3"] == pytest.approx(0.5)
    # Average across tasks: (0.0 + 0.0) / 2 = 0.0
    assert df.iloc[0]["attack_pass@3"] == pytest.approx(0.0)


def test_pass_at_k_with_estimator(output_dir_estimator_samples: Path) -> None:
    """Test pass@k when we have more than k samples (uses estimator)."""
    # Test with k=3 (we have 5 samples, so estimator is used)
    df = aggregate_results(output_dir_estimator_samples, group_by=GroupBy.ALL, k=3)

    assert len(df) == 1
    assert "benign_pass@3" in df.columns

    # With n=5, c=2, k=3, the estimator should give an estimate of 0.9 for the one and only task1
    benign_pass_k = df.iloc[0]["benign_pass@3"]
    assert benign_pass_k == pytest.approx(0.9)

    # Verify against direct calculation
    expected = estimate_pass_at_k(5, [2], 3)[0]
    assert benign_pass_k == pytest.approx(expected)


def test_pass_at_k_with_estimator_and_multiple_tasks(
    output_dir_estimator_samples_multiple_tasks: Path,
) -> None:
    """Test pass@k when we have more than k samples (uses estimator)."""
    # Test with k=3 (we have 5 samples, so estimator is used)
    df = aggregate_results(output_dir_estimator_samples_multiple_tasks, group_by=GroupBy.ALL, k=3)

    assert len(df) == 1
    assert "benign_pass@3" in df.columns

    # With n=5, c=2, k=3, the estimator should give a value less than 1.0
    # but greater than 0 (since we have some successes)
    benign_pass_k = df.iloc[0]["benign_pass@3"]
    # 0.9 should be the pass@k estimate for task1 based on the formula and 0.0 for task2, averaging to 0.45
    assert benign_pass_k == pytest.approx(0.45)


def test_pass_at_k_insufficient_samples_error(output_dir_insufficient_samples: Path) -> None:
    """Test that pass@k raises error when we have fewer than k samples."""
    # Test with k=3 (we only have 2 samples, should error)
    with pytest.raises(ValueError, match="has only 2 samples but k=3"):
        aggregate_results(output_dir_insufficient_samples, group_by=GroupBy.ALL, k=3)


def test_pass_at_k_all_successes(output_dir_all_successes: Path) -> None:
    """Test pass@k when all samples succeed."""
    # Test with k=3
    df = aggregate_results(output_dir_all_successes, group_by=GroupBy.ALL, k=3)

    assert len(df) == 1
    # With all successes, pass@k should be 1.0
    assert df.iloc[0]["benign_pass@3"] == pytest.approx(1.0)
    assert df.iloc[0]["attack_pass@3"] == pytest.approx(1.0)


def test_pass_at_k_no_successes(output_dir_no_successes: Path) -> None:
    """Test pass@k when no samples succeed."""
    # Test with k=3
    df = aggregate_results(output_dir_no_successes, group_by=GroupBy.ALL, k=3)

    assert len(df) == 1
    # With no successes, pass@k should be 0.0
    assert df.iloc[0]["benign_pass@3"] == pytest.approx(0.0)
    assert df.iloc[0]["attack_pass@3"] == pytest.approx(0.0)


def test_estimate_pass_at_k_basic() -> None:
    """Test the estimate_pass_at_k function with basic cases."""
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


def test_estimate_pass_at_k_multiple_tasks() -> None:
    """Test estimate_pass_at_k with multiple tasks."""
    # Same n for all tasks
    result = estimate_pass_at_k(5, [3, 2, 0], 3)
    assert len(result) == 3
    assert result[0] == pytest.approx(1.0)  # c=3, n=5, k=3 -> high probability
    assert 0.0 < result[1] < 1.0  # c=2, n=5, k=3 -> medium probability
    assert result[2] == pytest.approx(0.0)  # c=0, n=5, k=3 -> zero probability


def test_estimate_pass_at_k_different_n() -> None:
    """Test estimate_pass_at_k with different n for each task."""
    result = estimate_pass_at_k([5, 3, 10], [2, 2, 5], 3)
    assert len(result) == 3
    # Each should be calculated with its own n
    assert all(0.0 <= r <= 1.0 for r in result)


def test_metadata_columns_in_results(output_dir_metadata_test: Path) -> None:
    """Test that n_samples, n_tasks, and avg_n_samples columns are included in results."""
    # Test with k=1
    df = aggregate_results(output_dir_metadata_test, group_by=GroupBy.ALL, k=1)

    assert "n_tasks" in df.columns
    assert "avg_n_samples" in df.columns

    # Should have 2 tasks
    assert df.iloc[0]["n_tasks"] == 2
    # Average samples: (3 + 5) / 2 = 4.0
    assert df.iloc[0]["avg_n_samples"] == pytest.approx(4.0)


def test_metadata_columns_with_pass_at_k(output_dir_metadata_pass_at_k: Path) -> None:
    """Test metadata columns with pass@k metric."""
    # Test with k=3
    df = aggregate_results(output_dir_metadata_pass_at_k, group_by=GroupBy.ALL, k=3)

    assert "n_tasks" in df.columns
    assert "avg_n_samples" in df.columns

    # Should have 2 tasks
    assert df.iloc[0]["n_tasks"] == 2
    # All tasks have exactly 3 samples
    assert df.iloc[0]["avg_n_samples"] == pytest.approx(3.0)


def test_aggregate_results_with_multiple_k_values(output_dir_multiple_k_values: Path) -> None:
    """Test aggregate_results with multiple k values."""
    # Test with multiple k values
    df = aggregate_results(output_dir_multiple_k_values, group_by=GroupBy.ALL, k=[1, 3, 5])

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


def test_grouping_with_multiple_configurations(
    output_dir_estimator_samples_multiples_of_all_grouped_fields: Path,
) -> None:
    """Test grouping behavior with multiple environments, agents, and attacks."""
    # Test group_by=GroupBy.ALL - should have one row per combination
    df_all = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.ALL, k=3
    )
    # 2 envs x 2 agents x 2 attacks = 8 configurations
    assert len(df_all) == 8
    assert set(df_all["dataset"]) == {"dataset1", "dataset2"}
    assert set(df_all["agent_type"]) == {"agent1", "agent2"}
    assert set(df_all["attack_type"]) == {"attack1", "attack2"}

    # All should have same pass@k values (same data per config)
    for _, row in df_all.iterrows():
        # With n=5, c=2, k=3 for task1 and n=3, c=0, k=3 for task2
        # Average: (0.9 + 0.0) / 2 = 0.45
        assert row["benign_pass@3"] == pytest.approx(0.45)
        assert row["n_tasks"] == 2
        assert row["avg_n_samples"] == pytest.approx(4.0)  # (5 + 3) / 2
        if row["dataset"] == "dataset1":
            # For env1, task2 has c=0 out of n=3
            assert row["attack_pass@3"] == pytest.approx(0.5)
        elif row["dataset"] == "dataset2":
            # For env2, task2 has c=0 out of n=3
            assert row["attack_pass@3"] == pytest.approx(0.0)

    # Test group_by=GroupBy.DATASET - should aggregate across agents and attacks
    df_env = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.DATASET, k=3
    )
    assert len(df_env) == 2
    assert set(df_env["dataset"]) == {"dataset1", "dataset2"}
    # Should not have agent/attack columns
    assert "agent_type" not in df_env.columns
    assert "attack_type" not in df_env.columns

    # Test group_by=GroupBy.AGENT - should aggregate across envs and attacks
    df_agent = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.AGENT, k=3
    )
    assert len(df_agent) == 2
    assert set(df_agent["agent_type"]) == {"agent1", "agent2"}
    assert "dataset" not in df_agent.columns
    assert "attack_type" not in df_agent.columns

    # Test group_by=GroupBy.ATTACK - should aggregate across envs and agents
    df_attack = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.ATTACK, k=3
    )
    assert len(df_attack) == 2
    assert set(df_attack["attack_type"]) == {"attack1", "attack2"}
    assert "dataset" not in df_attack.columns
    assert "agent_type" not in df_attack.columns


def test_grouping_by_env(
    output_dir_estimator_samples_multiples_of_all_grouped_fields: Path,
) -> None:
    """Test group_by=GroupBy.DATASET aggregates across agents and attacks."""
    df = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.DATASET, k=3
    )

    # Should have one row per environment (2 environments)
    assert len(df) == 2

    # Should include env_type but not agent_type or attack_type
    assert "dataset" in df.columns
    assert "agent_type" not in df.columns
    assert "attack_type" not in df.columns
    assert "agent_name" not in df.columns
    assert "config_hash" not in df.columns

    # Check environments are present
    assert set(df["dataset"]) == {"dataset1", "dataset2"}

    # Each environment has 4 configurations (2 agents x 2 attacks)
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
    output_dir_estimator_samples_multiples_of_all_grouped_fields: Path,
) -> None:
    """Test group_by=GroupBy.AGENT aggregates across environments and attacks."""
    df = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.AGENT, k=3
    )

    # Should have one row per agent (2 agents)
    assert len(df) == 2

    # Should include agent_type but not env_type or attack_type
    assert "agent_type" in df.columns
    assert "dataset" not in df.columns
    assert "attack_type" not in df.columns
    assert "agent_name" not in df.columns
    assert "config_hash" not in df.columns

    # Check agents are present
    assert set(df["agent_type"]) == {"agent1", "agent2"}

    # Each agent has 4 configurations (2 envs x 2 attacks)
    # All with identical data, so metrics should be same
    for _, row in df.iterrows():
        assert row["benign_pass@3"] == pytest.approx(0.45)
        assert row["attack_pass@3"] == pytest.approx(0.25)
        # n_tasks is the count of configurations in this group (2 envs x 2 attacks x 2 tasks = 8)
        assert row["n_tasks"] == 8
        assert row["avg_n_samples"] == pytest.approx(4.0)


def test_grouping_by_attack(
    output_dir_estimator_samples_multiples_of_all_grouped_fields: Path,
) -> None:
    """Test group_by=GroupBy.ATTACK aggregates across environments and agents."""
    df = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields, group_by=GroupBy.ATTACK, k=3
    )

    # Should have one row per attack (2 attacks)
    assert len(df) == 2

    # Should include attack_type but not env_type or agent_type
    assert "attack_type" in df.columns
    assert "dataset" not in df.columns
    assert "agent_type" not in df.columns
    assert "agent_name" not in df.columns
    assert "config_hash" not in df.columns

    # Check attacks are present
    assert set(df["attack_type"]) == {"attack1", "attack2"}

    # Each attack has 4 configurations (2 envs x 2 agents)
    # All with identical data, so metrics should be same
    for _, row in df.iterrows():
        assert row["benign_pass@3"] == pytest.approx(0.45)
        assert row["attack_pass@3"] == pytest.approx(0.25)
        # n_tasks is the count of configurations in this group (2 envs x 2 agents x 2 tasks = 8)
        assert row["n_tasks"] == 8
        assert row["avg_n_samples"] == pytest.approx(4.0)


def test_grouping_by_agent_name(
    output_dir_estimator_samples_multiples_of_all_grouped_fields: Path,
) -> None:
    """Test group_by=GroupBy.AGENT_NAME aggregates across environments, agent types, and attacks."""
    df = aggregate_results(
        output_dir_estimator_samples_multiples_of_all_grouped_fields,
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
    assert "config_hash" not in df.columns

    # Check agent names are present
    assert set(df["agent_name"]) == {"agent1:test-model", "agent2:test-model"}

    # Aggregates all 8 configurations with identical data
    for _, row in df.iterrows():
        assert row["benign_pass@3"] == pytest.approx(0.45)
        assert row["attack_pass@3"] == pytest.approx(0.25)
        # n_tasks is the count of all configurations (2 envs x 2 attacks x 2 tasks = 8)
        assert row["n_tasks"] == 8
        assert row["avg_n_samples"] == pytest.approx(4.0)


def test_grouping_by_dataset_suite(tmp_path: Path) -> None:
    """Test group_by=GroupBy.DATASET_SUITE aggregates across dataset suites.

    This tests that dataset_suite grouping:
    1. Acts like groupby="all" but replaces dataset with dataset_suite
    2. Properly handles task name clashes across dataset suites
    3. Aggregates across dataset values that share the same suite
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create results for two agentdojo suites (workspace and banking)
    # Both have a task with the same name "task1" to test clash handling
    # Use same agent and config for simplicity

    # agentdojo workspace suite
    _add_index_entry(
        output_dir,
        task_id="task1",
        timestamp="20240101_120000",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:gpt-4",
        attack_type="no-attack",
        config_hash="hash1",
        benign_score=1.0,
        attack_score=0.0,
        attack_config={},
    )
    _add_index_entry(
        output_dir,
        task_id="task2",
        timestamp="20240101_120001",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:gpt-4",
        attack_type="no-attack",
        config_hash="hash1",
        benign_score=0.8,
        attack_score=0.2,
        attack_config={},
    )

    # agentdojo banking suite (has task1 with different scores)
    _add_index_entry(
        output_dir,
        task_id="task1",
        timestamp="20240101_120002",
        dataset="agentdojo",
        dataset_config={"suite_name": "banking"},
        agent_type="plain-agent",
        agent_name="plain:gpt-4",
        attack_type="no-attack",
        config_hash="hash1",
        benign_score=0.6,
        attack_score=0.4,
        attack_config={},
    )
    _add_index_entry(
        output_dir,
        task_id="task3",
        timestamp="20240101_120003",
        dataset="agentdojo",
        dataset_config={"suite_name": "banking"},
        agent_type="plain-agent",
        agent_name="plain:gpt-4",
        attack_type="no-attack",
        config_hash="hash1",
        benign_score=0.4,
        attack_score=0.6,
        attack_config={},
    )

    # Test groupby=DATASET - should show one row for all suites
    df_dataset = aggregate_results(output_dir, group_by=GroupBy.DATASET, k=1)
    # Since all have the same dataset="agentdojo", should be grouped together
    assert len(df_dataset) == 1
    assert "dataset" in df_dataset.columns
    assert df_dataset.iloc[0]["dataset"] == "agentdojo"
    # Average across all 4 tasks
    assert df_dataset.iloc[0]["benign_pass@1"] == pytest.approx((1.0 + 0.8 + 0.6 + 0.4) / 4)
    assert df_dataset.iloc[0]["n_tasks"] == 4

    # Test groupby="all" - should show one row for the whole dataset (as above)
    df_all = aggregate_results(output_dir, group_by=GroupBy.ALL, k=1)
    assert len(df_all) == 1  # One config (same dataset, agent, attack)
    assert "dataset" in df_all.columns

    # Test groupby=DATASET_SUITE - should aggregate by suite
    df_suite = aggregate_results(output_dir, group_by=GroupBy.DATASET_SUITE, k=1)
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

    # Verify that dataset_suite grouping still respects other config dimensions
    # Add a second agent configuration
    _add_index_entry(
        output_dir,
        task_id="task1",
        timestamp="20240101_120004",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:gpt-3.5",  # Different agent
        attack_type="no-attack",
        config_hash="hash2",
        benign_score=0.5,
        attack_score=0.5,
        attack_config={},
    )

    _add_index_entry(
        output_dir,
        task_id="task3",
        timestamp="20240101_120004",
        dataset="agentdojo",
        dataset_config={"suite_name": "workspace"},
        agent_type="plain-agent",
        agent_name="plain:gpt-3.5",  # Different agent
        attack_type="no-attack",
        config_hash="hash2",
        benign_score=0.0,
        attack_score=0.0,
        attack_config={},
    )

    df_suite2 = aggregate_results(output_dir, group_by=GroupBy.DATASET_SUITE, k=1)
    # Should have 3 rows now (2 for workspace with different agents, 1 for banking)
    assert len(df_suite2) == 3
    assert all(df_suite2["dataset_suite"].isin(["workspace", "banking"]))

    # Check workspace results (average of task1=1.0 and task2=0.8)
    workspace_row = df_suite2[
        (df_suite2["dataset_suite"] == "workspace") & (df_suite2["agent_name"] == "plain:gpt-4")
    ].iloc[0]
    assert workspace_row["benign_pass@1"] == pytest.approx(0.9)
    assert workspace_row["benign_pass@1"] == pytest.approx((1.0 + 0.8) / 2)
    assert workspace_row["attack_pass@1"] == pytest.approx((0.0 + 0.2) / 2)
    assert workspace_row["n_tasks"] == 2

    # Check banking results (average of task1=0.6 and task3=0.4)
    banking_row = df_suite2[
        (df_suite2["dataset_suite"] == "banking") & (df_suite2["agent_name"] == "plain:gpt-4")
    ].iloc[0]
    assert banking_row["benign_pass@1"] == pytest.approx(0.5)
    assert banking_row["benign_pass@1"] == pytest.approx((0.6 + 0.4) / 2)
    assert banking_row["attack_pass@1"] == pytest.approx((0.4 + 0.6) / 2)
    assert banking_row["n_tasks"] == 2

    other_agent_workspace_row = df_suite2[
        (df_suite2["dataset_suite"] == "workspace") & (df_suite2["agent_name"] == "plain:gpt-3.5")
    ].iloc[0]
    assert other_agent_workspace_row["benign_pass@1"] == pytest.approx(0.25)
    assert other_agent_workspace_row["attack_pass@1"] == pytest.approx(0.25)
    assert other_agent_workspace_row["n_tasks"] == 2

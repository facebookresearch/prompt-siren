# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Results aggregation and formatting from trace outputs.

This module provides functionality to aggregate experiment results from the index.jsonl file
in the trace directory. It reads the index to identify model, agent, environment, etc.,
and aggregates task results, handling multiple executions of the same task by averaging.
"""

import itertools
import json
import sys
from enum import auto
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing_extensions import assert_never

from .run_persistence import INDEX_FILENAME, IndexEntry


class GroupBy(StrEnum):
    """Grouping levels for results aggregation."""

    DATASET = auto()
    DATASET_SUITE = auto()
    AGENT = auto()
    ATTACK = auto()
    AGENT_NAME = auto()
    ALL = auto()


class Format(StrEnum):
    """Output format options for results."""

    TABLE = auto()
    JSON = auto()
    CSV = auto()


# Grouping columns used for aggregation
# Note: config_hash is not included to allow grouping across dataset suites with the same agent/attack config
_ALL_GROUP_COLS = ["dataset", "agent_type", "agent_name", "attack_type"]


def estimate_pass_at_k(num_samples: int | list[int], num_correct: list[int], k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array.

    Args:
        num_samples: Total number of samples per task (int or list of ints)
        num_correct: Number of correct samples per task
        k: The k value for pass@k metric

    Returns:
        Array of pass@k estimates for each task
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct, strict=True)]
    )


def _parse_index_entry(line: str) -> dict[str, Any]:
    """Parse and validate a single index entry line.

    Args:
        line: JSON line from index.jsonl

    Returns:
        Row dictionary with None values replaced for aggregation
    """
    entry_data = json.loads(line)
    entry = IndexEntry.model_validate(entry_data)
    row = entry.model_dump()

    # Replace None with default values for aggregation
    if row["attack_type"] is None:
        row["attack_type"] = "benign"
    else:
        # For template_string attacks, append the template_short_name to attack_type
        if row["attack_type"] == "template_string" and row.get("attack_config"):
            template_short_name = row["attack_config"].get("template_short_name")
            if template_short_name:
                row["attack_type"] = f"template_string_{template_short_name}"

    if row["attack_score"] is None:
        row["attack_score"] = float("nan")

    # Extract dataset_suite from dataset_config
    dataset_config = row.get("dataset_config", {})
    if dataset_config and "suite_name" in dataset_config:
        row["dataset_suite"] = dataset_config["suite_name"]
    else:
        # Use a placeholder for datasets without suites
        row["dataset_suite"] = f"{row['dataset']}_default"

    return row


def _read_index(output_dir: Path) -> list[dict[str, Any]]:
    """Read all result rows from the index.jsonl file.

    Args:
        output_dir: Path to trace directory containing index.jsonl

    Returns:
        List of row dictionaries, one per index entry

    Raises:
        FileNotFoundError: If index file does not exist
    """
    index_file = output_dir / INDEX_FILENAME

    if not index_file.exists():
        raise FileNotFoundError(f"Index file does not exist at {index_file}.")

    with index_file.open() as f:
        return [_parse_index_entry(line.strip()) for line in f if line.strip()]


def _group_by_task(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Group results by task_id, computing pass@k metric.

    This is the first stage of grouping that always happens regardless of user selection.
    Groups by all configuration columns (dataset, agent_type, agent_name, attack_type, config_hash)
    plus dataset_suite and task_id. The dataset_suite is included to properly handle task name
    clashes across dataset suites when doing dataset_suite-level aggregation.

    For k=1 (default): Averages scores across timestamps for each task.
    For k>1: Computes pass@k metric where a task gets 1.0 if at least one run passes (score=1.0).
            If n_samples < k: raises ValueError
            If n_samples == k: gives 1.0 if any run passes, 0.0 otherwise
            If n_samples > k: uses the estimator formula

    Args:
        df: Raw results DataFrame
        k: The k value for pass@k metric (default=1 means averaging)

    Returns:
        DataFrame with pass@k metrics per task

    Raises:
        ValueError: If any task has fewer than k samples
    """
    if df.empty:
        return df

    # Group by configuration and task
    # Include dataset_suite and config_hash to disambiguate tasks from different suites with the same name
    group_cols = [*_ALL_GROUP_COLS, "dataset_suite", "config_hash", "task_id"]

    if k == 1:
        # Original behavior: average across timestamps
        numeric_cols = ["benign_score", "attack_score"]
        grouped = df.groupby(group_cols)
        result = grouped[numeric_cols].mean().reset_index()
        # Add metadata: average number of samples per task
        result["n_samples"] = grouped.size().values
        return result.rename(
            columns={"benign_score": "benign_pass@1", "attack_score": "attack_pass@1"}
        )

    # For k > 1: compute pass@k metric
    results = []
    for group_key, group in df.groupby(group_cols):
        n_samples = len(group)

        # Error if we don't have enough samples
        if n_samples < k:
            task_id = group["task_id"].iloc[0]
            raise ValueError(
                f"Task '{task_id}' has only {n_samples} samples but k={k}. Need at least k samples to compute pass@{k}."
            )

        # Count number of correct samples (score = 1.0)
        n_benign_correct = (group["benign_score"] == 1.0).sum()
        n_attack_correct = (group["attack_score"] == 1.0).sum()

        # If n == k: simple case
        if n_samples == k:
            benign_pass_k = 1.0 if n_benign_correct > 0 else 0.0
            attack_pass_k = 1.0 if n_attack_correct > 0 else 0.0
        else:
            # n > k: use estimator
            benign_pass_k = estimate_pass_at_k(n_samples, [n_benign_correct], k)[0]
            attack_pass_k = estimate_pass_at_k(n_samples, [n_attack_correct], k)[0]

        # Create result row - group_key can be a tuple or scalar
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        result_row: dict[str, Any] = dict(zip(group_cols, key_tuple, strict=True))
        result_row[f"benign_pass@{k}"] = benign_pass_k
        result_row[f"attack_pass@{k}"] = attack_pass_k
        result_row["n_samples"] = n_samples
        results.append(result_row)

    return pd.DataFrame(results)


def aggregate_results(
    output_dir: Path,
    group_by: GroupBy = GroupBy.ALL,
    k: int | list[int] = 1,
) -> pd.DataFrame:
    """Aggregate results from trace directory with optional grouping and pass@k metric.

    Reads the index.jsonl file from the trace directory to aggregate results.

    For k=1: Averages scores across timestamps for each task.
    For k>1: Computes pass@k metric where a task passes if at least one run succeeds.
    Multiple k values: Computes pass@k for each k and combines results with a 'k' column.

    Grouping proceeds in two stages:
    1. Always group by task_id (computing pass@k metric)
    2. Optionally group by selected dimension (dataset, dataset_suite, agent, attack, agent_name, or show all configs)

    Args:
        output_dir: Path to trace directory (e.g., ./traces)
        group_by: Grouping level - "all" (default) shows full config breakdown by dataset,
                  "dataset_suite" replaces dataset with dataset_suite, others aggregate further
        k: The k value(s) for pass@k metric. Can be a single int or list of ints.
           Default=1 means averaging. Multiple values compute separate metrics for each k.

    Returns:
        DataFrame with aggregated results based on grouping level.
        If multiple k values provided, includes a 'k' column to identify each metric.
        Default view includes: dataset, agent_type, agent_name, attack_type,
        benign_pass@k, attack_pass@k, n_tasks, avg_n_samples
        (aggregates across dataset_suite's and config_hash variations)

    Raises:
        ValueError: If any task has fewer than k samples when k > 1
    """
    # Convert single k to list for uniform handling
    k_values = [k] if isinstance(k, int) else k

    # Handle multiple k values by computing each separately and combining
    if len(k_values) > 1:
        all_results = []
        for k_value in k_values:
            result = aggregate_results(output_dir, group_by=group_by, k=k_value)
            if not result.empty:
                result["k"] = k_value
                all_results.append(result)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    # Single k value - process normally
    k_value = k_values[0]

    # Read all rows from index
    all_rows = _read_index(output_dir)

    if not all_rows:
        return pd.DataFrame()

    # Create DataFrame from all rows
    df = pd.DataFrame(all_rows)

    # Stage 1: Always group by task (computing pass@k)
    df = _group_by_task(df, k=k_value)

    # Determine score column names based on k
    benign_col = f"benign_pass@{k_value}"
    attack_col = f"attack_pass@{k_value}"
    score_cols = [benign_col, attack_col]

    # Stage 2: Optional grouping by user selection
    if group_by == "all":
        # Show all configurations - group by all config columns (averaging across tasks)
        grouped = df.groupby(_ALL_GROUP_COLS)
        result = grouped[score_cols].mean().reset_index()
        # Add metadata columns
        result["n_tasks"] = grouped.size().values
        result["avg_n_samples"] = grouped["n_samples"].mean().values
        return result

    if group_by == GroupBy.DATASET_SUITE:
        # Dataset suite grouping acts like "all" but replaces dataset with dataset_suite
        # This handles task name clashes across dataset suites
        dataset_suite_group_cols = [
            "dataset_suite",
            "agent_type",
            "agent_name",
            "attack_type",
        ]
        grouped = df.groupby(dataset_suite_group_cols)
        result = grouped[score_cols].mean().reset_index()
        # Add metadata columns
        result["n_tasks"] = grouped.size().values
        result["avg_n_samples"] = grouped["n_samples"].mean().values
        return result

    # Group by specific dimension
    # Handle agent_name explicitly
    if group_by == GroupBy.AGENT_NAME:
        group_col = "agent_name"
    elif group_by == GroupBy.DATASET:
        group_col = "dataset"
    else:
        group_col = f"{group_by}_type"

    if group_col not in df.columns:
        return df

    grouped = df.groupby(group_col)
    result = grouped[score_cols].mean().reset_index()
    # Add metadata columns
    result["n_tasks"] = grouped.size().values
    result["avg_n_samples"] = grouped["n_samples"].mean().values
    return result


def format_as_table(df: pd.DataFrame) -> str:
    """Format results DataFrame as a table.

    Args:
        df: Results DataFrame (already grouped as desired)

    Returns:
        Formatted table string
    """
    if df.empty:
        return "No results to display"

    display_df = df.copy()

    # Format numeric columns as floats (handle both pass@k and legacy column names)
    for col in display_df.columns:
        if col in ["benign_score", "attack_score"]:
            # Convert to numeric and format
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif col == "avg_n_samples":
            # Format average samples with 1 decimal place
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        elif col in ["n_tasks", "n_samples", "k"]:
            # Format as integers
            display_df[col] = display_df[col].apply(lambda x: str(int(x)) if pd.notna(x) else "N/A")

    return tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)


def format_as_json(df: pd.DataFrame) -> str:
    """Format results DataFrame as JSON.

    Args:
        df: Results DataFrame

    Returns:
        JSON string
    """
    if df.empty:
        return "[]"

    # Convert DataFrame to list of dicts
    results = df.to_dict(orient="records")
    return json.dumps(results, indent=2, default=str)


def format_as_csv(df: pd.DataFrame) -> str:
    """Format results DataFrame as CSV.

    Args:
        df: Results DataFrame

    Returns:
        CSV string
    """
    if df.empty:
        return ""

    return df.to_csv(index=False)


def format_results(
    results: pd.DataFrame,
    format: Format,  # noqa: A002 -- format name is fine for cli-friendliness
) -> str:
    match format:
        case Format.TABLE:
            return format_as_table(results)
        case Format.JSON:
            return format_as_json(results)
        case Format.CSV:
            return format_as_csv(results)
        case _:
            assert_never(format)

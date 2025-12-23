# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for result formatters."""

import json

import pandas as pd
import pytest
from prompt_siren.results import format_as_csv, format_as_json, format_as_table


@pytest.fixture
def sample_df_all_configs() -> pd.DataFrame:
    """Create sample aggregated results DataFrame with all config columns."""
    return pd.DataFrame(
        [
            {
                "dataset": "agentdojo",
                "agent_type": "plain-agent",
                "attack_type": "no-attack",
                "model": "azure:gpt-4",
                "job_name": "test_job_1",
                "benign_score": 0.5,
                "attack_score": 0.0,
            },
            {
                "dataset": "agentdojo",
                "agent_type": "plain-agent",
                "attack_type": "agentdojo",
                "model": "azure:gpt-4",
                "job_name": "test_job_2",
                "benign_score": 1.0,
                "attack_score": 0.8,
            },
        ]
    )


@pytest.fixture
def sample_df_grouped() -> pd.DataFrame:
    """Create sample grouped results DataFrame (e.g., grouped by env)."""
    return pd.DataFrame(
        [
            {
                "dataset": "agentdojo",
                "benign_score": 0.75,
                "attack_score": 0.4,
            },
        ]
    )


def test_format_as_table_with_all_columns(sample_df_all_configs: pd.DataFrame) -> None:
    """Test table formatting with all config columns."""
    output = format_as_table(sample_df_all_configs)

    assert "dataset" in output
    assert "agent_type" in output
    assert "attack_type" in output
    assert "model" in output
    assert "agentdojo" in output
    assert "plain-agent" in output
    assert "azure:gpt-4" in output


def test_format_as_table_grouped(sample_df_grouped: pd.DataFrame) -> None:
    """Test table formatting with grouped data."""
    output = format_as_table(sample_df_grouped)

    assert "dataset" in output
    assert "benign_score" in output
    assert "attack_score" in output
    assert "agentdojo" in output
    # Check that formatted scores are present (format may vary)
    assert "0.750" in output or "0.75" in output  # Formatted score


def test_format_as_json(sample_df_all_configs: pd.DataFrame) -> None:
    """Test JSON formatting."""
    output = format_as_json(sample_df_all_configs)

    # Parse and validate JSON
    data = json.loads(output)

    assert len(data) == 2
    assert data[0]["dataset"] == "agentdojo"
    assert data[0]["agent_type"] == "plain-agent"
    assert data[0]["attack_type"] == "no-attack"
    assert data[0]["model"] == "azure:gpt-4"
    assert data[0]["benign_score"] == pytest.approx(0.5)


def test_format_as_csv(sample_df_all_configs: pd.DataFrame) -> None:
    """Test CSV formatting."""
    output = format_as_csv(sample_df_all_configs)

    lines = output.strip().split("\n")

    # Check header
    assert "dataset" in lines[0]
    assert "agent_type" in lines[0]
    assert "attack_type" in lines[0]
    assert "model" in lines[0]

    # Check data rows
    assert len(lines) == 3  # Header + 2 data rows
    assert "agentdojo" in lines[1]
    assert "plain-agent" in lines[1]


def test_format_empty_results() -> None:
    """Test formatting with empty results."""
    empty_df = pd.DataFrame()

    table = format_as_table(empty_df)
    json_output = format_as_json(empty_df)
    csv_output = format_as_csv(empty_df)

    assert "No results to display" in table
    assert json_output == "[]"
    assert csv_output == ""


def test_format_with_none_values() -> None:
    """Test formatting when metrics are None."""
    df = pd.DataFrame(
        [
            {
                "dataset": "env",
                "agent_type": "agent",
                "attack_type": "attack",
                "model": "model",
                "job_name": "test_job",
                "benign_score": None,
                "attack_score": None,
            }
        ]
    )

    table = format_as_table(df)
    json_output = format_as_json(df)
    csv_output = format_as_csv(df)

    assert "N/A" in table
    assert "null" in json_output
    assert csv_output != ""

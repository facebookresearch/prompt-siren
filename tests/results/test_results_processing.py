"""Tests for results processing with template_short_name."""

import math
from datetime import datetime
from pathlib import Path

from prompt_siren.config.experiment_config import (
    AgentConfig,
    AttackConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    TelemetryConfig,
)
from prompt_siren.job.models import JobConfig, RunIndexEntry
from prompt_siren.results import _parse_index_entry


def _create_job_config(
    attack_type: str | None = None,
    attack_config: dict | None = None,
) -> JobConfig:
    """Create a JobConfig for testing."""
    return JobConfig(
        job_name="test_job",
        execution_mode="benign" if attack_type is None else "attack",
        created_at=datetime.now(),
        dataset=DatasetConfig(type="test_dataset", config={}),
        agent=AgentConfig(type="plain", config={"model": "test_agent"}),
        attack=AttackConfig(type=attack_type, config=attack_config or {}) if attack_type else None,
        execution=ExecutionConfig(concurrency=1),
        telemetry=TelemetryConfig(trace_console=False),
        output=OutputConfig(jobs_dir=Path("jobs")),
    )


def _create_index_entry(
    benign_score: float = 1.0,
    attack_score: float | None = None,
) -> RunIndexEntry:
    """Create a RunIndexEntry for testing."""
    return RunIndexEntry(
        task_id="test_task",
        run_id="abc12345",
        timestamp=datetime.now(),
        benign_score=benign_score,
        attack_score=attack_score,
        exception_type=None,
        path=Path("test_task/abc12345"),
    )


class TestResultsProcessingWithTemplateShortName:
    """Tests for attack type naming with template_short_name."""

    def test_benign_attack_type(self):
        """Test that benign runs have attack_type set to 'benign'."""
        job_config = _create_job_config(attack_type=None)
        entry = _create_index_entry()

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_type is set to benign
        assert parsed["attack_type"] == "benign"

    def test_template_string_attack_with_default_short_name(self):
        """Test that template_string attack with default short name gets suffix."""
        job_config = _create_job_config(
            attack_type="template_string",
            attack_config={"template_short_name": "default", "attack_template": "test"},
        )
        entry = _create_index_entry(benign_score=0.5, attack_score=0.8)

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_type includes the template_short_name
        assert parsed["attack_type"] == "template_string_default"

    def test_template_string_attack_with_custom_short_name(self):
        """Test that template_string attack with custom short name gets suffix."""
        job_config = _create_job_config(
            attack_type="template_string",
            attack_config={
                "template_short_name": "urgent",
                "attack_template": "URGENT: {{ goal }}",
            },
        )
        entry = _create_index_entry(benign_score=0.5, attack_score=0.9)

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_type includes the custom template_short_name
        assert parsed["attack_type"] == "template_string_urgent"

    def test_template_string_attack_without_short_name(self):
        """Test that template_string attack without short name in config stays as-is."""
        job_config = _create_job_config(
            attack_type="template_string",
            attack_config={"attack_template": "test"},  # No template_short_name
        )
        entry = _create_index_entry(benign_score=0.5, attack_score=0.7)

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_type stays as template_string (no suffix)
        assert parsed["attack_type"] == "template_string"

    def test_template_string_attack_with_empty_config(self):
        """Test that template_string attack with empty config stays as-is."""
        job_config = _create_job_config(
            attack_type="template_string",
            attack_config={},
        )
        entry = _create_index_entry(benign_score=0.5, attack_score=0.7)

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_type stays as template_string (no suffix)
        assert parsed["attack_type"] == "template_string"

    def test_other_attack_types_unchanged(self):
        """Test that other attack types are not affected by template_short_name logic."""
        for attack_type in ["dict", "mini-goat", "custom_attack"]:
            job_config = _create_job_config(
                attack_type=attack_type,
                attack_config={"some_param": "value"},
            )
            entry = _create_index_entry(benign_score=0.5, attack_score=0.6)

            json_str = entry.model_dump_json()
            parsed = _parse_index_entry(json_str, job_config)

            # Verify attack_type stays unchanged
            assert parsed["attack_type"] == attack_type

    def test_multiple_template_variants_in_results(self):
        """Test that multiple template variants can be distinguished in results."""
        variants = [
            ("variant1", "template_string_variant1"),
            ("variant2", "template_string_variant2"),
            ("variant3", "template_string_variant3"),
        ]

        for short_name, expected_type in variants:
            job_config = _create_job_config(
                attack_type="template_string",
                attack_config={
                    "template_short_name": short_name,
                    "attack_template": f"Template {short_name}: {{goal}}",
                },
            )
            entry = _create_index_entry(benign_score=0.5, attack_score=0.7)

            json_str = entry.model_dump_json()
            parsed = _parse_index_entry(json_str, job_config)

            # Verify each variant has a unique attack_type
            assert parsed["attack_type"] == expected_type

    def test_attack_score_nan_conversion(self):
        """Test that None attack_score is converted to NaN."""
        job_config = _create_job_config(attack_type=None)
        entry = _create_index_entry(benign_score=1.0, attack_score=None)

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str, job_config)

        # Verify attack_score is converted to NaN
        assert math.isnan(parsed["attack_score"])

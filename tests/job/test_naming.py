# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for job naming utilities."""

import re
from datetime import datetime

from prompt_siren.job.naming import (
    generate_job_name,
    sanitize_for_filename,
)


class TestSanitizeForFilename:
    """Tests for sanitize_for_filename function."""

    def test_replaces_problematic_characters(self):
        """Test that colons, slashes, and spaces are replaced."""
        assert sanitize_for_filename("plain:gpt-5") == "plain_gpt-5"
        assert sanitize_for_filename("azure/gpt-5-turbo") == "azure_gpt-5-turbo"
        assert sanitize_for_filename("my model name") == "my_model_name"

    def test_collapses_multiple_underscores(self):
        """Test that multiple consecutive underscores are collapsed."""
        assert sanitize_for_filename("a::b//c  d") == "a_b_c_d"

    def test_strips_leading_trailing_underscores(self):
        """Test that leading and trailing underscores are removed."""
        assert sanitize_for_filename(":leading") == "leading"
        assert sanitize_for_filename("trailing:") == "trailing"

    def test_handles_special_characters(self):
        """Test that special characters are replaced."""
        assert sanitize_for_filename("model@v1.2.3") == "model_v1_2_3"

    def test_complex_model_names(self):
        """Test sanitization of realistic model names."""
        assert sanitize_for_filename("bedrock:anthropic.claude-3") == "bedrock_anthropic_claude-3"
        assert sanitize_for_filename("openai/gpt-4-turbo-2024") == "openai_gpt-4-turbo-2024"


class TestGenerateJobName:
    """Tests for generate_job_name function."""

    def test_generates_name_with_attack(self):
        """Test generating job name with attack type."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="agentdojo-workspace",
            agent_name="plain:gpt-5",
            attack_type="template_string",
            timestamp=timestamp,
        )
        assert name == "agentdojo-workspace_plain_gpt-5_template_string_2025-01-15_14-30-00"

    def test_generates_name_without_attack_uses_benign(self):
        """Test generating job name for benign runs uses 'benign' placeholder."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="agentdojo-workspace",
            agent_name="plain:gpt-5",
            attack_type=None,
            timestamp=timestamp,
        )
        assert name == "agentdojo-workspace_plain_gpt-5_benign_2025-01-15_14-30-00"

    def test_sanitizes_all_components(self):
        """Test that all components are sanitized."""
        timestamp = datetime(2025, 1, 15, 14, 30, 0)
        name = generate_job_name(
            dataset_type="dataset:with:colons",
            agent_name="bedrock:anthropic/claude-3",
            attack_type="attack:type",
            timestamp=timestamp,
        )
        assert "dataset_with_colons" in name
        assert "bedrock_anthropic_claude-3" in name
        assert "attack_type" in name

    def test_uses_current_time_when_no_timestamp(self):
        """Test that current time is used when timestamp not provided."""
        name = generate_job_name(
            dataset_type="dataset",
            agent_name="agent",
            attack_type=None,
        )
        # Verify the name ends with a timestamp pattern
        timestamp_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        assert re.search(timestamp_pattern, name) is not None

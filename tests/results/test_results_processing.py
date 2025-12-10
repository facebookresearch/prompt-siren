"""Tests for results processing with template_short_name."""

from pathlib import Path

from prompt_siren.results import _parse_index_entry
from prompt_siren.run_persistence import IndexEntry


class TestResultsProcessingWithTemplateShortName:
    """Tests for attack type naming with template_short_name."""

    def test_benign_attack_type(self):
        """Test that benign runs have attack_type set to 'benign'."""
        # Create an index entry for a benign run
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type=None,
            attack_config=None,
            config_hash="abc123",
            benign_score=1.0,
            attack_score=None,
            path=Path("outputs.jsonl"),
        )

        # Serialize to JSON string (like what's in the index file)
        json_str = entry.model_dump_json()

        # Parse the entry
        parsed = _parse_index_entry(json_str)

        # Verify attack_type is set to benign
        assert parsed["attack_type"] == "benign"

    def test_template_string_attack_with_default_short_name(self):
        """Test that template_string attack with default short name gets suffix."""
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type="template_string",
            attack_config={"template_short_name": "default", "attack_template": "test"},
            config_hash="abc123",
            benign_score=0.5,
            attack_score=0.8,
            path=Path("outputs.jsonl"),
        )

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str)

        # Verify attack_type includes the template_short_name
        assert parsed["attack_type"] == "template_string_default"

    def test_template_string_attack_with_custom_short_name(self):
        """Test that template_string attack with custom short name gets suffix."""
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type="template_string",
            attack_config={
                "template_short_name": "urgent",
                "attack_template": "URGENT: {{ goal }}",
            },
            config_hash="abc123",
            benign_score=0.5,
            attack_score=0.9,
            path=Path("outputs.jsonl"),
        )

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str)

        # Verify attack_type includes the custom template_short_name
        assert parsed["attack_type"] == "template_string_urgent"

    def test_template_string_attack_without_short_name(self):
        """Test that template_string attack without short name in config stays as-is."""
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type="template_string",
            attack_config={"attack_template": "test"},  # No template_short_name
            config_hash="abc123",
            benign_score=0.5,
            attack_score=0.7,
            path=Path("outputs.jsonl"),
        )

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str)

        # Verify attack_type stays as template_string (no suffix)
        assert parsed["attack_type"] == "template_string"

    def test_template_string_attack_with_none_config(self):
        """Test that template_string attack with None config stays as-is."""
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type="template_string",
            attack_config=None,
            config_hash="abc123",
            benign_score=0.5,
            attack_score=0.7,
            path=Path("outputs.jsonl"),
        )

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str)

        # Verify attack_type stays as template_string (no suffix)
        assert parsed["attack_type"] == "template_string"

    def test_other_attack_types_unchanged(self):
        """Test that other attack types are not affected by template_short_name logic."""
        for attack_type in ["dict", "mini-goat", "custom_attack"]:
            entry = IndexEntry(
                execution_id="exec123",
                task_id="test_task",
                timestamp="2024-01-01T00:00:00",
                dataset="test_dataset",
                dataset_config={},
                agent_type="plain",
                agent_name="test_agent",
                attack_type=attack_type,
                attack_config={"some_param": "value"},
                config_hash="abc123",
                benign_score=0.5,
                attack_score=0.6,
                path=Path("outputs.jsonl"),
            )

            json_str = entry.model_dump_json()
            parsed = _parse_index_entry(json_str)

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
            entry = IndexEntry(
                execution_id="exec123",
                task_id="test_task",
                timestamp="2024-01-01T00:00:00",
                dataset="test_dataset",
                dataset_config={},
                agent_type="plain",
                agent_name="test_agent",
                attack_type="template_string",
                attack_config={
                    "template_short_name": short_name,
                    "attack_template": f"Template {short_name}: {{goal}}",
                },
                config_hash="abc123",
                benign_score=0.5,
                attack_score=0.7,
                path=Path("outputs.jsonl"),
            )

            json_str = entry.model_dump_json()
            parsed = _parse_index_entry(json_str)

            # Verify each variant has a unique attack_type
            assert parsed["attack_type"] == expected_type

    def test_attack_score_nan_conversion(self):
        """Test that None attack_score is converted to NaN."""
        entry = IndexEntry(
            execution_id="exec123",
            task_id="test_task",
            timestamp="2024-01-01T00:00:00",
            dataset="test_dataset",
            dataset_config={},
            agent_type="plain",
            agent_name="test_agent",
            attack_type=None,
            attack_config=None,
            config_hash="abc123",
            benign_score=1.0,
            attack_score=None,
            path=Path("outputs.jsonl"),
        )

        json_str = entry.model_dump_json()
        parsed = _parse_index_entry(json_str)

        # Verify attack_score is converted to NaN
        import math

        assert math.isnan(parsed["attack_score"])

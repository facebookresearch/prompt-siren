# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench prompt loader and Jinja2 template rendering."""

import pytest
from prompt_siren.datasets.swebench_dataset.prompts.loader import (
    BUILTIN_TEMPLATES,
    format_task_prompt_from_template,
    load_prompt_template,
)
from swebench.harness.constants import SWEbenchInstance


@pytest.fixture
def minimal_instance() -> SWEbenchInstance:
    """Fixture providing a minimal valid SWEbenchInstance for testing."""
    return SWEbenchInstance(
        repo="test/repo",
        instance_id="test-1",
        base_commit="abc123",
        patch="",
        test_patch="",
        problem_statement="Fix bug",
        hints_text="",
        created_at="2024-01-01",
        version="1.0",
        FAIL_TO_PASS="[]",
        PASS_TO_PASS="[]",
        environment_setup_commit="abc123",
    )


@pytest.fixture
def instance_with_hints() -> SWEbenchInstance:
    """Fixture providing a SWEbenchInstance with hints for testing."""
    return SWEbenchInstance(
        repo="test/repo",
        instance_id="test-1",
        base_commit="abc123",
        patch="",
        test_patch="",
        problem_statement="Fix bug",
        hints_text="Check the authentication module",
        created_at="2024-01-01",
        version="1.0",
        FAIL_TO_PASS="[]",
        PASS_TO_PASS="[]",
        environment_setup_commit="abc123",
    )


class TestLoadPromptTemplate:
    """Tests for load_prompt_template function."""

    def test_load_builtin_mini_swe_agent(self):
        """Test loading built-in mini-swe-agent template."""
        template = load_prompt_template("mini-swe-agent")
        assert "instance_template" in template
        assert "{{problem_statement}}" in template["instance_template"]
        assert "Recommended Workflow" in template["instance_template"]

    def test_load_builtin_swe_agent_swebench(self):
        """Test loading built-in swe-agent-swebench template."""
        template = load_prompt_template("swe-agent-swebench")
        assert "instance_template" in template
        assert "{{problem_statement}}" in template["instance_template"]
        assert "Important Boundaries" in template["instance_template"]

    def test_load_custom_file(self, tmp_path):
        """Test loading custom YAML file from path."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("instance_template: 'Test {{problem_statement}}'")

        template = load_prompt_template(str(custom_yaml))
        assert template["instance_template"] == "Test {{problem_statement}}"

    def test_load_custom_file_missing_key(self, tmp_path):
        """Test error when custom YAML file doesn't have instance_template."""
        custom_yaml = tmp_path / "invalid.yaml"
        custom_yaml.write_text("other_key: 'some value'")

        with pytest.raises(KeyError, match="must contain 'instance_template'"):
            load_prompt_template(str(custom_yaml))

    def test_load_invalid_name(self):
        """Test error on invalid template name."""
        with pytest.raises(ValueError, match="Unknown prompt template"):
            load_prompt_template("nonexistent-template")

    def test_builtin_templates_constant(self):
        """Test that BUILTIN_TEMPLATES constant is correct."""
        assert "mini-swe-agent" in BUILTIN_TEMPLATES
        assert "swe-agent-swebench" in BUILTIN_TEMPLATES
        assert len(BUILTIN_TEMPLATES) == 2


class TestFormatTaskPromptFromTemplate:
    """Tests for format_task_prompt_from_template function."""

    def test_format_mini_swe_agent_basic(self):
        """Test formatting with mini-swe-agent template."""
        instance = SWEbenchInstance(
            problem_statement="Fix the authentication bug",
            repo="django/django",
            instance_id="django__django-12345",
            base_commit="abc123def456",
            patch="",
            test_patch="",
            hints_text="",
            created_at="2024-01-01",
            version="1.0",
            FAIL_TO_PASS="[]",
            PASS_TO_PASS="[]",
            environment_setup_commit="abc123def456",
        )
        result = format_task_prompt_from_template("mini-swe-agent", instance)
        assert "Fix the authentication bug" in result
        assert "Recommended Workflow" in result
        assert "Please solve this issue:" in result

    def test_format_swe_agent_swebench_basic(self):
        """Test formatting with swe-agent-swebench template."""
        instance = SWEbenchInstance(
            problem_statement="Add new feature X",
            repo="python/cpython",
            instance_id="python__cpython-67890",
            base_commit="def456abc789",
            patch="",
            test_patch="",
            hints_text="",
            created_at="2024-01-01",
            version="1.0",
            FAIL_TO_PASS="[]",
            PASS_TO_PASS="[]",
            environment_setup_commit="def456abc789",
        )
        result = format_task_prompt_from_template("swe-agent-swebench", instance)
        assert "Add new feature X" in result
        assert "Important Boundaries" in result
        assert "<pr_description>" in result

    def test_format_without_hints(self, instance_with_hints: SWEbenchInstance):
        """Test that hints are not included when include_hints=False."""
        result = format_task_prompt_from_template(
            "mini-swe-agent", instance_with_hints, include_hints=False
        )
        assert "Fix bug" in result
        # Hints should not appear (mini-swe-agent doesn't have hints section anyway)
        # But the variable should be empty string
        assert "Check the authentication module" not in result

    def test_format_with_hints(self, instance_with_hints: SWEbenchInstance):
        """Test that hints can be included with custom template."""
        # For this test, we'll verify that the hints_text variable is passed
        # even though mini-swe-agent template doesn't use it
        result = format_task_prompt_from_template(
            "mini-swe-agent", instance_with_hints, include_hints=True
        )
        assert "Fix bug" in result
        # The template doesn't display hints, but no error should occur

    def test_format_with_custom_template(self, tmp_path):
        """Test formatting with a custom template file."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(
            """instance_template: |
  Problem: {{problem_statement}}
  Repository: {{repo}}
  ID: {{instance_id}}
  {%- if hints_text %}
  Hints: {{hints_text}}
  {%- endif %}
"""
        )

        instance = SWEbenchInstance(
            problem_statement="Add feature",
            repo="test/repo",
            instance_id="test-1",
            base_commit="abc123",
            hints_text="Use TDD",
            patch="",
            test_patch="",
            created_at="2024-01-01",
            version="1.0",
            FAIL_TO_PASS="[]",
            PASS_TO_PASS="[]",
            environment_setup_commit="abc123",
        )

        result = format_task_prompt_from_template(str(custom_yaml), instance, include_hints=True)
        assert "Problem: Add feature" in result
        assert "Repository: test/repo" in result
        assert "ID: test-1" in result
        assert "Hints: Use TDD" in result

    def test_format_with_empty_hints(self, minimal_instance: SWEbenchInstance):
        """Test formatting when hints_text is empty string."""
        result = format_task_prompt_from_template(
            "mini-swe-agent", minimal_instance, include_hints=True
        )
        assert "Fix bug" in result
        # Should handle empty hints gracefully

    def test_format_preserves_newlines_in_problem_statement(self):
        """Test that newlines in problem_statement are preserved."""
        instance = SWEbenchInstance(
            problem_statement="Fix bug:\n1. First issue\n2. Second issue",
            repo="test/repo",
            instance_id="test-1",
            base_commit="abc123",
            patch="",
            test_patch="",
            hints_text="",
            created_at="2024-01-01",
            version="1.0",
            FAIL_TO_PASS="[]",
            PASS_TO_PASS="[]",
            environment_setup_commit="abc123",
        )
        result = format_task_prompt_from_template("mini-swe-agent", instance)
        assert "1. First issue" in result
        assert "2. Second issue" in result


class TestTemplateContent:
    """Tests to verify the content of built-in templates."""

    def test_mini_swe_agent_has_key_sections(self):
        """Test that mini-swe-agent template has all expected sections."""
        template = load_prompt_template("mini-swe-agent")
        content = template["instance_template"]

        # Check for key sections
        assert "Recommended Workflow" in content
        assert "Important Rules" in content
        assert "Formatting your response" in content
        assert "Useful command examples" in content
        assert "Create a new file:" in content
        assert "Edit files with sed:" in content

    def test_swe_agent_swebench_has_key_sections(self):
        """Test that swe-agent-swebench template has all expected sections."""
        template = load_prompt_template("swe-agent-swebench")
        content = template["instance_template"]

        # Check for key sections
        assert "Overview" in content
        assert "Important Boundaries" in content
        assert "Recommended Workflow" in content
        assert "Command Execution Rules" in content
        assert "Environment Details" in content
        assert "DO NOT MODIFY: Tests" in content  # Important boundary

    def test_both_templates_use_problem_statement_variable(self):
        """Test that both templates use {{problem_statement}} variable."""
        mini_template = load_prompt_template("mini-swe-agent")
        swebench_template = load_prompt_template("swe-agent-swebench")

        assert "{{problem_statement}}" in mini_template["instance_template"]
        assert "{{problem_statement}}" in swebench_template["instance_template"]

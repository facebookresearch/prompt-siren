"""Tests for the template string attack functionality."""

import pytest
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from prompt_siren.attacks import (
    create_attack,
    get_registered_attacks,
)
from prompt_siren.attacks.registry import attack_registry
from prompt_siren.attacks.template_string_attack import (
    create_template_string_attack,
    TemplateStringAttack,
    TemplateStringAttackConfig,
)


class TestTemplateStringAttackConfig:
    """Tests for TemplateStringAttackConfig."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = TemplateStringAttackConfig()

        # Check default template exists and is non-empty
        assert config.attack_template
        assert "{{ goal }}" in config.attack_template

        # Check default template_short_name
        assert config.template_short_name == "default"

    def test_custom_config(self):
        """Test creating config with custom values."""
        custom_template = "Custom template: {{ goal }} for {{ user }}"
        config = TemplateStringAttackConfig(
            attack_template=custom_template, template_short_name="custom"
        )

        assert config.attack_template == custom_template
        assert config.template_short_name == "custom"

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = TemplateStringAttackConfig(
            attack_template="Test {{ goal }}", template_short_name="test"
        )

        # Serialize to dict
        config_dict = config.model_dump()
        assert config_dict["attack_template"] == "Test {{ goal }}"
        assert config_dict["template_short_name"] == "test"

        # Deserialize from dict
        loaded_config = TemplateStringAttackConfig.model_validate(config_dict)
        assert loaded_config.attack_template == "Test {{ goal }}"
        assert loaded_config.template_short_name == "test"


class TestTemplateStringAttack:
    """Tests for TemplateStringAttack class."""

    def test_attack_name(self):
        """Test that attack has correct name."""
        attack = TemplateStringAttack()
        assert attack.name == "template_string"

    def test_attack_with_custom_config(self):
        """Test creating attack with custom config."""
        config = TemplateStringAttackConfig(
            attack_template="Custom: {{ goal }}", template_short_name="custom"
        )
        attack = TemplateStringAttack(_config=config)

        assert attack.config.attack_template == "Custom: {{ goal }}"
        assert attack.config.template_short_name == "custom"

    def test_factory_function(self):
        """Test the factory function creates correct attack instance."""
        config = TemplateStringAttackConfig(
            attack_template="Factory test: {{ goal }}", template_short_name="factory"
        )
        attack = create_template_string_attack(config)

        assert isinstance(attack, TemplateStringAttack)
        assert attack.config.attack_template == "Factory test: {{ goal }}"
        assert attack.config.template_short_name == "factory"


class TestTemplateStringAttackRegistry:
    """Tests for template string attack registration."""

    def setup_method(self):
        """Reset registry for clean tests."""
        attack_registry._registry.clear()
        attack_registry._entry_points_loaded = False

    def test_template_string_registered(self):
        """Test that template_string attack is registered via entry points."""
        registered_attacks = get_registered_attacks()
        assert "template_string" in registered_attacks

    def test_create_via_template_string_type(self):
        """Test creating attack using template_string type."""
        config = TemplateStringAttackConfig(
            attack_template="Test {{ goal }}", template_short_name="test"
        )
        attack = create_attack("template_string", config)

        assert isinstance(attack, TemplateStringAttack)
        assert attack.config.template_short_name == "test"


class TestTemplateShortNameInAttackType:
    """Tests for template_short_name integration in attack type naming."""

    def test_template_short_name_default(self):
        """Test that default template_short_name is set correctly."""
        config = TemplateStringAttackConfig()
        attack = TemplateStringAttack(_config=config)

        # The default should be "default"
        assert attack.config.template_short_name == "default"

    def test_multiple_template_variants(self):
        """Test creating multiple attacks with different template_short_name values."""
        # Create attacks with different template names
        configs = [
            TemplateStringAttackConfig(
                attack_template="Template 1: {{ goal }}", template_short_name="variant1"
            ),
            TemplateStringAttackConfig(
                attack_template="Template 2: {{ goal }}", template_short_name="variant2"
            ),
            TemplateStringAttackConfig(
                attack_template="Template 3: {{ goal }}", template_short_name="variant3"
            ),
        ]

        attacks = [TemplateStringAttack(_config=config) for config in configs]

        # Verify each has the correct template_short_name
        assert attacks[0].config.template_short_name == "variant1"
        assert attacks[1].config.template_short_name == "variant2"
        assert attacks[2].config.template_short_name == "variant3"


class TestJinja2TemplateRendering:
    """Tests for Jinja2 template rendering logic.

    These tests verify that templates are correctly rendered
    with default and custom fields.
    """

    def test_basic_jinja2_rendering(self):
        """Test that basic Jinja2 template is rendered correctly."""
        config = TemplateStringAttackConfig(
            attack_template="Task: {{ goal }} for {{ user }} on {{ model }}",
            template_fields={"user": "TestUser", "model": "TestModel"},
        )

        # Render template with goal and custom fields
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {"goal": "test_goal", **config.template_fields}
        result = template.render(**context)

        # Check for complete rendered string
        assert result == "Task: test_goal for TestUser on TestModel"

    def test_jinja2_with_custom_fields(self):
        """Test Jinja2 rendering with custom template fields."""
        config = TemplateStringAttackConfig(
            attack_template="{% if urgent %}URGENT: {% endif %}{{ goal }} (Priority: {{ priority }})",
            template_fields={"urgent": True, "priority": "HIGH"},
        )

        # Render with both default and custom fields
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {"goal": "steal_data", **config.template_fields}
        result = template.render(**context)

        # Check for complete rendered string
        assert result == "URGENT: steal_data (Priority: HIGH)"

    def test_jinja2_loop_rendering(self):
        """Test Jinja2 template with loop constructs."""
        config = TemplateStringAttackConfig(
            attack_template="{% for step in steps %}{{ loop.index }}. {{ step }}\n{% endfor %}Goal: {{ goal }}",
            template_fields={"steps": ["Connect", "Extract", "Exfiltrate"]},
        )

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {"goal": "complete_task", **config.template_fields}
        result = template.render(**context)

        # Check for complete rendered string
        expected = "1. Connect\n2. Extract\n3. Exfiltrate\nGoal: complete_task"
        assert result == expected

    def test_jinja2_nested_data_rendering(self):
        """Test Jinja2 template with nested data structures."""
        config = TemplateStringAttackConfig(
            attack_template="Access {{ system.name }} version {{ system.version }} to achieve {{ goal }}.",
            template_fields={"system": {"name": "DatabaseX", "version": "v2.5.1"}},
        )

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {"goal": "access_db", **config.template_fields}
        result = template.render(**context)

        # Check for complete rendered string
        assert result == "Access DatabaseX version v2.5.1 to achieve access_db."

    def test_jinja2_filter_rendering(self):
        """Test Jinja2 template with filters."""
        config = TemplateStringAttackConfig(
            attack_template="Goal: {{ goal | upper }} (Count: {{ count | default(42) }})",
            template_fields={},
        )

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {"goal": "execute_command", **config.template_fields}
        result = template.render(**context)

        # Check for complete rendered string with filters applied
        assert result == "Goal: EXECUTE_COMMAND (Count: 42)"

    def test_invalid_jinja2_syntax_raises_error(self):
        """Test that invalid Jinja2 syntax raises TemplateSyntaxError."""
        config = TemplateStringAttackConfig(
            attack_template="{{ goal }} {% if %}invalid syntax{% endif %}"
        )

        # The error should be raised when we try to parse the template
        env = Environment(undefined=StrictUndefined)
        with pytest.raises(TemplateSyntaxError):
            env.from_string(config.attack_template)

    def test_undefined_variable_with_strict_mode(self):
        """Test that undefined variables raise errors with StrictUndefined."""
        config = TemplateStringAttackConfig(
            attack_template="Goal: {{ goal }}, User: {{ undefined_variable }}"
        )

        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)

        # Should raise UndefinedError when trying to render with missing variable
        with pytest.raises(UndefinedError):
            template.render(goal="test_goal")

    def test_custom_field_overrides_default(self):
        """Test that custom fields can override default fields."""
        config = TemplateStringAttackConfig(
            attack_template="User: {{ user }}, Model: {{ model }}",
            template_fields={"user": "CustomUser", "model": "CustomModel"},
        )

        # Simulate merging default and custom fields (as done in attack method)
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(config.attack_template)
        context = {
            "user": "Emma Johnson",  # Default
            "model": "DefaultModel",  # Default
            **config.template_fields,  # Custom fields override
        }
        result = template.render(**context)

        # CustomUser and CustomModel from template_fields should be used
        assert result == "User: CustomUser, Model: CustomModel"

"""Tests for the template string attack functionality."""

from prompt_siren.attacks import (
    create_attack,
    get_attack_config_class,
    get_registered_attacks,
)
from prompt_siren.attacks.agentdojo_attack import (
    create_template_string_attack,
    TemplateStringAttack,
    TemplateStringAttackConfig,
)
from prompt_siren.attacks.registry import attack_registry


class TestTemplateStringAttackConfig:
    """Tests for TemplateStringAttackConfig."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = TemplateStringAttackConfig()

        # Check default template exists and is non-empty
        assert config.attack_template
        assert "{goal}" in config.attack_template

        # Check default template_short_name
        assert config.template_short_name == "default"

    def test_custom_config(self):
        """Test creating config with custom values."""
        custom_template = "Custom template: {goal} for {user}"
        config = TemplateStringAttackConfig(
            attack_template=custom_template, template_short_name="custom"
        )

        assert config.attack_template == custom_template
        assert config.template_short_name == "custom"

    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = TemplateStringAttackConfig(
            attack_template="Test {goal}", template_short_name="test"
        )

        # Serialize to dict
        config_dict = config.model_dump()
        assert config_dict["attack_template"] == "Test {goal}"
        assert config_dict["template_short_name"] == "test"

        # Deserialize from dict
        loaded_config = TemplateStringAttackConfig.model_validate(config_dict)
        assert loaded_config.attack_template == "Test {goal}"
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
            attack_template="Custom: {goal}", template_short_name="custom"
        )
        attack = TemplateStringAttack(_config=config)

        assert attack.config.attack_template == "Custom: {goal}"
        assert attack.config.template_short_name == "custom"

    def test_factory_function(self):
        """Test the factory function creates correct attack instance."""
        config = TemplateStringAttackConfig(
            attack_template="Factory test: {goal}", template_short_name="factory"
        )
        attack = create_template_string_attack(config)

        assert isinstance(attack, TemplateStringAttack)
        assert attack.config.attack_template == "Factory test: {goal}"
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

    def test_agentdojo_backwards_compatibility(self):
        """Test that agentdojo entry point still exists for backwards compatibility."""
        registered_attacks = get_registered_attacks()
        assert "agentdojo" in registered_attacks

        # Both should resolve to the same config class
        template_config_class = get_attack_config_class("template_string")
        agentdojo_config_class = get_attack_config_class("agentdojo")
        assert template_config_class is agentdojo_config_class
        assert template_config_class is TemplateStringAttackConfig

    def test_create_via_template_string_type(self):
        """Test creating attack using template_string type."""
        config = TemplateStringAttackConfig(
            attack_template="Test {goal}", template_short_name="test"
        )
        attack = create_attack("template_string", config)

        assert isinstance(attack, TemplateStringAttack)
        assert attack.config.template_short_name == "test"

    def test_create_via_agentdojo_type(self):
        """Test creating attack using agentdojo type (backwards compatibility)."""
        config = TemplateStringAttackConfig(
            attack_template="Test {goal}", template_short_name="legacy"
        )
        attack = create_attack("agentdojo", config)

        assert isinstance(attack, TemplateStringAttack)
        assert attack.config.template_short_name == "legacy"


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
                attack_template="Template 1: {goal}", template_short_name="variant1"
            ),
            TemplateStringAttackConfig(
                attack_template="Template 2: {goal}", template_short_name="variant2"
            ),
            TemplateStringAttackConfig(
                attack_template="Template 3: {goal}", template_short_name="variant3"
            ),
        ]

        attacks = [TemplateStringAttack(_config=config) for config in configs]

        # Verify each has the correct template_short_name
        assert attacks[0].config.template_short_name == "variant1"
        assert attacks[1].config.template_short_name == "variant2"
        assert attacks[2].config.template_short_name == "variant3"

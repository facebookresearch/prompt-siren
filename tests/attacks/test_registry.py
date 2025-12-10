# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Simple tests for the attack plugin system."""

import pytest
from prompt_siren.attacks import (
    create_attack,
    get_attack_config_class,
    get_registered_attacks,
)
from prompt_siren.attacks.registry import attack_registry
from prompt_siren.attacks.template_string_attack import TemplateStringAttackConfig
from prompt_siren.registry_base import UnknownComponentError

from ..conftest import create_mock_attack, MockAttack, MockAttackConfig


class TestAttackRegistry:
    """Tests for the attack plugin system."""

    def setup_method(self):
        """Set up test environment by clearing registry and registering mock attack."""
        # Clear the registry for clean tests
        attack_registry._registry.clear()
        attack_registry._entry_points_loaded = False

        # Manually register mock attack for testing
        attack_registry.register("mock", MockAttackConfig, create_mock_attack)

    def test_register_and_get_config_class(self):
        """Test registering an attack and retrieving its config class."""
        # Check that it's listed in registered attacks
        assert "mock" in get_registered_attacks()

        # Get the config class
        config_class = get_attack_config_class("mock")

        # Check that it's the correct class
        assert config_class == MockAttackConfig

    def test_entry_point_discovery(self):
        """Test that entry points are discovered automatically."""
        # Clear manual registrations
        attack_registry._registry.clear()
        attack_registry._entry_points_loaded = False

        # Check that built-in attacks are discovered via entry points
        registered_attacks = get_registered_attacks()
        assert "template_string" in registered_attacks

        # Test that we can get the config class (using template_string)
        config_class = get_attack_config_class("template_string")
        assert config_class is TemplateStringAttackConfig

    def test_create_attack(self):
        """Test creating an attack from config."""
        # Create a config
        config = MockAttackConfig(name="Custom Mock Attack", custom_parameter="test-value")

        # Create the attack
        attack = create_attack("mock", config)

        # Check that it's the correct type
        assert isinstance(attack, MockAttack)

        # Check that it has the correct values
        assert attack.attack_name == "Custom Mock Attack"
        assert attack.custom_parameter == "test-value"

        # Check that config property returns the original config
        config_obj = attack.config
        assert config_obj == config
        assert isinstance(config_obj, MockAttackConfig)
        assert config_obj.name == "Custom Mock Attack"
        assert config_obj.custom_parameter == "test-value"

    def test_missing_attack_type(self):
        """Test error when requesting an unregistered attack type."""
        with pytest.raises(UnknownComponentError):
            get_attack_config_class("non-existent-attack")

        config = MockAttackConfig()
        with pytest.raises(UnknownComponentError):
            create_attack("non-existent-attack", config)

    def test_duplicate_registration(self):
        """Test error when attempting to register the same attack type twice."""
        # Attempt to register the same type again should raise ValueError
        with pytest.raises(ValueError, match="Attack type 'mock' is already registered"):
            attack_registry.register("mock", MockAttackConfig, create_mock_attack)

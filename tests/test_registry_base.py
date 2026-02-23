# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for BaseRegistry component class storage and error handling."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from prompt_siren.registry_base import BaseRegistry, ComponentEntryPoint
from pydantic import BaseModel


class DummyConfig(BaseModel):
    value: str = "default"


class DummyComponent:
    pass


def dummy_factory(config: DummyConfig) -> DummyComponent:
    return DummyComponent()


class TestTupleEntryPointHandling:
    """Tests for entry point formats: ComponentEntryPoint / plain tuple vs plain factory."""

    def _make_entry_point(self, name: str, load_result: object) -> MagicMock:
        ep = MagicMock()
        ep.name = name
        ep.load.return_value = load_result
        return ep

    def test_component_entry_point_stores_component_class(self) -> None:
        """ComponentEntryPoint entry stores the component class in _component_classes."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        ep = self._make_entry_point(
            "my_plugin", ComponentEntryPoint(dummy_factory, DummyConfig, DummyComponent)
        )

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            classes = registry.get_component_classes()

        assert "my_plugin" in classes
        assert classes["my_plugin"] is DummyComponent

    def test_component_entry_point_extracts_config_class(self) -> None:
        """ComponentEntryPoint entry uses the second element as the config class."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        ep = self._make_entry_point(
            "my_plugin", ComponentEntryPoint(dummy_factory, DummyConfig, DummyComponent)
        )

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            config_class = registry.get_config_class("my_plugin")

        assert config_class is DummyConfig

    def test_plain_tuple_still_works(self) -> None:
        """Plain 3-tuple entry points still work for backward compatibility."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        ep = self._make_entry_point("my_plugin", (dummy_factory, DummyConfig, DummyComponent))

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            classes = registry.get_component_classes()
            config_class = registry.get_config_class("my_plugin")

        assert "my_plugin" in classes
        assert classes["my_plugin"] is DummyComponent
        assert config_class is DummyConfig

    def test_plain_factory_entry_not_in_component_classes(self) -> None:
        """Non-tuple entry point does not appear in component_classes."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        ep = self._make_entry_point("plain_plugin", dummy_factory)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            classes = registry.get_component_classes()

        assert "plain_plugin" not in classes
        # But it should still be registered as a component
        assert "plain_plugin" in registry.get_registered_components()


class TestRegisterWithComponentClass:
    """Tests for programmatic registration with component_class parameter."""

    def test_register_stores_component_class(self) -> None:
        """register() with component_class stores the class."""
        registry = BaseRegistry[DummyComponent, None]("test")
        registry.register("my_type", DummyConfig, dummy_factory, component_class=DummyComponent)

        classes = registry.get_component_classes()
        assert classes["my_type"] is DummyComponent

    def test_register_without_component_class(self) -> None:
        """register() without component_class does not add to component_classes."""
        registry = BaseRegistry[DummyComponent, None]("test")
        registry.register("my_type", DummyConfig, dummy_factory)

        classes = registry.get_component_classes()
        assert "my_type" not in classes


class TestFailedEntryPointErrorHandling:
    """Tests for error handling when entry points fail to load."""

    def _make_entry_point(self, name: str, side_effect: Exception) -> MagicMock:
        ep = MagicMock()
        ep.name = name
        ep.load.side_effect = side_effect
        return ep

    def test_import_error_stored_silently(self) -> None:
        """ImportError is stored and re-raised when the plugin is requested."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        original_error = ImportError("No module named 'swebench'")
        ep = self._make_entry_point("failing_plugin", original_error)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with pytest.raises(ImportError, match="No module named"):
                registry.get_config_class("failing_plugin")

    def test_non_import_error_stored_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-ImportError is logged as warning and stored."""
        registry = BaseRegistry[DummyComponent, None]("test", "test.group")

        ep = self._make_entry_point("broken_plugin", RuntimeError("Unexpected"))

        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            caplog.at_level(logging.WARNING, logger="prompt_siren.registry_base"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected"):
                registry.get_config_class("broken_plugin")

        assert any("broken_plugin" in record.message for record in caplog.records)

    def test_error_chaining_in_get_config_class_from_factory(self) -> None:
        """String annotation resolution failure chains the original error."""
        registry = BaseRegistry[DummyComponent, None]("test")

        def factory_with_string_annotation(config: "NonexistentType") -> DummyComponent:  # type: ignore[name-defined]  # noqa: F821
            return DummyComponent()

        with pytest.raises(ValueError, match="Cannot resolve string annotation"):
            registry._get_config_class_from_factory(factory_with_string_annotation, "test_factory")

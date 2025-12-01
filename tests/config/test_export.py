# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for configuration export functionality."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from prompt_siren.config.export import export_default_config


def test_export_default_config(tmp_path: Path) -> None:
    """Test exporting the default configuration directory."""
    output_dir = tmp_path / "exported_config"

    # Export the configuration
    export_default_config(output_dir)

    # Check the directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Check the main config file was created
    config_file = output_dir / "config.yaml"
    assert config_file.exists()

    # Check the file contains expected content
    content = config_file.read_text()
    assert "# Default Configuration" in content
    assert 'name: "default_experiment"' in content
    assert "agent:" in content
    assert "execution:" in content
    assert "task_ids:" in content

    # Verify it can be loaded back as OmegaConf
    cfg = OmegaConf.create(content)
    assert cfg.name is not None
    assert cfg.agent is not None

    # Check subdirectories were created
    assert (output_dir / "attack").exists()
    assert (output_dir / "attack").is_dir()
    assert (output_dir / "dataset").exists()
    assert (output_dir / "dataset").is_dir()


def test_export_default_config_fails_if_directory_missing(
    tmp_path: Path,
) -> None:
    """Test that export fails if parent directory doesn't exist."""
    output_dir = tmp_path / "nested" / "dirs" / "config"

    # Should raise FileNotFoundError since parent directory doesn't exist
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        export_default_config(output_dir)


def test_export_default_config_fails_if_directory_exists(
    tmp_path: Path,
) -> None:
    """Test that export fails if the output directory already exists."""
    output_dir = tmp_path / "config"

    # Create an existing directory
    output_dir.mkdir()

    # Should raise FileExistsError since directory already exists
    with pytest.raises(OSError, match="Failed to export configuration"):
        export_default_config(output_dir)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Utilities for exporting configuration files."""

import shutil
from importlib.resources import as_file, files
from pathlib import Path


def export_default_config(output_path: Path) -> None:
    """Export the default configuration directory to the specified path.

    Args:
        output_path: Path where to save the configuration directory.

    Raises:
        FileNotFoundError: If the parent directory of the output path does not exist.
        OSError: For all other failures during export, including if the output path already exists,
        if the default configuration directory is missing, or if it is not a directory.
    """

    # Get the path to the default configuration directory using importlib.resources
    default_config_resource = files("prompt_siren.config").joinpath("default")

    # Check if parent directory exists
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {output_path.parent}")

    # Copy the configuration directory (will fail if output_path already exists)
    # Use as_file to ensure we have a filesystem path (handles zip files, wheels, etc.)
    try:
        with as_file(default_config_resource) as default_config_path:
            if not default_config_path.exists():
                raise FileNotFoundError(
                    f"Default configuration directory not found at {default_config_path}"
                )

            if not default_config_path.is_dir():
                raise NotADirectoryError(f"Expected a directory at {default_config_path}")

            shutil.copytree(default_config_path, output_path)
        print(f"Default configuration exported to: {output_path}")
    except Exception as e:
        raise OSError(f"Failed to export configuration to {output_path}: {e}") from e

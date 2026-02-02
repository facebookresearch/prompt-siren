# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser-based dataset for web agent tasks.

This package currently exposes a single browser dataset that provides screenshot
observations and coordinate-based tools for interaction.
"""

from .config import BrowserDatasetConfig
from .dataset import browser_entry, BrowserDataset, create_browser_dataset

__all__ = [
    "BrowserDataset",
    "BrowserDatasetConfig",
    "browser_entry",
    "create_browser_dataset",
]

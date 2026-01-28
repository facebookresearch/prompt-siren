# Copyright (c) Meta Platforms, Inc. and affiliates.
"""PurpleLlama dataset for prompt injection benchmarks."""

from .dataset import (
    create_purplellama_dataset,
    load_purplellama_dataset,
    PurpleLlamaDataset,
    PurpleLlamaDatasetConfig,
)

__all__ = [
    "PurpleLlamaDataset",
    "PurpleLlamaDatasetConfig",
    "create_purplellama_dataset",
    "load_purplellama_dataset",
]

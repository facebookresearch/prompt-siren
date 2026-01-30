# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared Docker image tag helpers for datasets.

This module centralizes dataset image tag formatting so multiple datasets can
reuse a consistent scheme.
"""

from __future__ import annotations

from ..sandbox_managers.docker import extract_registry_from_tag


def normalize_tag_component(value: str) -> str:
    """Normalize a string for use as a Docker tag component.

    Replaces characters not allowed in Docker tags with double underscores.
    """
    return value.replace("/", "__").replace(":", "__").replace(" ", "__").lower()


def apply_registry_prefix(tag: str, registry: str | None) -> str:
    """Apply a registry prefix to an image tag if provided."""
    if registry:
        # Don't double-prefix if tag already has a registry
        if extract_registry_from_tag(tag) is not None:
            return tag
        return f"{registry}/{tag}"
    return tag


def make_dataset_tag(
    dataset_prefix: str,
    role: str,
    identifier: str,
    registry: str | None = None,
) -> str:
    """Construct a dataset-scoped tag using the common scheme."""
    tag = f"{dataset_prefix}-{role}:{normalize_tag_component(identifier)}"
    return apply_registry_prefix(tag, registry)

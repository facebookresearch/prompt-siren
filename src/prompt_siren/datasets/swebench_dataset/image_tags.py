# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image tag utilities for SWEBench tasks.

This module provides utilities for generating consistent Docker image tags
that are used both by the build_images script and by the task definitions.

The image tags follow a consistent naming convention:
- Benign tasks: siren-swebench-benign:{normalized_instance_id}
- Service containers: siren-swebench-service:{normalized_task_id}
- Benign x malicious pairs: siren-swebench-pair:{normalized_benign_id}__{normalized_malicious_id}
- Agent images: siren-swebench-agent:basic
- Base/env cache images: siren-swebench-base:{hash}, siren-swebench-env:{hash}
"""
from .constants import SWEBENCH_IMAGE_PREFIX
from ..image_tags import apply_registry_prefix, make_dataset_tag, normalize_tag_component


def normalize_tag(name: str) -> str:
    """Normalize a name for use as a Docker image tag.

    Replaces characters not allowed in Docker tags with underscores.

    Args:
        name: The name to normalize

    Returns:
        Normalized tag string
    """
    return normalize_tag_component(name)


def get_benign_image_tag(instance_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a benign task.

    Args:
        instance_id: The SWEBench instance ID (e.g., "django__django-11179")
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-benign:django__django-11179")
    """
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "benign", instance_id, registry)


def get_service_image_tag(task_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a malicious task's service container.

    Args:
        task_id: The malicious task ID (e.g., "env_direct_exfil_task")
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-service:env_direct_exfil_task")
    """
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "service", task_id, registry)


def get_malicious_image_tag(task_id: str, registry: str | None = None) -> str:
    """Deprecated: use get_service_image_tag instead."""
    return get_service_image_tag(task_id, registry)


def get_pair_image_tag(benign_id: str, malicious_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a benign x malicious pair.

    Args:
        benign_id: The benign task instance ID
        malicious_id: The malicious task ID
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-pair:django__django-11179__env_direct_exfil_task")
    """
    pair_id = f"{benign_id}__{malicious_id}"
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "pair", pair_id, registry)


def get_basic_agent_image_tag(registry: str | None = None) -> str:
    """Get the Docker image tag for the basic agent container.

    Args:
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag for the basic agent
    """
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "agent", "basic", registry)


def get_base_image_tag(cache_key_hash: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a base cache image."""
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "base", cache_key_hash, registry)


def get_env_image_tag(cache_key_hash: str, registry: str | None = None) -> str:
    """Get the Docker image tag for an environment cache image."""
    return make_dataset_tag(SWEBENCH_IMAGE_PREFIX, "env", cache_key_hash, registry)

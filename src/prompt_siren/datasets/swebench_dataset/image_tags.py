# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image tag utilities for SWEBench tasks.

This module provides utilities for generating consistent Docker image tags
that are used both by the build_images script and by the task definitions.

The image tags follow a consistent naming convention:
- Benign tasks: swebench-benign:{normalized_instance_id}
- Malicious service containers: swebench-malicious:{normalized_task_id}
- Benign x malicious pairs: swebench-pair:{normalized_benign_id}__{normalized_malicious_id}
"""

from .constants import SWEBENCH_IMAGE_PREFIX


def normalize_tag(name: str) -> str:
    """Normalize a name for use as a Docker image tag.

    Replaces characters not allowed in Docker tags with underscores.

    Args:
        name: The name to normalize

    Returns:
        Normalized tag string
    """
    return name.replace("/", "__").replace(":", "__").replace(" ", "__").lower()


def apply_registry_prefix(tag: str, registry: str | None) -> str:
    """Apply a registry prefix to an image tag if provided.

    Args:
        tag: The base image tag (e.g., "siren-swebench-benign:instance")
        registry: Optional registry prefix (e.g., "my-registry.com/myrepo")

    Returns:
        Tag with registry prefix if provided, otherwise the original tag
        (e.g., "my-registry.com/myrepo/siren-swebench-benign:instance")
    """
    if registry:
        return f"{registry}/{tag}"
    return tag


def get_benign_image_tag(instance_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a benign task.

    Args:
        instance_id: The SWEBench instance ID (e.g., "django__django-11179")
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-benign:django__django-11179")
    """
    tag = f"{SWEBENCH_IMAGE_PREFIX}-benign:{normalize_tag(instance_id)}"
    return apply_registry_prefix(tag, registry)


def get_malicious_image_tag(task_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a malicious task's service container.

    Args:
        task_id: The malicious task ID (e.g., "env_direct_exfil_task")
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-malicious:env_direct_exfil_task")
    """
    tag = f"{SWEBENCH_IMAGE_PREFIX}-malicious:{normalize_tag(task_id)}"
    return apply_registry_prefix(tag, registry)


def get_pair_image_tag(benign_id: str, malicious_id: str, registry: str | None = None) -> str:
    """Get the Docker image tag for a benign x malicious pair.

    Args:
        benign_id: The benign task instance ID
        malicious_id: The malicious task ID
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag (e.g., "siren-swebench-pair:django__django-11179__env_direct_exfil_task")
    """
    tag = f"{SWEBENCH_IMAGE_PREFIX}-pair:{normalize_tag(benign_id)}__{normalize_tag(malicious_id)}"
    return apply_registry_prefix(tag, registry)


def get_basic_agent_image_tag(registry: str | None = None) -> str:
    """Get the Docker image tag for the basic agent container.

    Args:
        registry: Optional registry prefix to prepend

    Returns:
        Docker image tag for the basic agent
    """
    tag = f"{SWEBENCH_IMAGE_PREFIX}-basic-agent:latest"
    return apply_registry_prefix(tag, registry)

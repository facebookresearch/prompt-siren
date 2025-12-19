# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image tag utilities for SWEBench tasks.

This module provides utilities for generating consistent Docker image tags
that are used both by the build_images script and by the task definitions.

The image tags follow a consistent naming convention:
- Benign tasks: swebench-benign:{normalized_instance_id}
- Malicious service containers: swebench-malicious:{normalized_task_id}
- Benign x malicious pairs: swebench-pair:{normalized_benign_id}__{normalized_malicious_id}
"""


def normalize_tag(name: str) -> str:
    """Normalize a name for use as a Docker image tag.

    Replaces characters not allowed in Docker tags with underscores.

    Args:
        name: The name to normalize

    Returns:
        Normalized tag string
    """
    return name.replace("/", "_").replace(":", "_").replace(" ", "_").lower()


def get_benign_image_tag(instance_id: str) -> str:
    """Get the Docker image tag for a benign task.

    Args:
        instance_id: The SWEBench instance ID (e.g., "django__django-11179")

    Returns:
        Docker image tag (e.g., "swebench-benign:django__django-11179")
    """
    return f"siren-swebench-benign:{normalize_tag(instance_id)}"


def get_malicious_image_tag(task_id: str) -> str:
    """Get the Docker image tag for a malicious task's service container.

    Args:
        task_id: The malicious task ID (e.g., "env_direct_exfil_task")

    Returns:
        Docker image tag (e.g., "siren-swebench-malicious:env_direct_exfil_task")
    """
    return f"siren-swebench-malicious:{normalize_tag(task_id)}"


def get_pair_image_tag(benign_id: str, malicious_id: str) -> str:
    """Get the Docker image tag for a benign x malicious pair.

    Args:
        benign_id: The benign task instance ID
        malicious_id: The malicious task ID

    Returns:
        Docker image tag (e.g., "siren-swebench-pair:django__django-11179__env_direct_exfil_task")
    """
    return f"siren-swebench-pair:{normalize_tag(benign_id)}__{normalize_tag(malicious_id)}"


def get_basic_agent_image_tag() -> str:
    """Get the Docker image tag for the basic agent container.

    Returns:
        Docker image tag for the basic agent
    """
    return "siren-swebench-basic-agent:latest"

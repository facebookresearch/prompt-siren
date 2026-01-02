# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Job naming utilities for generating and sanitizing job names."""

import re
from datetime import datetime


def sanitize_for_filename(name: str) -> str:
    """Sanitize a name for use in filenames and directory names.

    Replaces problematic characters:
    - ':' -> '_'  (common in model names like "plain:gpt-5")
    - '/' -> '_'  (common in provider prefixes like "azure/gpt-5")
    - ' ' -> '_'  (spaces)
    - Other non-alphanumeric characters (except - and _) -> '_'

    Args:
        name: The name to sanitize

    Returns:
        Sanitized name safe for use in file/directory names

    Examples:
        >>> sanitize_for_filename("plain:gpt-5")
        'plain_gpt-5'
        >>> sanitize_for_filename("azure/gpt-5-turbo")
        'azure_gpt-5-turbo'
        >>> sanitize_for_filename("my model name")
        'my_model_name'
    """
    # Replace common separators with underscore
    sanitized = name.replace(":", "_").replace("/", "_").replace(" ", "_")
    # Replace any remaining problematic characters
    sanitized = re.sub(r"[^\w\-]", "_", sanitized)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")


def generate_job_name(
    dataset_type: str,
    agent_name: str,
    attack_type: str | None,
    timestamp: datetime | None = None,
) -> str:
    """Generate a job name from its components.

    Format: <dataset>_<sanitized-agent-name>_<attack>_<YYYY-MM-DD_HH-MM-SS>

    Args:
        dataset_type: Type of dataset (e.g., "agentdojo-workspace")
        agent_name: Agent name from agent.get_agent_name() (e.g., "plain:gpt-5")
        attack_type: Attack type or None for benign runs
        timestamp: Timestamp for the job (defaults to now)

    Returns:
        Generated job name

    Examples:
        >>> generate_job_name("agentdojo-workspace", "plain:gpt-5", None)
        'agentdojo-workspace_plain_gpt-5_benign_2025-01-15_14-30-00'
        >>> generate_job_name("agentdojo-workspace", "plain:gpt-5", "template_string")
        'agentdojo-workspace_plain_gpt-5_template_string_2025-01-15_14-30-00'
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Sanitize components
    dataset_safe = sanitize_for_filename(dataset_type)
    agent_safe = sanitize_for_filename(agent_name)
    attack_safe = sanitize_for_filename(attack_type) if attack_type else "benign"

    # Format timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    return f"{dataset_safe}_{agent_safe}_{attack_safe}_{timestamp_str}"

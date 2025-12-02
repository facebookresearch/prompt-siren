# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Configuration for SWE-bench dataset."""

from pydantic import BaseModel


class SwebenchDatasetConfig(BaseModel):
    """Configuration for SWE-bench dataset integration.

    This configuration allows customization of the base Docker environment
    while using SWE-bench's repository setup logic.
    """

    # Dataset selection
    dataset_name: str = "SWE-bench/SWE-bench_Lite"
    """HuggingFace dataset name or path to local JSON/JSONL file."""

    max_instances: int | None = None
    """Maximum number of instances to load (None = all). Useful for testing."""

    instance_ids: list[str] | None = None
    """Specific instance IDs to load (None = all). Takes precedence over max_instances."""

    # Build options
    use_cache: bool = True
    """Whether to use Docker build cache."""

    build_timeout: int = 1800
    """Docker build timeout in seconds (default: 30 minutes)."""

    cache_dir: str = ".swebench_cache"
    """Directory to cache build contexts and generated scripts."""

    # Prompt customization
    prompt_template: str = "swe-agent-swebench"
    """Prompt template to use. Can be:
    - Built-in name: "mini-swe-agent" or "swe-agent-swebench"
    - Path to custom YAML file containing 'instance_template' key with Jinja2 template

    Defaults to "swe-agent-swebench".

    Available Jinja2 variables in templates:
    - problem_statement: Issue description from SWE-bench
    - repo: Repository name (e.g., "django/django")
    - instance_id: Instance identifier (e.g., "django__django-12345")
    - base_commit: Base commit hash
    - hints_text: Hints (empty string if include_hints=False)
    """

    include_hints: bool = False
    """Whether to include hints_text variable in prompt templates."""

    # Network configuration (passed to sandbox manager)
    enable_network: bool = False
    """Whether to enable network access in containers during task execution."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Jinja2-based prompt template loader for SWE-bench dataset."""

import os
from importlib import resources

import yaml

try:
    from jinja2 import Environment, StrictUndefined
    from swebench.harness.constants import SWEbenchInstance
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

# Built-in prompt templates available
BUILTIN_TEMPLATES = ["mini-swe-agent", "swe-agent-swebench"]


def load_prompt_template(name_or_path: str) -> dict[str, str]:
    """Load a prompt template from a built-in name or file path.

    Args:
        name_or_path: Either a built-in template name ("mini-swe-agent", "swe-agent-swebench")
            or a path to a custom YAML file containing an 'instance_template' and a 'system_prompt' key.

    Returns:
        Dictionary containing template keys (at minimum 'instance_template' and 'system_prompt')

    Raises:
        ValueError: If the template name is not recognized or file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        KeyError: If the YAML file doesn't contain 'instance_template'
    """
    # Check if it's a file path (must be a file, not a directory)
    if os.path.isfile(name_or_path):
        with open(name_or_path) as f:
            template_data = yaml.safe_load(f)
            if "instance_template" not in template_data:
                raise KeyError(f"Template file {name_or_path} must contain 'instance_template' key")
            return template_data

    # Check if it's a built-in template
    if name_or_path in BUILTIN_TEMPLATES:
        # Use importlib.resources to access package data
        package = resources.files("prompt_siren.datasets.swebench_dataset.prompts")
        template_file = package / f"{name_or_path}.yaml"
        template_content = template_file.read_text()
        return yaml.safe_load(template_content)

    # Not found
    raise ValueError(
        f"Unknown prompt template: {name_or_path}. "
        f"Available built-in templates: {', '.join(BUILTIN_TEMPLATES)}. "
        f"Or provide a path to a custom YAML file."
    )


def render_prompt(template_str: str, **variables) -> str:
    """Render a Jinja2 template string with provided variables.

    Args:
        template_str: Jinja2 template string to render
        **variables: Variables to substitute in the template

    Returns:
        Rendered template string

    Raises:
        jinja2.exceptions.UndefinedError: If a required variable is missing
    """
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template_str)
    return template.render(**variables)


def format_task_prompt_from_template(
    template_name_or_path: str,
    instance: SWEbenchInstance,
    include_hints: bool = False,
) -> str:
    """Format a task prompt from a template and SWE-bench instance.

    Args:
        template_name_or_path: Name of built-in template or path to custom YAML file
        instance: SWE-bench instance containing task data
        include_hints: Whether to include hints_text in the prompt

    Returns:
        Formatted prompt string rendered from the Jinja2 template

    Raises:
        ValueError: If template is not found
        KeyError: If instance is missing required fields
    """
    # Load the template
    template_data = load_prompt_template(template_name_or_path)
    instance_template = template_data["instance_template"]

    # Prepare variables for Jinja2 rendering
    variables = {
        "problem_statement": instance["problem_statement"],
        "repo": instance["repo"],
        "instance_id": instance["instance_id"],
        "base_commit": instance["base_commit"],
        "hints_text": instance["hints_text"] if include_hints else "",
    }

    # Render and return the prompt
    return render_prompt(instance_template, **variables)

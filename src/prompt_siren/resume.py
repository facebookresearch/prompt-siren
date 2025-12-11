# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Resume logic for re-running incomplete experiments."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import yaml
from omegaconf import DictConfig, OmegaConf

from .config.experiment_config import AgentConfig, AttackConfig, DatasetConfig
from .run_persistence import CONFIG_FILENAME, get_completed_task_ids
from .tasks import Task, TaskCouple

EnvStateT = TypeVar("EnvStateT")


class HasId(Protocol):
    """Protocol for objects with an id attribute."""

    @property
    def id(self) -> str: ...


_HasIdT = TypeVar("_HasIdT", bound=HasId)


@dataclass(frozen=True)
class SavedJobConfig:
    """Configuration loaded from a job directory's config.yaml."""

    dataset: DatasetConfig
    agent: AgentConfig
    attack: AttackConfig | None


def load_saved_job_config(job_path: Path) -> SavedJobConfig:
    """Load saved component configs from job directory.

    Args:
        job_path: Path to job directory containing config.yaml

    Returns:
        SavedJobConfig with dataset, agent, and attack configs

    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        ValueError: If config.yaml is malformed
    """
    config_file = job_path / CONFIG_FILENAME
    if not config_file.exists():
        raise FileNotFoundError(f"No {CONFIG_FILENAME} found in {job_path}")

    with open(config_file) as f:
        content = f.read()

    data = yaml.safe_load(content)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_file}: expected dict, got {type(data)}")

    if "dataset" not in data:
        raise ValueError(f"Missing 'dataset' key in {config_file}")
    if "agent" not in data:
        raise ValueError(f"Missing 'agent' key in {config_file}")

    return SavedJobConfig(
        dataset=DatasetConfig.model_validate(data["dataset"]),
        agent=AgentConfig.model_validate(data["agent"]),
        attack=AttackConfig.model_validate(data["attack"]) if data.get("attack") else None,
    )


def merge_configs(
    base_cfg: DictConfig,
    saved_config: SavedJobConfig,
) -> DictConfig:
    """Merge saved job config on top of base Hydra config.

    The saved dataset/agent/attack configs override the base config,
    preserving other settings (execution, telemetry, output, etc.)
    from the base config.

    Args:
        base_cfg: Base Hydra configuration (from --config-dir)
        saved_config: Saved component configs from job directory

    Returns:
        Merged DictConfig ready for experiment execution
    """
    # Create a copy to avoid mutating the original
    merged = OmegaConf.to_container(base_cfg, resolve=False)

    if not isinstance(merged, dict):
        raise ValueError("Base config must be a dictionary")

    # Override with saved configs (these define the experiment identity)
    merged["dataset"] = saved_config.dataset.model_dump()
    merged["agent"] = saved_config.agent.model_dump()
    merged["attack"] = saved_config.attack.model_dump() if saved_config.attack else None

    return OmegaConf.create(merged)


def _filter_incomplete(
    items: Sequence[_HasIdT],
    base_dir: Path,
    config_hash: str,
) -> list[_HasIdT]:
    """Filter out items that have already completed.

    Args:
        items: Sequence of items with an `id` attribute to filter
        base_dir: Base directory containing index.jsonl
        config_hash: Configuration hash to filter by

    Returns:
        List of items that have not yet completed
    """
    completed_ids = get_completed_task_ids(base_dir, config_hash)
    return [item for item in items if item.id not in completed_ids]


def filter_incomplete_tasks(
    tasks: Sequence[Task[EnvStateT]],
    base_dir: Path,
    config_hash: str,
) -> list[Task[EnvStateT]]:
    """Filter out tasks that have already completed.

    Args:
        tasks: Sequence of tasks to filter
        base_dir: Base directory containing index.jsonl
        config_hash: Configuration hash to filter by

    Returns:
        List of tasks that have not yet completed
    """
    return _filter_incomplete(tasks, base_dir, config_hash)


def filter_incomplete_couples(
    couples: Sequence[TaskCouple[EnvStateT]],
    base_dir: Path,
    config_hash: str,
) -> list[TaskCouple[EnvStateT]]:
    """Filter out task couples that have already completed.

    Args:
        couples: Sequence of task couples to filter
        base_dir: Base directory containing index.jsonl
        config_hash: Configuration hash to filter by

    Returns:
        List of task couples that have not yet completed
    """
    return _filter_incomplete(couples, base_dir, config_hash)

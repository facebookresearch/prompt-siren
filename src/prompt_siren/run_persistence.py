# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Persistence layer for saving task execution results and generated attacks."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from filelock import FileLock
from logfire import LogfireSpan
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from .config.experiment_config import AgentConfig, AttackConfig, DatasetConfig
from .tasks import EvaluationResult, Task, TaskCouple
from .types import (
    InjectionAttack,
    InjectionAttacksDict,
    InjectionAttacksDictTypeAdapter,
)

EnvStateT = TypeVar("EnvStateT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)

CONFIG_FILENAME = "config.yaml"
INDEX_FILENAME = "index.jsonl"


class ResultsData(BaseModel):
    """Results data for a task execution."""

    benign_score: float
    attack_score: float | None = None


class ExecutionData(BaseModel):
    """Complete execution data saved to JSON file."""

    execution_id: str
    task_id: str
    dataset_type: str
    dataset_config: dict[str, Any]
    agent: str
    config_hash: str
    timestamp: str
    trace_id: str | None
    span_id: str | None
    messages: list[ModelMessage]
    usage: RunUsage
    results: ResultsData
    attacks: InjectionAttacksDict[InjectionAttack] | None = None  # Optional generated attacks


class IndexEntry(BaseModel):
    """Entry in the global index.jsonl file."""

    execution_id: str
    task_id: str
    timestamp: str
    dataset: str
    dataset_config: dict[str, Any]
    agent_type: str
    agent_name: str
    attack_type: str | None
    attack_config: dict[str, Any] | None
    config_hash: str
    benign_score: float
    attack_score: float | None
    path: Path


def compute_config_hash(
    dataset_config: DatasetConfig,
    agent_config: AgentConfig,
    attack_config: AttackConfig | None,
) -> str:
    """Compute deterministic hash of combined configs including component types.

    Uses Pydantic's JSON serialization mode for consistency with special types
    (datetime, Path, UUID, etc). Includes component types to ensure different
    component types with same config produce different hashes.

    Args:
        dataset_config: Dataset configuration with type and config
        agent_config: Agent configuration with type and config
        attack_config: Attack configuration with type and config (None for benign runs)

    Returns:
        First 8 characters of SHA256 hash of combined configs

    Example:
        >>> from prompt_siren.config.experiment_config import (
        ...     AgentConfig,
        ...     DatasetConfig,
        ...     AttackConfig,
        ... )
        >>> dataset = DatasetConfig(type="agentdojo-workspace", config={"suite_name": "workspace"})
        >>> agent = AgentConfig(
        ...     type="plain",
        ...     config={"model": "claude-3-5-sonnet", "temperature": 0.0},
        ... )
        >>> attack = AttackConfig(type="gcg", config={"learning_rate": 0.1, "epochs": 100})
        >>> compute_config_hash(dataset, agent, attack)
        'a1b2c3d4'
    """
    # Combine all configs into a single dictionary for hashing
    # Exclude sandbox manager config from dataset config as it's an implementation detail
    # that shouldn't affect experiment identity (sandbox type shouldn't change results)
    dataset_dict = dataset_config.model_dump()
    # Remove sandbox fields from the nested config dict if they exist
    if "config" in dataset_dict and isinstance(dataset_dict["config"], dict):
        dataset_dict["config"] = {
            k: v
            for k, v in dataset_dict["config"].items()
            if k not in ("sandbox_manager_type", "sandbox_manager_config")
        }

    combined: dict[str, Any] = {
        "dataset": dataset_dict,
        "agent": agent_config.model_dump(),
        "attack": attack_config.model_dump() if attack_config is not None else None,
    }

    # Use json.dumps with sort_keys for deterministic key ordering
    config_str = json.dumps(combined, sort_keys=True)

    # Hash and return first 8 characters
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:8]


@dataclass(frozen=True)
class ExecutionPersistence:
    """Manages persistence for task executions in config-organized directories.

    Creates directory structure: {base_dir}/{dataset_type}/{agent_type}/{attack_type}/{config_hash}/
    Each directory contains:
    - config.yaml: Hydra-compatible configuration snapshot
    - {timestamp}_{task_id}.json: Individual task execution results
    """

    base_dir: Path
    dataset_config: DatasetConfig
    agent_config: AgentConfig
    attack_config: AttackConfig | None
    config_hash: str
    output_dir: Path

    @classmethod
    def create(
        cls,
        base_dir: Path,
        dataset_config: DatasetConfig,
        agent_config: AgentConfig,
        attack_config: AttackConfig | None,
    ) -> "ExecutionPersistence":
        """Factory method to create ExecutionPersistence with directory setup.

        Args:
            base_dir: Base directory for all outputs (e.g., "outputs/")
            dataset_config: Dataset configuration with type and config
            agent_config: Agent configuration with type and config
            attack_config: Attack configuration with type and config (None for benign runs)

        Returns:
            ExecutionPersistence instance with initialized directory structure
        """
        config_hash = compute_config_hash(dataset_config, agent_config, attack_config)

        # Construct output directory path
        attack_type_safe = attack_config.type if attack_config else "benign"
        output_dir = (
            base_dir / dataset_config.type / agent_config.type / attack_type_safe / config_hash
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config.yaml if it doesn't exist
        config_file = output_dir / CONFIG_FILENAME
        if not config_file.exists():
            _save_config_yaml(
                config_file=config_file,
                config_hash=config_hash,
                dataset_config=dataset_config,
                agent_config=agent_config,
                attack_config=attack_config,
            )

        return cls(
            base_dir=base_dir,
            dataset_config=dataset_config,
            agent_config=agent_config,
            attack_config=attack_config,
            config_hash=config_hash,
            output_dir=output_dir,
        )

    def save_single_task_execution(
        self,
        task: Task[EnvStateT],
        agent_name: str,
        result_ctx: RunContext[EnvStateT],
        evaluation: EvaluationResult,
        task_span: LogfireSpan,
    ) -> Path:
        """Save a single task execution to JSON file and update global index.

        Used for benign mode where only one task (benign or malicious run as benign) is executed.

        Args:
            task: Task that was executed
            agent_name: Name of the agent
            result_ctx: Run context with messages and usage
            evaluation: Task evaluation results
            task_span: Logfire span for trace context

        Returns:
            Path to the created execution file
        """
        # Generate unique execution ID
        execution_id = str(uuid.uuid4())[:8]

        # Create timestamp
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"{timestamp_str}_{task.id}.json"
        filepath = self.output_dir / filename

        # Get trace context from span
        span_context = task_span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context is not None else None
        span_id = format(span_context.span_id, "016x") if span_context is not None else None

        # Calculate score
        score = (
            sum(evaluation.results.values()) / len(evaluation.results)
            if evaluation.results
            else 0.0
        )

        # Build execution data
        results_data = ResultsData(
            benign_score=score,
            attack_score=None,  # No attack in benign mode
        )

        execution_data = ExecutionData(
            execution_id=execution_id,
            task_id=task.id,
            dataset_type=self.dataset_config.type,
            dataset_config=self.dataset_config.config,
            agent=agent_name,
            config_hash=self.config_hash,
            timestamp=timestamp.isoformat(),
            trace_id=trace_id,
            span_id=span_id,
            messages=result_ctx.messages,
            usage=result_ctx.usage,
            results=results_data,
            attacks=None,  # No attacks in benign mode
        )

        # Write execution file
        with open(filepath, "w") as f:
            f.write(execution_data.model_dump_json(indent=2))

        # Update global index
        self._append_to_index(
            execution_id=execution_id,
            task_id=task.id,
            agent_name=agent_name,
            results=results_data,
            filepath=filepath,
        )

        return filepath

    def save_couple_execution(
        self,
        couple: TaskCouple[EnvStateT],
        agent_name: str,
        result_ctx: RunContext[EnvStateT],
        benign_eval: EvaluationResult,
        malicious_eval: EvaluationResult,
        task_span: LogfireSpan,
        generated_attacks: dict[str, InjectionAttackT] | None,
    ) -> Path:
        """Save a task couple execution to JSON file and update global index.

        Used for attack mode where both benign and malicious tasks are evaluated.

        Args:
            couple: Task couple that was executed
            agent_name: Name of the agent
            result_ctx: Run context with messages and usage
            benign_eval: Benign task evaluation results
            malicious_eval: Malicious task evaluation results
            task_span: Logfire span for trace context
            generated_attacks: Generated attack vectors (None if not applicable)

        Returns:
            Path to the created execution file
        """
        # Generate unique execution ID
        execution_id = str(uuid.uuid4())[:8]

        # Create timestamp
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Sanitize couple ID for filename (replace : with _)
        couple_id_safe = couple.id.replace(":", "_")

        # Create filename
        filename = f"{timestamp_str}_{couple_id_safe}.json"
        filepath = self.output_dir / filename

        # Get trace context from span
        span_context = task_span.get_span_context()
        trace_id = format(span_context.trace_id, "032x") if span_context is not None else None
        span_id = format(span_context.span_id, "016x") if span_context is not None else None

        # Calculate scores
        benign_score = (
            sum(benign_eval.results.values()) / len(benign_eval.results)
            if benign_eval.results
            else 0.0
        )
        malicious_score = (
            sum(malicious_eval.results.values()) / len(malicious_eval.results)
            if malicious_eval.results
            else 0.0
        )

        # Build execution data
        results_data = ResultsData(
            benign_score=benign_score,
            attack_score=malicious_score,
        )

        # Parse generated attacks if present
        attacks_dict = (
            InjectionAttacksDictTypeAdapter.dump_python(generated_attacks)
            if generated_attacks
            else None
        )

        execution_data = ExecutionData(
            execution_id=execution_id,
            task_id=couple.id,
            dataset_type=self.dataset_config.type,
            dataset_config=self.dataset_config.config,
            agent=agent_name,
            config_hash=self.config_hash,
            timestamp=timestamp.isoformat(),
            trace_id=trace_id,
            span_id=span_id,
            messages=result_ctx.messages,
            usage=result_ctx.usage,
            results=results_data,
            attacks=attacks_dict,
        )

        # Write execution file
        with open(filepath, "w") as f:
            f.write(execution_data.model_dump_json(indent=2))

        # Update global index
        self._append_to_index(
            execution_id=execution_id,
            task_id=couple.id,
            agent_name=agent_name,
            results=results_data,
            filepath=filepath,
        )

        return filepath

    def _append_to_index(
        self,
        execution_id: str,
        task_id: str,
        agent_name: str,
        results: ResultsData,
        filepath: Path,
    ) -> None:
        """Atomically append to global index.jsonl with file locking for multi-process safety.

        Uses FileLock to ensure that concurrent writes from multiple processes don't corrupt
        the index file. The lock file (index.jsonl.lock) will remain after writes - this is
        expected behavior from FileLock and the file is used only for coordination.

        Args:
            execution_id: Unique execution identifier
            task_id: Task identifier
            agent_name: Agent name from agent.get_agent_name()
            results: Dictionary with benign_score and optional attack_score
            filepath: Path to the execution file
        """
        index_file = self.base_dir / INDEX_FILENAME
        lock_file = self.base_dir / f"{INDEX_FILENAME}.lock"

        # Build index entry using the IndexEntry model
        index_entry = IndexEntry(
            execution_id=execution_id,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            dataset=self.dataset_config.type,  # Using alias for serialization
            dataset_config=self.dataset_config.config,
            agent_type=self.agent_config.type,
            agent_name=agent_name,
            attack_type=self.attack_config.type if self.attack_config else None,
            attack_config=self.attack_config.config if self.attack_config else None,
            config_hash=self.config_hash,
            benign_score=results.benign_score,
            attack_score=results.attack_score,
            path=filepath,
        )

        # Use file lock to ensure atomic append across processes
        with FileLock(lock_file):
            with open(index_file, "a") as f:
                f.write(index_entry.model_dump_json() + "\n")


def _save_config_yaml(
    config_file: Path,
    config_hash: str,
    dataset_config: DatasetConfig,
    agent_config: AgentConfig,
    attack_config: AttackConfig | None,
) -> None:
    """Save config.yaml file with component configs.

    Creates a Hydra-compatible YAML file with dataset, agent, and attack configurations.
    Includes a header comment with the config hash and creation timestamp.

    Args:
        config_file: Path where to save the config.yaml
        config_hash: Hash of the combined configuration
        dataset_config: Dataset configuration with type and config
        agent_config: Agent configuration with type and config
        attack_config: Attack configuration with type and config (None for benign)
    """
    # Build config structure for YAML output
    config_data: dict[str, Any] = {
        "dataset": dataset_config.model_dump(),
        "agent": agent_config.model_dump(),
        "attack": attack_config.model_dump() if attack_config is not None else None,
    }

    # Create header with metadata
    header = f"# Config hash: {config_hash}\n"
    header += f"# Created: {datetime.now().isoformat()}\n\n"

    # Write YAML file
    with open(config_file, "w") as f:
        f.write(header)
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Main Hydra application for the Siren."""

import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from pydantic_ai.exceptions import UserError

from .agents.registry import get_agent_config_class
from .attacks.registry import get_attack_config_class
from .config.exceptions import ConfigValidationError
from .config.experiment_config import ExperimentConfig
from .config.registry_bridge import (
    create_agent_from_config,
    create_attack_from_config,
    create_dataset_from_config,
    create_sandbox_manager_from_config,
)
from .datasets.registry import get_dataset_config_class
from .registry_base import UnknownComponentError
from .run import run_single_tasks_without_attack, run_task_couples_with_attack
from .run_persistence import ExecutionPersistence
from .telemetry import setup_telemetry
from .telemetry.formatted_span import formatted_span
from .types import ExecutionMode


def validate_config(cfg: DictConfig, execution_mode: ExecutionMode) -> ExperimentConfig:
    """Validate experiment configuration using Pydantic and component registries.

    Performs validation without instantiating components:
    1. Validates against Pydantic ExperimentConfig schema
    2. Checks component types are registered
    3. Validates task selection logic

    Args:
        cfg: Hydra configuration
        execution_mode: Execution mode ('benign' or 'attack')

    Returns:
        Validated ExperimentConfig instance

    Raises:
        ValidationError: If configuration validation fails
        UnknownComponentError: If component type is not registered
        ConfigValidationError: If component configuration is invalid
        UserError: If external resources (e.g., API keys) are missing
    """
    # Convert OmegaConf to Pydantic for validation
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Check for missing dataset with helpful error message
    if config_dict is None or not isinstance(config_dict, dict) or "dataset" not in config_dict:
        raise ValueError(
            "Dataset configuration is required. "
            "Specify it in your config file or use +dataset=<dataset_name> override.\n"
            "Example: prompt-siren run benign +dataset=agentdojo-workspace"
        )

    experiment_config = ExperimentConfig.model_validate(config_dict)

    # Validate component types exist in registries (lightweight check)
    try:
        agent_config_class = get_agent_config_class(experiment_config.agent.type)
        agent_config_class.model_validate(experiment_config.agent.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("agent", experiment_config.agent.type, e) from e

    try:
        dataset_config_class = get_dataset_config_class(experiment_config.dataset.type)
        dataset_config_class.model_validate(experiment_config.dataset.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("dataset", experiment_config.dataset.type, e) from e

    # Validate task selection logic: attack mode requires attack config
    if execution_mode == "attack" and experiment_config.attack is None:
        raise ValueError(
            "Attack configuration is required for attack mode. "
            "Specify it in your config file or use +attack=<attack_name> override.\n"
            "Example: prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string"
        )

    if experiment_config.attack is None:
        return experiment_config

    try:
        attack_config_class = get_attack_config_class(experiment_config.attack.type)
        attack_config_class.model_validate(experiment_config.attack.config)
    except UnknownComponentError:
        # Let the specific exception propagate with its detailed message
        raise
    except ValidationError as e:
        # Re-raise with ConfigValidationError for better error handling
        raise ConfigValidationError("attack", experiment_config.attack.type, e) from e

    return experiment_config


async def run_benign_experiment(
    experiment_config: ExperimentConfig,
) -> dict[str, dict[str, float]]:
    """Run benign-only experiment.

    Args:
        experiment_config: Validated experiment configuration

    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    # Setup telemetry
    setup_telemetry(
        enable_console_export=experiment_config.telemetry.trace_console,
        otlp_endpoint=experiment_config.telemetry.otel_endpoint,
    )

    # Create components
    agent = create_agent_from_config(experiment_config.agent)

    # Create sandbox manager if configured
    sandbox_manager = None
    if experiment_config.sandbox_manager is not None:
        sandbox_manager = create_sandbox_manager_from_config(experiment_config.sandbox_manager)

    dataset = create_dataset_from_config(experiment_config.dataset, sandbox_manager)
    env_instance = dataset.environment
    toolsets = dataset.default_toolsets

    # Create trace directory
    trace_dir = experiment_config.output.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Get tasks from dataset: benign + malicious (both can be run as benign)
    all_tasks = dataset.benign_tasks + dataset.malicious_tasks

    # Filter by task_ids if specified
    if experiment_config.task_ids is not None:
        task_id_set = set(experiment_config.task_ids)
        selected_tasks = [t for t in all_tasks if t.id in task_id_set]

        # Warn about missing task IDs
        found_ids = {t.id for t in selected_tasks}
        missing_ids = task_id_set - found_ids
        if missing_ids:
            raise ValueError(f"Task IDs not found: {', '.join(sorted(missing_ids))}")
    else:
        selected_tasks = all_tasks

    # Create persistence instance
    persistence = ExecutionPersistence.create(
        base_dir=trace_dir,
        dataset_config=experiment_config.dataset,
        agent_config=experiment_config.agent,
        attack_config=None,  # No attack in benign mode
    )

    # Run benign experiment
    with formatted_span(
        "benign experiment with config {hash}",
        config=experiment_config.model_dump(),
        hash=persistence.config_hash,
    ):
        results = await run_single_tasks_without_attack(
            tasks=selected_tasks,
            agent=agent,
            env=env_instance,
            system_prompt=dataset.system_prompt,
            toolsets=toolsets,
            usage_limits=experiment_config.usage_limits,
            max_concurrency=experiment_config.execution.concurrency,
            persistence=persistence,
            instrument=True,
        )

        # Convert results to expected format
        return {result.task_id: result.results for result in results}


async def run_attack_experiment(
    experiment_config: ExperimentConfig,
) -> dict[str, dict[str, float]]:
    """Run attack experiment.

    Args:
        experiment_config: Validated experiment configuration (must include attack config)

    Returns:
        Dictionary mapping task IDs to evaluation results
    """
    if experiment_config.attack is None:
        raise ValueError("Attack configuration required for attack experiments")

    # Setup telemetry
    setup_telemetry(
        enable_console_export=experiment_config.telemetry.trace_console,
        otlp_endpoint=experiment_config.telemetry.otel_endpoint,
    )

    # Create components
    agent = create_agent_from_config(experiment_config.agent)

    # Create sandbox manager if configured
    sandbox_manager = None
    if experiment_config.sandbox_manager is not None:
        sandbox_manager = create_sandbox_manager_from_config(experiment_config.sandbox_manager)

    dataset = create_dataset_from_config(experiment_config.dataset, sandbox_manager)
    env_instance = dataset.environment
    attack_instance = create_attack_from_config(experiment_config.attack)
    toolsets = dataset.default_toolsets

    # Create trace directory
    trace_dir = experiment_config.output.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Get task couples from dataset
    all_couples = dataset.task_couples

    # Filter by task_ids if specified (couple IDs in format "benign:malicious")
    if experiment_config.task_ids is not None:
        couple_id_set = set(experiment_config.task_ids)
        selected_couples = [c for c in all_couples if c.id in couple_id_set]

        # Warn about missing couple IDs
        found_ids = {c.id for c in selected_couples}
        missing_ids = couple_id_set - found_ids
        if missing_ids:
            raise ValueError(f"Couple IDs not found: {', '.join(sorted(missing_ids))}")
    else:
        selected_couples = all_couples

    # Create persistence instance
    persistence = ExecutionPersistence.create(
        base_dir=trace_dir,
        dataset_config=experiment_config.dataset,
        agent_config=experiment_config.agent,
        attack_config=experiment_config.attack,
    )

    # Run attack experiment
    with formatted_span(
        "attack experiment with config {hash}",
        config=experiment_config.model_dump(),
        hash=persistence.config_hash,
    ):
        results = await run_task_couples_with_attack(
            couples=selected_couples,
            agent=agent,
            env=env_instance,
            system_prompt=dataset.system_prompt,
            toolsets=toolsets,
            attack=attack_instance,
            usage_limits=experiment_config.usage_limits,
            max_concurrency=experiment_config.execution.concurrency,
            persistence=persistence,
            instrument=True,
        )

        # Convert results to expected format (flatten benign + malicious)
        formatted_results = {}
        for benign_result, malicious_result in results:
            formatted_results[benign_result.task_id] = benign_result.results
            formatted_results[malicious_result.task_id] = malicious_result.results

        return formatted_results


def hydra_main_with_config_path(config_path: str, execution_mode: ExecutionMode) -> None:
    """Create and run Hydra main function with custom config path.

    This function creates a Hydra-decorated main function with the specified
    config path and runs it. This allows Hydra to handle --multirun and other
    Hydra-specific features.

    Args:
        config_path: Path to the Hydra configuration directory
        execution_mode: Execution mode ('benign' or 'attack')
    """

    @hydra.main(version_base=None, config_path=config_path, config_name="config")
    def hydra_app(cfg: DictConfig) -> None:
        """Hydra-decorated main function.

        This function is decorated with @hydra.main to enable Hydra's full feature set
        including multirun, launchers, and parameter sweeps.

        Args:
            cfg: Hydra configuration (automatically composed by Hydra)
        """
        # execution_mode is captured from the outer scope via closure
        # validate config first
        try:
            experiment_config = validate_config(cfg, execution_mode=execution_mode)
        except (
            ValidationError,
            UnknownComponentError,
            ConfigValidationError,
            ValueError,
            UserError,
        ) as e:
            print(f"Configuration validation failed: {e}")
            raise SystemExit(1) from e

        # Dispatch to appropriate experiment runner based on mode
        if execution_mode == "benign":
            asyncio.run(run_benign_experiment(experiment_config))
        else:  # attack
            asyncio.run(run_attack_experiment(experiment_config))

    # Run the Hydra app
    hydra_app()

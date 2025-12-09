# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Click-based CLI for Siren."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .config.export import export_default_config
from .hydra_app import hydra_main_with_config_path, validate_config
from .results import (
    aggregate_results,
    Format,
    format_results,
    GroupBy,
)
from .types import ExecutionMode


@click.group()
def main():
    """Siren - Prompt Injection Testing Framework."""


@main.group()
def config():
    """Configuration management commands."""


@config.command()
@click.argument("path", type=click.Path(path_type=Path), default="./config")
def export(path: Path):
    """Export default configuration to PATH.

    Examples:
        prompt-siren config export
        prompt-siren config export ./my_config
    """
    try:
        export_default_config(path)
    except Exception as e:
        click.echo(f"Error exporting configuration: {e}", err=True)
        raise SystemExit(1) from e


@config.command()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration directory path",
)
@click.option("--config-name", default="config", help="Configuration file name (without .yaml)")
@click.argument("overrides", nargs=-1)
def validate(config_dir: Path | None, config_name: str, overrides: tuple[str, ...]):
    """Validate configuration without running experiments.

    Examples:
        prompt-siren config validate +dataset=agentdojo-workspace
        prompt-siren config validate --config-dir=./my_config +dataset=agentdojo-workspace +attack=template_string
    """
    # Determine execution mode from overrides for validation
    # If +attack is specified, validate in attack mode; otherwise benign mode
    has_attack = any(override.startswith(("+attack=", "attack=")) for override in overrides)
    execution_mode = "attack" if has_attack else "benign"

    # Call validation directly
    _validate_config(config_dir, config_name, list(overrides), execution_mode=execution_mode)


@main.group()
def run():
    """Run experiments."""


@run.command()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration directory path",
)
@click.option("--config-name", default="config", help="Configuration file name (without .yaml)")
@click.option("--multirun", is_flag=True, help="Enable Hydra multirun mode for parameter sweeps")
@click.option(
    "--cfg",
    "--print-config",
    type=click.Choice(["job", "hydra", "all"]),
    help="Print composed config and exit (job=app config, hydra=hydra config, all=both)",
)
@click.option(
    "--resolve",
    is_flag=True,
    help="Resolve OmegaConf interpolations when printing config",
)
@click.option(
    "--info",
    type=click.Choice(["all", "config", "defaults", "defaults-tree", "plugins", "searchpath"]),
    help="Print Hydra information and exit",
)
@click.argument("overrides", nargs=-1)
def benign(
    config_dir: Path | None,
    config_name: str,
    multirun: bool,
    cfg: str | None,
    resolve: bool,
    info: str | None,
    overrides: tuple[str, ...],
):
    """Run benign-only evaluation (no attacks).

    Examples:
        prompt-siren run benign +dataset=agentdojo-workspace
        prompt-siren run benign +dataset=agentdojo-workspace agent.config.model=azure:gpt-5
        prompt-siren run benign --multirun +dataset=agentdojo-workspace agent.config.model=azure:gpt-5,azure:gpt-5-nano
        prompt-siren run benign --print-config job +dataset=agentdojo-workspace
        prompt-siren run benign --info plugins
    """
    _run_hydra(
        config_dir,
        config_name,
        list(overrides),
        execution_mode="benign",
        multirun=multirun,
        print_config=cfg,
        resolve=resolve,
        info=info,
    )


@run.command()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration directory path",
)
@click.option("--config-name", default="config", help="Configuration file name (without .yaml)")
@click.option("--multirun", is_flag=True, help="Enable Hydra multirun mode for parameter sweeps")
@click.option(
    "--cfg",
    "--print-config",
    type=click.Choice(["job", "hydra", "all"]),
    help="Print composed config and exit (job=app config, hydra=hydra config, all=both)",
)
@click.option(
    "--resolve",
    is_flag=True,
    help="Resolve OmegaConf interpolations when printing config",
)
@click.option(
    "--info",
    type=click.Choice(["all", "config", "defaults", "defaults-tree", "plugins", "searchpath"]),
    help="Print Hydra information and exit",
)
@click.argument("overrides", nargs=-1)
def attack(
    config_dir: Path | None,
    config_name: str,
    multirun: bool,
    cfg: str | None,
    resolve: bool,
    info: str | None,
    overrides: tuple[str, ...],
):
    """Run attack evaluation.

    Requires attack configuration (via +attack=<name> override or in config file).

    Examples:
        prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string
        prompt-siren run attack +dataset=agentdojo-workspace +attack=mini-goat agent.config.model=azure:gpt-5
        prompt-siren run attack --multirun +dataset=agentdojo-workspace +attack=template_string,mini-goat
        prompt-siren run attack --print-config job +dataset=agentdojo-workspace +attack=template_string
        prompt-siren run attack --info plugins
    """
    _run_hydra(
        config_dir,
        config_name,
        list(overrides),
        execution_mode="attack",
        multirun=multirun,
        print_config=cfg,
        resolve=resolve,
        info=info,
    )


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    default="./traces",
    help="Path to trace directory containing results",
)
@click.option(
    "--format",
    type=click.Choice(Format, case_sensitive=False),
    default="table",
    help="Output format for results",
)
@click.option(
    "--group-by",
    type=click.Choice(GroupBy, case_sensitive=False),
    default="all",
    help="Grouping level for results",
)
@click.option(
    "--k",
    type=int,
    multiple=True,
    default=[1],
    help="Value(s) for pass@k metric. Can specify multiple times (e.g., --k=1 --k=5 --k=10). Default is 1 (averaging). For k>1, computes pass@k where task passes if at least one of k runs succeeds.",
)
def results(
    output_dir: Path,
    format: Format,  # noqa: A002 -- format name is fine for cli-friendliness
    group_by: GroupBy,
    k: tuple[int, ...],
):
    """Show aggregated results from trace outputs.
    Results can be grouped by dataset, agent, attack, model, or show all configurations.
    Grouping is applied during aggregation, then results are formatted as requested.

    The --k parameter controls the pass@k metric:
    - k=1 (default): Average scores across runs for each task
    - k>1: Compute pass@k where a task passes if at least one of k runs succeeds (score=1.0)
    - Multiple k values: Specify --k multiple times to compute different pass@k metrics

    Examples:
        prompt-siren results
        prompt-siren results --output-dir=./traces
        prompt-siren results --format=json
        prompt-siren results --group-by=model
        prompt-siren results --k=5
        prompt-siren results --k=1 --k=5 --k=10
    """

    # Pass k values as list to aggregate_results (handles multiple k internally)
    aggregated = aggregate_results(output_dir, group_by=group_by, k=list(k))

    if aggregated is None or aggregated.empty:
        click.echo("No results found in output directory", err=True)
        return

    output = format_results(aggregated, format)
    click.echo(output)


def _resolve_config_dir(config_dir: Path | None) -> Path:
    """Resolve and validate configuration directory path."""
    # Determine config directory
    if config_dir is None:
        config_dir_path = Path.cwd() / "config"
    else:
        config_dir_path = config_dir.resolve()

    if not config_dir_path.exists():
        click.echo(f"Error: Configuration directory not found: {config_dir_path}", err=True)
        click.echo(
            "\nTo export the default configuration:\n  prompt-siren config export",
            err=True,
        )
        raise SystemExit(1)

    return config_dir_path


@contextmanager
def _compose_config(
    config_dir: Path | None, config_name: str, overrides: list[str]
) -> Iterator[DictConfig]:
    """Load and compose Hydra configuration."""
    config_dir_path = _resolve_config_dir(config_dir)

    # Initialize Hydra with the config directory and compose configuration
    with initialize_config_dir(version_base=None, config_dir=str(config_dir_path)):
        yield compose(config_name=config_name, overrides=overrides)


def _validate_config(
    config_dir: Path | None,
    config_name: str,
    overrides: list[str],
    execution_mode: ExecutionMode,
) -> None:
    """Validate configuration without running experiments.

    Args:
        config_dir: Configuration directory path (if None, uses default ./config)
        config_name: Configuration file name (without .yaml)
        overrides: List of Hydra overrides
        execution_mode: Execution mode ('benign' or 'attack')
    """
    with _compose_config(config_dir, config_name, overrides) as cfg:
        validate_config(cfg, execution_mode=execution_mode)
        click.echo("Configuration validation passed")


def _run_hydra(
    config_dir: Path | None,
    config_name: str,
    overrides: list[str],
    execution_mode: ExecutionMode,
    multirun: bool = False,
    print_config: str | None = None,
    resolve: bool = False,
    info: str | None = None,
) -> None:
    """Run Hydra with the given configuration.

    This function uses Hydra's decorator-based approach which fully supports
    multirun and other Hydra features like launchers and sweepers.

    Args:
        config_dir: Configuration directory path (if None, uses default ./config)
        config_name: Configuration file name (without .yaml)
        overrides: List of Hydra overrides
        execution_mode: Execution mode ('benign' or 'attack')
        multirun: Enable Hydra multirun mode for parameter sweeps
        print_config: Print configuration and exit (job, hydra, or all)
        resolve: Resolve OmegaConf interpolations when printing config
        info: Print Hydra information and exit
    """
    config_dir_path = _resolve_config_dir(config_dir)

    # Save original sys.argv
    original_argv = sys.argv.copy()

    try:
        # Construct new sys.argv for Hydra
        # Format: [script_name, config_name_override (if not default), ...hydra_flags, ...overrides]
        new_argv = [sys.argv[0]]

        # Add config name override if not default
        if config_name != "config":
            new_argv.append(f"--config-name={config_name}")

        # Add Hydra flags
        if multirun:
            new_argv.append("--multirun")
        if print_config:
            new_argv.append(f"--cfg={print_config}")
        if resolve:
            new_argv.append("--resolve")
        if info:
            new_argv.append(f"--info={info}")

        # Add user overrides
        new_argv.extend(overrides)

        # Replace sys.argv so Hydra can parse it
        sys.argv = new_argv

        # Run Hydra with the config path
        # The config_path must be absolute for the decorator
        hydra_main_with_config_path(str(config_dir_path), execution_mode=execution_mode)

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()

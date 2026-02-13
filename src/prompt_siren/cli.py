# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Click-based CLI for Siren."""

import asyncio
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .config.experiment_config import SlurmConfig
from .config.export import export_default_config
from .execution import _run_resume_job, SlurmBackend
from .hydra_app import (
    hydra_main_with_config_path,
    run_attack_experiment,
    run_benign_experiment,
    validate_config,
)
from .job import Job, JobConfigMismatchError
from .job.persistence import load_config_yaml
from .job.sweep import find_sweep_jobs_by_pattern, get_job_status, SweepRegistry
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


# ============================================================================
# Jobs commands (main interface)
# ============================================================================


@main.group()
def jobs():
    """Manage jobs."""


@jobs.group()
def start():
    """Start a new job."""


# Common options for start commands
_start_common_options = [
    click.option(
        "--config-dir",
        type=click.Path(exists=True, path_type=Path),
        help="Configuration directory path",
    ),
    click.option("--config-name", default="config", help="Configuration file name (without .yaml)"),
    click.option("--job-name", type=str, help="Custom job name (default: auto-generated)"),
    click.option(
        "--jobs-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Directory to store job results (default: from config)",
    ),
    click.option(
        "--multirun", is_flag=True, help="Enable Hydra multirun mode for parameter sweeps"
    ),
    click.option(
        "-k",
        "--runs",
        type=int,
        default=None,
        help="Number of independent runs (creates k separate jobs via SLURM or multirun)",
    ),
    click.option(
        "--slurm",
        is_flag=True,
        help="Submit job(s) to SLURM using submitit launcher",
    ),
    click.option(
        "--slurm-partition",
        type=str,
        default=None,
        help="SLURM partition (overrides config)",
    ),
    click.option(
        "--slurm-time",
        type=int,
        default=None,
        help="SLURM time limit in minutes (overrides config)",
    ),
    click.option(
        "--cfg",
        "--print-config",
        type=click.Choice(["job", "hydra", "all"]),
        help="Print composed config and exit (job=app config, hydra=hydra config, all=both)",
    ),
    click.option(
        "--resolve",
        is_flag=True,
        help="Resolve OmegaConf interpolations when printing config",
    ),
    click.option(
        "--info",
        type=click.Choice(["all", "config", "defaults", "defaults-tree", "plugins", "searchpath"]),
        help="Print Hydra information and exit",
    ),
    click.argument("overrides", nargs=-1),
]


_F = TypeVar("_F", bound=Callable[..., object])


def _add_options(options: list[Callable[[_F], _F]]) -> Callable[[_F], _F]:
    """Decorator to add multiple options to a command."""

    def decorator(func: _F) -> _F:
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@start.command(name="benign")
@_add_options(_start_common_options)
def start_benign(
    config_dir: Path | None,
    config_name: str,
    job_name: str | None,
    jobs_dir: Path | None,
    multirun: bool,
    runs: int | None,
    slurm: bool,
    slurm_partition: str | None,
    slurm_time: int | None,
    cfg: str | None,
    resolve: bool,
    info: str | None,
    overrides: tuple[str, ...],
):
    """Start a benign-only evaluation job (no attacks).

    Examples:
        prompt-siren jobs start benign +dataset=agentdojo-workspace
        prompt-siren jobs start benign --job-name=my-experiment +dataset=agentdojo-workspace
        prompt-siren jobs start benign -k 10 --slurm +dataset=agentdojo-workspace
        prompt-siren jobs start benign --multirun +dataset=agentdojo-workspace agent.config.model=azure:gpt-5,azure:gpt-5-nano
    """
    _run_job(
        config_dir=config_dir,
        config_name=config_name,
        overrides=list(overrides),
        execution_mode="benign",
        job_name=job_name,
        jobs_dir=jobs_dir,
        multirun=multirun,
        runs=runs,
        slurm=slurm,
        slurm_partition=slurm_partition,
        slurm_time=slurm_time,
        print_config=cfg,
        resolve=resolve,
        info=info,
    )


@start.command(name="attack")
@_add_options(_start_common_options)
def start_attack(
    config_dir: Path | None,
    config_name: str,
    job_name: str | None,
    jobs_dir: Path | None,
    multirun: bool,
    runs: int | None,
    slurm: bool,
    slurm_partition: str | None,
    slurm_time: int | None,
    cfg: str | None,
    resolve: bool,
    info: str | None,
    overrides: tuple[str, ...],
):
    """Start an attack evaluation job.

    Requires attack configuration (via +attack=<name> override or in config file).

    Examples:
        prompt-siren jobs start attack +dataset=agentdojo-workspace +attack=mini-goat
        prompt-siren jobs start attack -k 10 --slurm +dataset=agentdojo-workspace +attack=mini-goat
        prompt-siren jobs start attack --job-name=my-attack-test +dataset=agentdojo-workspace +attack=mini-goat
    """
    _run_job(
        config_dir=config_dir,
        config_name=config_name,
        overrides=list(overrides),
        execution_mode="attack",
        job_name=job_name,
        jobs_dir=jobs_dir,
        multirun=multirun,
        runs=runs,
        slurm=slurm,
        slurm_partition=slurm_partition,
        slurm_time=slurm_time,
        print_config=cfg,
        resolve=resolve,
        info=info,
    )


@jobs.command()
@click.option(
    "-p",
    "--job-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a single job directory containing config.yaml",
)
@click.option(
    "-s",
    "--sweep-id",
    type=str,
    default=None,
    help="Sweep ID (timestamp like 2026-01-31_17-01-04) to resume all jobs from",
)
@click.option(
    "--pattern",
    type=str,
    default=None,
    help="Glob pattern to match job directory names (alternative to --sweep-id)",
)
@click.option(
    "-d",
    "--jobs-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./jobs"),
    help="Directory containing jobs (used with --sweep-id or --pattern)",
)
@click.option(
    "-e",
    "--retry-on-error",
    multiple=True,
    default=("CancelledError",),
    show_default=True,
    help="Retry tasks that failed with this error type (can be used multiple times). "
    "Use -e '' to disable retries.",
)
@click.option(
    "--only-failed",
    is_flag=True,
    default=False,
    help="Only resume jobs that have failed tasks (sweep mode only)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show jobs that would be resumed without actually running them (sweep mode only)",
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Submit job(s) to SLURM instead of running locally",
)
@click.option(
    "--slurm-partition",
    type=str,
    default=None,
    help="SLURM partition (overrides config)",
)
@click.option(
    "--slurm-time",
    type=int,
    default=None,
    help="SLURM time limit in minutes (overrides config)",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Don't wait for SLURM job(s) to complete (only with --slurm)",
)
@click.argument("overrides", nargs=-1)
def resume(
    job_path: Path | None,
    sweep_id: str | None,
    pattern: str | None,
    jobs_dir: Path,
    retry_on_error: tuple[str, ...],
    only_failed: bool,
    dry_run: bool,
    slurm: bool,
    slurm_partition: str | None,
    slurm_time: int | None,
    no_wait: bool,
    overrides: tuple[str, ...],
):
    """Resume existing job(s) from a job directory or sweep.

    Use -p for a single job, or -s/--pattern for all jobs in a sweep.

    Accepts Hydra-style overrides for execution-related fields only
    (execution.*, telemetry.*, output.*, usage_limits.*).

    Examples:
        # Resume a single job
        prompt-siren jobs resume -p ./jobs/my-job
        prompt-siren jobs resume -p ./jobs/my-job -e TimeoutError
        prompt-siren jobs resume -p ./jobs/my-job --slurm

        # Resume all jobs from a sweep
        prompt-siren jobs resume -s 2026-01-31_17-01-04
        prompt-siren jobs resume -s 2026-01-31_17-01-04 --slurm
        prompt-siren jobs resume -s 2026-01-31_17-01-04 --only-failed
        prompt-siren jobs resume -s 2026-01-31_17-01-04 --dry-run

        # Resume with pattern matching
        prompt-siren jobs resume --pattern '*_gpt-4o_*'
    """
    # Validate mutual exclusivity
    options_set = sum(1 for opt in [job_path, sweep_id, pattern] if opt is not None)
    if options_set == 0:
        click.echo(
            "Error: Must specify one of -p/--job-path, -s/--sweep-id, or --pattern", err=True
        )
        raise SystemExit(1)
    if options_set > 1:
        click.echo(
            "Error: Cannot specify multiple of -p/--job-path, -s/--sweep-id, or --pattern", err=True
        )
        raise SystemExit(1)

    # Handle retry_on_error: empty string means "no retries"
    retry_errors: list[str] | None = None
    if retry_on_error:
        non_empty = [e for e in retry_on_error if e]
        if non_empty:
            retry_errors = non_empty

    # Single job mode
    if job_path is not None:
        if only_failed or dry_run:
            click.echo(
                "Warning: --only-failed and --dry-run are ignored for single job resume", err=True
            )

        if slurm:
            _resume_on_slurm(
                job_path=job_path,
                overrides=list(overrides) if overrides else [],
                retry_errors=retry_errors,
                slurm_partition=slurm_partition,
                slurm_time=slurm_time,
                wait_for_completion=not no_wait,
            )
        else:
            _resume_locally(
                job_path=job_path,
                overrides=list(overrides) if overrides else [],
                retry_errors=retry_errors,
            )
        return

    # Sweep mode: find job directories
    if sweep_id is not None:
        registry = SweepRegistry(jobs_dir)
        sweep = registry.get_sweep(sweep_id)

        if sweep is not None:
            job_dirs = registry.get_job_dirs(sweep_id)
            click.echo(f"Found sweep '{sweep_id}' with {len(job_dirs)} jobs (from registry)")
        else:
            job_dirs = find_sweep_jobs_by_pattern(jobs_dir, sweep_id)
            click.echo(f"Found {len(job_dirs)} jobs matching timestamp '{sweep_id}'")
    else:
        assert pattern is not None
        job_dirs = find_sweep_jobs_by_pattern(jobs_dir, pattern)
        click.echo(f"Found {len(job_dirs)} jobs matching pattern '{pattern}'")

    if not job_dirs:
        click.echo("No matching jobs found.", err=True)
        raise SystemExit(1)

    # Filter to only failed jobs if requested
    if only_failed:
        failed_dirs = [
            job_dir
            for job_dir in job_dirs
            if (status := get_job_status(job_dir))["has_failures"] or status["failed_tasks"] > 0
        ]
        click.echo(f"Filtered to {len(failed_dirs)} jobs with failures")
        job_dirs = failed_dirs

        if not job_dirs:
            click.echo("No jobs with failures found.")
            return

    # Show what we're going to do
    if dry_run:
        execution_mode = "SLURM" if slurm else "local"
        click.echo(f"\nDry run ({execution_mode}) - would resume the following jobs:")
        for job_dir in job_dirs:
            status = get_job_status(job_dir)
            status_str = f"tasks: {status['total_tasks']}"
            if status["failed_tasks"] > 0:
                status_str += f", failed: {status['failed_tasks']}"
            if status["failure_types"]:
                status_str += f" ({', '.join(sorted(status['failure_types']))})"
            click.echo(f"  {job_dir.name} ({status_str})")
        return

    # Execute: either locally or on SLURM
    if slurm:
        _resume_sweep_on_slurm(
            job_dirs=job_dirs,
            overrides=list(overrides) if overrides else [],
            retry_errors=retry_errors,
            slurm_partition=slurm_partition,
            slurm_time=slurm_time,
            wait_for_completion=not no_wait,
        )
    else:
        _resume_sweep_locally(
            job_dirs=job_dirs,
            overrides=list(overrides) if overrides else [],
            retry_errors=retry_errors,
        )


def _resume_locally(
    job_path: Path,
    overrides: list[str],
    retry_errors: list[str] | None,
) -> None:
    """Resume a single job locally."""
    try:
        job = Job.resume(
            job_dir=job_path,
            overrides=overrides if overrides else None,
            retry_on_errors=retry_errors,
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e
    except JobConfigMismatchError as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1) from e

    experiment_config = job.to_experiment_config()
    execution_mode = job.job_config.execution_mode

    click.echo(f"Resuming job: {job.job_config.job_name}")
    click.echo(f"Mode: {execution_mode}")

    if execution_mode == "benign":
        asyncio.run(run_benign_experiment(experiment_config, job=job))
    else:
        asyncio.run(run_attack_experiment(experiment_config, job=job))


def _resume_on_slurm(
    job_path: Path,
    overrides: list[str],
    retry_errors: list[str] | None,
    slurm_partition: str | None,
    slurm_time: int | None,
    wait_for_completion: bool,
) -> None:
    """Resume a single job on SLURM."""
    config_path = job_path / "config.yaml"
    try:
        job_config = load_config_yaml(config_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e

    slurm_config = job_config.slurm if hasattr(job_config, "slurm") else SlurmConfig()
    if slurm_partition:
        slurm_config = slurm_config.model_copy(update={"partition": slurm_partition})
    if slurm_time:
        slurm_config = slurm_config.model_copy(update={"time_minutes": slurm_time})

    log_dir = job_path / ".slurm_logs"

    click.echo(f"Submitting job to SLURM: {job_config.job_name}")
    click.echo(f"  Partition: {slurm_config.partition}")
    click.echo(f"  Time limit: {slurm_config.time_minutes} minutes")
    click.echo(f"  Log directory: {log_dir}")

    backend = SlurmBackend(slurm_config, log_dir, wait_for_completion)
    job_info = backend.submit(
        _run_resume_job,
        str(job_path),
        overrides,
        retry_errors,
        job_config.execution_mode,
    )

    click.echo(f"  Job ID: {job_info.job_id}")

    if wait_for_completion:
        click.echo("Waiting for job to complete...")
        results = backend.wait()
        if results and isinstance(results[0], Exception):
            click.echo(f"Job failed: {results[0]}", err=True)
            raise SystemExit(1)
        click.echo("Job completed successfully")
    else:
        click.echo("Job submitted. Use 'squeue' to monitor progress.")


def _resume_sweep_locally(
    job_dirs: list[Path],
    overrides: list[str],
    retry_errors: list[str] | None,
) -> None:
    """Resume sweep jobs locally (sequential)."""
    click.echo(f"\nResuming {len(job_dirs)} jobs locally...")

    success_count = 0
    error_count = 0

    for i, job_dir in enumerate(job_dirs, 1):
        click.echo(f"\n[{i}/{len(job_dirs)}] Resuming: {job_dir.name}")

        try:
            job = Job.resume(
                job_dir=job_dir,
                overrides=overrides if overrides else None,
                retry_on_errors=retry_errors,
            )

            experiment_config = job.to_experiment_config()
            execution_mode = job.job_config.execution_mode

            if execution_mode == "benign":
                asyncio.run(run_benign_experiment(experiment_config, job=job))
            else:
                asyncio.run(run_attack_experiment(experiment_config, job=job))

            success_count += 1

        except FileNotFoundError as e:
            click.echo(f"  Error: {e}", err=True)
            error_count += 1
        except JobConfigMismatchError as e:
            click.echo(f"  Configuration error: {e}", err=True)
            error_count += 1
        except Exception as e:
            click.echo(f"  Unexpected error: {e}", err=True)
            error_count += 1

    click.echo(f"\n{'=' * 50}")
    click.echo(f"Resume complete: {success_count} succeeded, {error_count} failed")


def _resume_sweep_on_slurm(
    job_dirs: list[Path],
    overrides: list[str],
    retry_errors: list[str] | None,
    slurm_partition: str | None,
    slurm_time: int | None,
    wait_for_completion: bool,
) -> None:
    """Resume sweep jobs on SLURM (parallel submission)."""
    click.echo(f"\nSubmitting {len(job_dirs)} jobs to SLURM...")

    first_config_path = job_dirs[0] / "config.yaml"
    try:
        first_job_config = load_config_yaml(first_config_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e

    slurm_config = first_job_config.slurm if hasattr(first_job_config, "slurm") else SlurmConfig()
    if slurm_partition:
        slurm_config = slurm_config.model_copy(update={"partition": slurm_partition})
    if slurm_time:
        slurm_config = slurm_config.model_copy(update={"time_minutes": slurm_time})

    log_dir = job_dirs[0].parent / ".slurm_logs"

    click.echo(f"  Partition: {slurm_config.partition}")
    click.echo(f"  Time limit: {slurm_config.time_minutes} minutes")
    click.echo(f"  Log directory: {log_dir}")

    job_args_list = []
    for job_dir in job_dirs:
        config_path = job_dir / "config.yaml"
        try:
            job_config = load_config_yaml(config_path)
        except FileNotFoundError:
            click.echo(f"  Warning: Skipping {job_dir.name} (config not found)")
            continue

        job_args_list.append((str(job_dir), overrides, retry_errors, job_config.execution_mode))

    if not job_args_list:
        click.echo("No valid jobs to submit.", err=True)
        raise SystemExit(1)

    backend = SlurmBackend(slurm_config, log_dir, wait_for_completion)
    job_infos = backend.submit_batch(_run_resume_job, job_args_list)

    click.echo(f"\nSubmitted {len(job_infos)} SLURM jobs:")
    for job_info in job_infos:
        click.echo(f"  {job_info.job_id}: {job_info.job_dir}")

    if wait_for_completion:
        click.echo("\nWaiting for all jobs to complete...")
        results = backend.wait()

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = sum(1 for r in results if isinstance(r, Exception))
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                click.echo(f"  Job {job_infos[i].job_id} failed: {result}", err=True)

        click.echo(f"\n{'=' * 50}")
        click.echo(f"Resume complete: {success_count} succeeded, {error_count} failed")
    else:
        click.echo("\nJobs submitted. Use 'squeue' to monitor progress.")


# ============================================================================
# Run commands (aliases for jobs start)
# ============================================================================


@main.group()
def run():
    """Run experiments. Alias for 'jobs start'."""


# Register the same command objects under the 'run' group as aliases
run.add_command(start_benign, name="benign")
run.add_command(start_attack, name="attack")


@main.command()
@click.option(
    "--jobs-dir",
    type=click.Path(exists=True, path_type=Path),
    default="./jobs",
    help="Path to jobs directory containing results",
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
    jobs_dir: Path,
    format: Format,  # noqa: A002 -- format name is fine for cli-friendliness
    group_by: GroupBy,
    k: tuple[int, ...],
):
    """Show aggregated results from job outputs.

    Results can be grouped by dataset, agent, attack, model, or show all configurations.
    Grouping is applied during aggregation, then results are formatted as requested.

    The --k parameter controls the pass@k metric:
    - k=1 (default): Average scores across runs for each task
    - k>1: Compute pass@k where a task passes if at least one of k runs succeeds (score=1.0)
    - Multiple k values: Specify --k multiple times to compute different pass@k metrics

    Examples:
        prompt-siren results
        prompt-siren results --jobs-dir=./jobs
        prompt-siren results --format=json
        prompt-siren results --group-by=model
        prompt-siren results --k=5
        prompt-siren results --k=1 --k=5 --k=10
    """
    # Pass k values as list to aggregate_results (handles multiple k internally)
    aggregated = aggregate_results(jobs_dir, group_by=group_by, k=list(k))

    if aggregated is None or aggregated.empty:
        click.echo("No results found in jobs directory", err=True)
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


def _run_job(
    config_dir: Path | None,
    config_name: str,
    overrides: list[str],
    execution_mode: ExecutionMode,
    job_name: str | None = None,
    jobs_dir: Path | None = None,
    multirun: bool = False,
    slurm: bool = False,
    slurm_partition: str | None = None,
    slurm_time: int | None = None,
    print_config: str | None = None,
    resolve: bool = False,
    info: str | None = None,
    runs: int | None = None,
) -> None:
    """Run a job with the given configuration.

    This function passes job settings to Hydra via overrides. The actual Job
    creation happens in hydra_app.py after Hydra composes the configuration.
    This design is intentional because:
    - Hydra needs to compose config first (handling defaults, interpolations)
    - In multirun mode, Hydra creates multiple configs, each needing its own Job
    - Job creation requires the fully resolved ExperimentConfig

    Args:
        config_dir: Configuration directory path (if None, uses default ./config)
        config_name: Configuration file name (without .yaml)
        overrides: List of Hydra overrides
        execution_mode: Execution mode ('benign' or 'attack')
        job_name: Custom job name (auto-generated if None)
        jobs_dir: Directory to store job results (if None, uses value from config)
        multirun: Enable Hydra multirun mode for parameter sweeps
        slurm: Submit job(s) to SLURM using submitit launcher
        slurm_partition: SLURM partition (overrides config)
        slurm_time: SLURM time limit in minutes (overrides config)
        print_config: Print configuration and exit (job, hydra, or all)
        resolve: Resolve OmegaConf interpolations when printing config
        info: Print Hydra information and exit
        runs: Number of independent runs (creates k jobs via run_id sweep)
    """
    # Add job-related overrides (Job is created in hydra_app.py after config composition)
    job_overrides = list(overrides)
    if job_name is not None:
        job_overrides.append(f"output.job_name={job_name}")
    if jobs_dir is not None:
        job_overrides.append(f"output.jobs_dir={jobs_dir}")

    # Add run_id sweep if -k/--runs is specified
    if runs is not None:
        if runs < 1:
            raise click.BadParameter("runs must be at least 1", param_hint="-k/--runs")
        # Add run_id sweep override for k independent runs
        job_overrides.append(f"run_id=range({runs})")
        # Enable multirun mode (required for sweeps)
        multirun = True

    # Add SLURM launcher overrides if --slurm flag is set
    if slurm:
        # Enable multirun mode (required for submitit launcher)
        multirun = True
        # Use submitit_slurm launcher
        job_overrides.append("hydra/launcher=submitit_slurm")
        # Apply CLI overrides for SLURM parameters
        if slurm_partition is not None:
            job_overrides.append(f"hydra.launcher.partition={slurm_partition}")
        if slurm_time is not None:
            job_overrides.append(f"hydra.launcher.timeout_min={slurm_time}")

    _run_hydra(
        config_dir=config_dir,
        config_name=config_name,
        overrides=job_overrides,
        execution_mode=execution_mode,
        multirun=multirun,
        print_config=print_config,
        resolve=resolve,
        info=info,
    )


if __name__ == "__main__":
    main()

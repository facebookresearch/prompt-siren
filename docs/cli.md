# CLI Reference

Siren uses a job-based system for running and managing experiments. The CLI design is inspired by [Harbor](https://harborframework.com/docs).

This guide covers the CLI commands for starting, resuming, and analyzing experiment runs.

## Quick Start

```bash
# Run a benign evaluation
prompt-siren run benign +dataset=agentdojo-workspace

# Run an attack evaluation
prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string

# View results
prompt-siren results
```

## Starting Jobs

### `jobs start` / `run`

Start a new experiment job. The `run` command is an alias for `jobs start`.

```bash
# Benign evaluation (no attacks)
prompt-siren jobs start benign +dataset=agentdojo-workspace
prompt-siren run benign +dataset=agentdojo-workspace  # equivalent

# Attack evaluation
prompt-siren jobs start attack +dataset=agentdojo-workspace +attack=template_string
prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string  # equivalent
```

### Options

```bash
--job-name NAME      Custom job name (default: auto-generated)
--jobs-dir PATH      Directory to store results (default: ./jobs)
--config-dir PATH    Custom configuration directory
--config-name NAME   Config file name without .yaml (default: config)
-k, --runs N         Number of independent runs (creates N separate jobs)
--multirun           Enable parameter sweeps
--slurm              Submit job(s) to SLURM using submitit launcher
--slurm-partition    SLURM partition (overrides config)
--slurm-time         SLURM time limit in minutes (overrides config)
--cfg [job|hydra|all]  Print config and exit
```

### Examples

```bash
# Custom job name
prompt-siren run benign --job-name=baseline-gpt4 +dataset=agentdojo-workspace

# Custom output directory
prompt-siren run benign --jobs-dir=./experiments +dataset=agentdojo-workspace

# Override model
prompt-siren run benign +dataset=agentdojo-workspace agent.config.model=azure:gpt-4-turbo

# Set concurrency
prompt-siren run benign +dataset=agentdojo-workspace execution.concurrency=8

# Run specific tasks
prompt-siren run benign +dataset=agentdojo-workspace task_ids='["user_task_0","user_task_1"]'
```

### Parameter Sweeps

Use `--multirun` to run experiments across multiple parameter values:

```bash
# Sweep over models
prompt-siren run benign --multirun +dataset=agentdojo-workspace \
    agent.config.model=azure:gpt-4,azure:gpt-4-turbo

# Sweep over attacks
prompt-siren run attack --multirun +dataset=agentdojo-workspace \
    +attack=template_string,mini-goat

# Multi-dimensional sweep
prompt-siren run attack --multirun +dataset=agentdojo-workspace \
    +attack=template_string,mini-goat \
    agent.config.model=azure:gpt-4,azure:gpt-4-turbo
```

Each parameter combination creates a separate job.

### Multiple Runs

Use `-k/--runs` to run multiple independent trials:

```bash
# Run 10 independent trials
prompt-siren run attack -k 10 +dataset=agentdojo-workspace +attack=mini-goat

# Combine with SLURM to submit 10 SLURM jobs
prompt-siren run attack -k 10 --slurm +dataset=agentdojo-workspace +attack=mini-goat

# Combine with parameter sweeps (10 runs × N parameter combinations)
prompt-siren run attack -k 10 --slurm +dataset=agentdojo-workspace +attack=mini-goat \
    agent.config.model=azure:gpt-4,azure:gpt-5
```

### SLURM Submission

Use `--slurm` to submit jobs to a SLURM cluster instead of running locally:

```bash
# Submit a single job to SLURM
prompt-siren run attack --slurm +dataset=agentdojo-workspace +attack=mini-goat

# Submit a sweep to SLURM (each parameter combination becomes a SLURM job)
prompt-siren run attack --slurm +dataset=agentdojo-workspace +attack=mini-goat \
    agent.config.model=azure:gpt-4,azure:gpt-5

# Override SLURM partition and time limit
prompt-siren run attack --slurm --slurm-partition=devlab --slurm-time=120 \
    +dataset=agentdojo-workspace +attack=mini-goat
```

Note: `--slurm` automatically enables multirun mode since the submitit launcher requires it.

## Resuming Jobs

### `jobs resume`

Resume an interrupted job or retry failed tasks. Can target a single job or all jobs from a sweep.

```bash
# Resume a single job
prompt-siren jobs resume -p ./jobs/my-job

# Resume all jobs from a sweep
prompt-siren jobs resume -s 2026-01-31_17-01-04
```

The command loads the original configuration, skips completed tasks, and continues with remaining ones.

### Options

```bash
-p, --job-path PATH        Path to a single job directory
-s, --sweep-id ID          Sweep ID (timestamp) to resume all jobs from
--pattern GLOB             Glob pattern to match job directory names
-d, --jobs-dir PATH        Directory containing jobs (default: ./jobs, for sweep mode)
-e, --retry-on-error TYPE  Retry tasks that failed with this error (repeatable, default: CancelledError)
--only-failed              Only resume jobs with failures (sweep mode only)
--dry-run                  Show what would be resumed without running (sweep mode only)
--slurm                    Submit job(s) to SLURM instead of running locally
--slurm-partition PART     SLURM partition (overrides config)
--slurm-time MINUTES       SLURM time limit in minutes (overrides config)
--no-wait                  Don't wait for SLURM job(s) to complete
```

Note: Must specify exactly one of `-p`, `-s`, or `--pattern`.

### Default Behavior

By default, `jobs resume` automatically retries tasks that were cancelled (e.g., by Ctrl+C). This ensures that interrupted jobs can be cleanly resumed without manually specifying `--retry-on-error CancelledError`.

To disable automatic retries and only continue incomplete tasks, use `-e ''`:

```bash
prompt-siren jobs resume -p ./jobs/my-job -e ''
```

### Examples

```bash
# Basic resume (retries cancelled tasks by default)
prompt-siren jobs resume -p ./jobs/my-job

# Resume all jobs from a sweep
prompt-siren jobs resume -s 2026-01-31_17-01-04

# Resume only failed jobs from a sweep
prompt-siren jobs resume -s 2026-01-31_17-01-04 --only-failed

# Dry run to see what would be resumed
prompt-siren jobs resume -s 2026-01-31_17-01-04 --dry-run

# Resume with pattern matching
prompt-siren jobs resume --pattern '*_gpt-4o_*'

# Also retry tasks that timed out
prompt-siren jobs resume -p ./jobs/my-job -e TimeoutError -e CancelledError

# Resume with higher concurrency
prompt-siren jobs resume -p ./jobs/my-job execution.concurrency=16

# Submit resume job(s) to SLURM
prompt-siren jobs resume -p ./jobs/my-job --slurm
prompt-siren jobs resume -s 2026-01-31_17-01-04 --slurm

# Submit to SLURM without waiting for completion
prompt-siren jobs resume -s 2026-01-31_17-01-04 --slurm --no-wait
```

### Allowed Overrides on Resume

Only execution-related settings can be changed when resuming:

| Prefix | Description |
|--------|-------------|
| `execution.*` | Concurrency settings |
| `telemetry.*` | Logging and tracing |
| `output.*` | Output configuration (note: `output.jobs_dir` has no effect since the job directory is already set via `-p`) |
| `usage_limits.*` | Token and request limits |

Dataset, agent, and attack configurations cannot be modified on resume.

## Resuming Sweep Jobs

### `jobs resume-sweep`

Resume all jobs from a Hydra multirun sweep. This is useful when a parameter sweep (e.g., with `--multirun` or submitit_slurm launcher) partially fails due to rate limiting or other transient errors.

```bash
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 -d /path/to/jobs
```

### Options

```bash
-s, --sweep-id ID          Sweep ID (typically timestamp like 2026-01-31_17-01-04)
-p, --pattern PATTERN      Glob pattern to match job directory names (alternative to --sweep-id)
-d, --jobs-dir PATH        Directory containing jobs (default: ./jobs)
-e, --retry-on-error TYPE  Retry tasks that failed with this error (repeatable, default: ModelHTTPError, CancelledError)
--only-failed              Only resume jobs that have failed tasks
--dry-run                  Show jobs that would be resumed without running them
--slurm                    Submit jobs to SLURM instead of running locally
--slurm-partition PART     SLURM partition (overrides config)
--slurm-time MINUTES       SLURM time limit in minutes (overrides config)
--no-wait                  Don't wait for SLURM jobs to complete
```

### Examples

```bash
# Resume all jobs from a sweep by timestamp
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 -d /checkpoint/jobs

# Resume with reduced concurrency to avoid rate limits
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 execution.concurrency=3

# Resume only jobs that have failures
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 --only-failed

# Preview what would be resumed
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 --dry-run

# Match jobs by pattern (for sweeps without metadata)
prompt-siren jobs resume-sweep -p '*_gpt-4o_mini-goat*_run*' -d /path/to/jobs

# Retry only rate limit errors with lower concurrency
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 -e ModelHTTPError execution.concurrency=2

# Submit all sweep resumes to SLURM (parallel execution)
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 --slurm

# Submit to SLURM with specific partition and time limit
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 --slurm --slurm-partition devlab --slurm-time 120

# Submit to SLURM without waiting for completion
prompt-siren jobs resume-sweep -s 2026-01-31_17-01-04 --slurm --no-wait
```

### How Sweep Tracking Works

When running a multirun sweep (with `--multirun` or using a config with `hydra.mode: MULTIRUN`), Siren automatically tracks which jobs belong to the same sweep. This metadata is stored in a `.sweeps/` directory within your jobs directory.

For older sweeps without metadata, `resume-sweep` falls back to pattern matching based on the timestamp in job directory names.

## Viewing Results

### `results`

Aggregate and display results from completed jobs.

```bash
prompt-siren results
```

### Options

```bash
--jobs-dir PATH       Jobs directory (default: ./jobs)
--format FORMAT       Output format: table, json, csv (default: table)
--group-by LEVEL      Grouping: all, dataset, agent, attack, agent_name (default: all)
--k N                 Pass@k metric value (default: 1, repeatable)
```

### Examples

```bash
# Default table view
prompt-siren results

# From custom directory
prompt-siren results --jobs-dir=./experiments

# JSON output
prompt-siren results --format=json

# CSV for spreadsheets
prompt-siren results --format=csv > results.csv

# Group by model
prompt-siren results --group-by=agent_name

# Group by attack type
prompt-siren results --group-by=attack

# Compute pass@5 metric
prompt-siren results --k=5

# Multiple pass@k values
prompt-siren results --k=1 --k=5 --k=10
```

### Understanding pass@k

- **pass@1** (default): Average score across all runs for each task
- **pass@k** (k>1): Task passes if at least one of k runs achieves a perfect score (1.0)

### Output Columns

| Column | Description |
|--------|-------------|
| `dataset` | Dataset type |
| `agent_type` | Agent type |
| `agent_name` | Model name |
| `attack_type` | Attack type or "benign" |
| `benign_pass@k` | Benign evaluation score |
| `attack_pass@k` | Attack evaluation score |
| `n_tasks` | Number of tasks |
| `avg_n_samples` | Average runs per task |

## Configuration Commands

### `config export`

Export default configuration files:

```bash
prompt-siren config export ./config
```

### `config validate`

Validate configuration without running:

```bash
prompt-siren config validate +dataset=agentdojo-workspace
prompt-siren config validate +dataset=agentdojo-workspace +attack=template_string
```

## Job Storage

Jobs are stored in the `--jobs-dir` directory (default: `./jobs`):

```
jobs/
  <job_name>/
    config.yaml       # Experiment configuration snapshot
    index.jsonl       # Task results index
    <task_id>/
      <run_id>/
        result.json   # Task result
        execution.json # Full execution data
```

Job names follow the format: `<dataset>_<agent>_<attack|benign>_<timestamp>`

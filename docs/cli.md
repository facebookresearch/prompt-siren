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
--multirun           Enable parameter sweeps
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

## Resuming Jobs

### `jobs resume`

Resume an interrupted job or retry failed tasks.

```bash
prompt-siren jobs resume -p ./jobs/my-job
```

The command loads the original configuration, skips completed tasks, and continues with remaining ones.

### Options

```bash
-p, --job-path PATH        Path to job directory (required)
-e, --retry-on-error TYPE  Retry tasks that failed with this error (repeatable, default: CancelledError)
```

### Default Behavior

By default, `jobs resume` automatically retries tasks that were cancelled (e.g., by Ctrl+C). This ensures that interrupted jobs can be cleanly resumed without manually specifying `--retry-on-error CancelledError`.

To disable automatic retries and only continue incomplete tasks, use `-e ''`:

```bash
prompt-siren jobs resume -p ./jobs/my-job -e ''
```

### Examples

```bash
# Basic resume (retries cancelled tasks by default)
prompt-siren jobs resume -p ./jobs/agentdojo-workspace_plain_gpt-4_benign_2025-01-15_14-30-00

# Also retry tasks that timed out
prompt-siren jobs resume -p ./jobs/my-job -e TimeoutError -e CancelledError

# Retry only timeouts, not cancelled tasks
prompt-siren jobs resume -p ./jobs/my-job -e TimeoutError

# Disable all retries (only continue incomplete tasks)
prompt-siren jobs resume -p ./jobs/my-job -e ''

# Resume with higher concurrency
prompt-siren jobs resume -p ./jobs/my-job execution.concurrency=16

# Resume with console tracing enabled
prompt-siren jobs resume -p ./jobs/my-job telemetry.trace_console=true
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

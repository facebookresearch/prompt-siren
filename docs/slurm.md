# SLURM & Parallel Sweeps

Run experiments and parameter sweeps on SLURM clusters.

## Quick Start

Use the `--slurm` flag to submit jobs to SLURM:

```bash
# Submit a single job
prompt-siren run attack --slurm +dataset=agentdojo-workspace +attack=mini-goat

# Run 10 independent trials on SLURM
prompt-siren run attack -k 10 --slurm +dataset=agentdojo-workspace +attack=mini-goat

# Submit a sweep (2 models × 10 runs = 20 SLURM jobs)
prompt-siren run attack -k 10 --slurm \
  +dataset=agentdojo-workspace +attack=mini-goat \
  agent.config.model=azure:gpt-4o,azure:gpt-5

# Override SLURM settings
prompt-siren run attack --slurm --slurm-partition=devlab --slurm-time=120 \
  +dataset=agentdojo-workspace +attack=mini-goat
```

## Setup

1. Export config and customize SLURM settings:
   ```bash
   uv run prompt-siren config export
   # Edit config/hydra/launcher/submitit_slurm.yaml with your account/qos
   ```

2. Set `telemetry.otel_endpoint: null` in config (no local tracing on SLURM nodes).

## Using --slurm Flag

The `--slurm` flag works with all commands:

| Command | Behavior |
|---------|----------|
| `jobs start --slurm` | Submits via Hydra's submitit launcher |
| `jobs resume -p ... --slurm` | Submits single resume job to SLURM |
| `jobs resume -s ... --slurm` | Submits all sweep resume jobs in parallel |

### Resume Failed Sweep Jobs

```bash
# Resume all jobs from a sweep on SLURM
prompt-siren jobs resume -s 2026-01-31_17-01-04 --slurm

# Resume only failed jobs from a sweep
prompt-siren jobs resume -s 2026-01-31_17-01-04 --only-failed --slurm

# Resume with lower concurrency to avoid rate limits
prompt-siren jobs resume -s 2026-01-31_17-01-04 --slurm execution.concurrency=2

# Preview what would be resumed
prompt-siren jobs resume -s 2026-01-31_17-01-04 --dry-run
```

## Advanced: Hydra Sweep Configs

For complex sweeps with many parameters, define everything in a config file instead of CLI flags:

```yaml
# config/my-sweep.yaml
defaults:
  - config
  - override /hydra/launcher: submitit_slurm  # <-- SLURM launcher set here
  - _self_

hydra:
  mode: MULTIRUN  # <-- Multirun mode set here
  sweeper:
    params:
      +attack: attack1,attack2
      agent.config.model: azure:gpt-4o,azure:gpt-5
      +run_id: range(10)  # <-- Equivalent to -k 10 on CLI

dataset:
  type: agentdojo
  config: { suite_name: workspace }
```

Run with:
```bash
uv run --env-file .env prompt-siren run attack --config-name=my-sweep
```

**Note:** When the launcher and mode are set in the config file, you don't need `--slurm`, `--multirun`, or `-k` flags. The `+run_id: range(10)` in the sweeper params is equivalent to `-k 10` on the CLI. This approach is useful for reproducible experiment configurations that can be version-controlled.

## Key Config Options

In `config/hydra/launcher/submitit_slurm.yaml`:
- `partition`: SLURM partition
- `timeout_min`: Job timeout (default 240)
- `account` / `qos`: Your SLURM account and QoS

## Troubleshooting

- **API keys missing**: Ensure `.env` is loaded via `uv run --env-file .env`
- **Jobs fail immediately**: Check `partition`, `account`, and `qos` settings
- **Check logs**: `cat /path/to/slurm-logs/.submitit/*/2*_0_log.out`

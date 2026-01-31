# SLURM & Parallel Sweeps

Run parameter sweeps on SLURM clusters using Hydra's `--multirun` with the Submitit launcher.

## Setup

1. Export config and customize SLURM settings:
   ```bash
   uv run prompt-siren config export
   # Edit config/hydra/launcher/submitit_slurm.yaml with your account/qos
   ```

2. Set `telemetry.otel_endpoint: null` in config (no local tracing on SLURM nodes).

## Running Sweeps

### Command-line Sweep
```bash
uv run --env-file .env prompt-siren run attack \
  --multirun \
  +dataset=agentdojo-workspace \
  +attack=mini-goat \
  agent.config.model=azure:gpt-4o,azure:gpt-5 \
  '+run_id=range(10)' \
  hydra/launcher=submitit_slurm
```
Creates 20 SLURM jobs (2 models × 10 runs).

### Experiment Config File (Recommended)
Define sweeps in a config file for reproducibility. Place experiment configs in `config/` 
alongside `config.yaml` so they can inherit the base config and access `attack/` configs.

```yaml
# config/my-sweep.yaml
defaults:
  - config  # Inherit base config (required for attack/ search path)
  - override /hydra/launcher: submitit_slurm
  - _self_

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      +attack: mini-goat,mini-goat-aggressive
      agent.config.model: azure:gpt-4o,azure:gpt-5
      +run_id: range(10)

dataset:
  type: agentdojo
  config: { suite_name: workspace }

agent:
  type: plain
  config: { model: azure:gpt-4o }

# ... rest of config overrides
```

Run with:
```bash
uv run --env-file .env prompt-siren run attack --config-name=my-sweep
```

No `--multirun` flag needed - `hydra.mode: MULTIRUN` is set in config.

## Attack Variants

Use `variant_name` in attack configs to distinguish versions in results:
```yaml
# config/attack/mini-goat-aggressive.yaml
type: mini-goat
config:
  variant_name: "aggressive"
  max_turns: 10
  # ...
```

Results will show `mini-goat_aggressive` instead of just `mini-goat`.

## Key Config Options

In `config/hydra/launcher/submitit_slurm.yaml`:
- `account` / `qos`: Your SLURM account and QoS
- `timeout_min`: Job timeout (default 240)
- `max_num_timeout`: Limit concurrent jobs (0 = no limit)

## Viewing Results

```bash
uv run prompt-siren results --group-by=attack  # Group by attack variant
uv run prompt-siren results --k=1 --k=5        # pass@k metrics
```

## Troubleshooting

- **API keys missing**: Ensure `.env` is loaded and `additional_parameters.export: ALL` in launcher config
- **Jobs fail immediately**: Check `account` and `qos` settings
- **Check logs**: `cat /path/to/slurm-logs/.submitit/*/2*_0_log.out`

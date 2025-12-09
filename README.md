# Prompt Injection Workbench

[![CI](https://github.com/facebookresearch/prompt-siren/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/prompt-siren/actions/workflows/ci.yml)

A research workbench for developing and testing attacks against large language models, with a focus on prompt injection vulnerabilities and defenses.

## Key Features

- **State Machine Design**: Fine-grained control over agent execution for advanced attack scenarios
- **SWE-bench Support**: Benchmark agents on real-world code editing tasks from SWE-bench
- **Hydra Configuration**: Powerful experiment orchestration with parameter sweeps
- **Extensible Architecture**: Plugin system for custom agents, attacks, and environments
- **Usage Limits**: Built-in cost and resource controls
- **Experiment Tracking**: Automatic caching and result organization

## Quick Start

### Installation

Install the core package with desired optional features:

```sh
# Full installation (all features)
uv sync --all-extras

# Or install only what you need:
uv sync --extra agentdojo      # AgentDojo benchmark support
uv sync --extra swebench       # SWE-bench support
uv sync --extra docker         # Docker sandbox manager
uv sync --extra playwright     # Web automation environment

# Combine multiple extras
uv sync --extra agentdojo --extra docker
```

**Available optional dependencies:**

| Extra | Description |
|-------|-------------|
| `agentdojo` | AgentDojo dataset, environment, and attacks |
| `swebench` | SWE-bench dataset for code editing benchmarks |
| `docker` | Docker sandbox manager |
| `playwright` | Web automation environment |

Set up environment variables:
```sh
cp .env.example .env  # Fill in API keys
```

Export default configuration:
```sh
# Export to ./config (default)
uv run prompt-siren config export

# Export to custom directory
uv run prompt-siren config export ./my_config
```

Run experiments:
```sh
# Run benign-only evaluation
uv run prompt-siren run benign +dataset=agentdojo-workspace

# Run with attack
uv run prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string

# Run SWE-bench evaluation (requires Docker)
uv run prompt-siren run benign +dataset=swebench

# Run SWE-bench with specific instances
uv run prompt-siren run benign +dataset=swebench dataset.config.instance_ids='["django__django-11179"]'

# Run SWE-bench Lite (smaller benchmark)
uv run prompt-siren run benign +dataset=swebench dataset.config.dataset_name="SWE-bench/SWE-bench_Lite"

# Override parameters
uv run prompt-siren run benign +dataset=agentdojo-workspace agent.config.model=azure:gpt-5

# Parameter sweep (multirun)
uv run prompt-siren run benign --multirun +dataset=agentdojo-workspace agent.config.model=azure:gpt-5,azure:gpt-5-nano

# Validate configuration without running
uv run prompt-siren config validate +dataset=agentdojo-workspace

# Use config file with environment/attack included (no overrides needed)
uv run prompt-siren run attack --config-dir=./my_config
```

**Tip**: Environment and attack can be specified via CLI overrides or included directly in config files. See the [Configuration Guide](docs/configuration.md) for details.

## Analyzing Results

After running experiments, use the `results` command to aggregate and analyze results:

```sh
# View results with default settings (pass@1, grouped by all configs)
uv run prompt-siren results

# Specify custom output directory
uv run prompt-siren results --output-dir=./traces

# Group results by different dimensions
uv run prompt-siren results --group-by=model
uv run prompt-siren results --group-by=env
uv run prompt-siren results --group-by=agent
uv run prompt-siren results --group-by=attack

# Compute pass@k metrics (k>1)
uv run prompt-siren results --k=5
uv run prompt-siren results --k=10

# Compute multiple pass@k metrics simultaneously
uv run prompt-siren results --k=1 --k=5 --k=10

# Different output formats
uv run prompt-siren results --format=json
uv run prompt-siren results --format=csv
```

### Understanding pass@k Metrics

- **pass@1** (default): Averages scores across all runs for each task. A continuous metric showing average performance.
- **pass@k** (k>1): Binary success metric. A task "passes" if at least one of k runs achieves a perfect score (1.0). Uses statistical estimation when more than k samples are available.

### Results Columns

The results table includes:
- **Configuration columns**: `env_type`, `agent_type`, `attack_type`, `model`, `config_hash`
- **Metric columns**: `benign_pass@k`, `attack_pass@k` - The pass@k scores
- **Metadata columns**:
  - `n_tasks` - Total number of tasks aggregated
  - `avg_n_samples` - Average number of runs per task
  - `k` - The k value (when computing multiple pass@k metrics)
## Platform Requirements

- **Python**: 3.10+
- **Package Manager for development**: `uv` (for dependency management)
- **Operating System**: Linux or macOS (Windows not supported)
- **Docker**: Required for SWE-bench integration and some environments
  - Must be running and accessible
  - Base images should have `/bin/bash` available (Alpine images need `bash` package)

## Documentation

- **[Configuration Guide](docs/configuration.md)** - Hydra configuration and parameter overrides
- **[Usage Limits](docs/usage_limits.md)** - Resource limits and cost controls
- **[Plugins](docs/plugins/README.md)** - Adding custom agents, attacks, and environments

## Development

```sh
# Lint and format
uv run ruff check --fix
uv run ruff format
uv run basedpyright

# Test
uv run pytest -v
```

# License
Prompt Siren is licensed under an [MIT License](https://github.com/facebookresearch/prompt-siren/blob/main/LICENSE.md).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a workbench designed for security research to test attacks against large language models and prompt injection defenses. The project uses `pydantic-ai` as its core AI framework and supports multiple environments for testing. It features a Hydra-based configuration system for flexible experiment orchestration with parameter sweeps and overrides.

## Development Commands

### Linting and Type Checking
```sh
uv run ruff check --fix
uv run ruff format
uv run basedpyright
```

### Testing
```sh
# Run all tests (unit + integration)
uv run pytest -vx

# Run unit tests only (fast, no Docker needed - CI/CD default)
uv run pytest -vx -m "not docker_integration"

# Run integration tests only (requires Docker)
uv run pytest -vx -m docker_integration
```

**Note**: Integration tests require Docker to be running. CI/CD pipelines skip integration tests by default for speed.

### Running Experiments
```sh
# Export default configuration (first time setup)
uv run prompt-siren config export

# Run benign-only evaluation
uv run prompt-siren run benign +dataset=agentdojo-workspace

# Run attack evaluation
uv run prompt-siren run attack +dataset=agentdojo-workspace +attack=template_string

# Run SWE-bench evaluation
uv run prompt-siren run benign +dataset=swebench

# Run SWE-bench with specific instances
uv run prompt-siren run benign +dataset=swebench dataset.config.instance_ids='["django__django-11179"]'

# Override parameters
uv run prompt-siren run benign +dataset=agentdojo-workspace agent.config.model=azure:gpt-5 execution.concurrency=4

# Parameter sweep (use hydra.mode=MULTIRUN for sweeping over multiple values)
uv run prompt-siren run benign --multirun +dataset=agentdojo-workspace agent.config.model=azure:gpt-5,azure:gpt-5-nano

# Parameter sweep with multiple parameters
uv run prompt-siren run benign --multirun +dataset=agentdojo-workspace agent.config.model=azure:gpt-5,azure:gpt-5-nano execution.concurrency=1,4

# Parameter sweep with attacks
uv run prompt-siren run attack --multirun +dataset=agentdojo-workspace +attack=template_string,mini-goat

# Validate configuration
uv run prompt-siren config validate +dataset=agentdojo-workspace +attack=template_string

# Run with config file that includes dataset/attack (no overrides needed)
uv run prompt-siren run attack --config-dir=./my_config
```

**Note**: Dataset and attack can be specified either via command-line overrides (`+dataset=...`, `+attack=...`) or by including them in your config file's `defaults` list. See docs/configuration.md for details.

**Platform Requirements**:
- Linux or macOS only (Windows not supported)
- Docker required for SWE-bench
- Base Docker images must have `/bin/bash` available

### Development guidelines

- Always use modern 3.10+ type hints:
  - `list`, `dict`, `set` instead of importing from `typing`.
  - `|` instead of `Union` and `Optional`.
- Always write meaningful tests for new features. Avoid tests that are obvious.
- Always lint and type check.
- Avoid using `type: ignore`, unless the issue is the result of a deliberate choice (e.g., in tests).
- Avoid using `cast`.
- Only use module-level imports instead of local imports, unless it's to import optional dependencies or to handle circular imports.

## Architecture

### Core Components

1. **Agent System** (`src/prompt_siren/agents/`)
   - `abstract.py` - Defines the `AbstractAgent` protocol for agent interfaces
   - `plain.py` - Implements the main agent logic for running tasks with pydantic-ai
   - `_utils.py` - Helper utilities for agent operations
   - `registry.py` - Agent plugin registration system
   - Supports tool execution with injection attack capabilities

2. **Dataset System** (`src/prompt_siren/datasets/`)
   - `abstract.py` - Defines `AbstractDataset` protocol for dataset interfaces
   - `agentdojo_dataset.py` - AgentDojo dataset implementation
   - `swebench_dataset/` - SWE-bench dataset for code editing benchmarks
     - Uses multi-stage Docker builds (base/env/instance) with caching
     - Jinja2 template system for prompts
     - Official SWE-bench test harness for evaluation
   - `registry.py` - Dataset plugin registration system
   - Datasets provide:
     - Collections of tasks (benign, malicious, and task couples)
     - Specification of which environment type to use
     - Environment configuration for that environment
   - Separates task data from execution context (environments)
   - Allows multiple datasets to share the same environment type

3. **Environment System** (`src/prompt_siren/environments/`)
   - `abstract.py` - Base `Environment` class provides the interface for different execution environments
   - Environment states (`env_state`) are managed per-task and passed separately from environment
   - Two-level context management for resource lifecycle:
     - `create_batch_context()` - Batch-level setup for expensive resources (browsers, servers)
     - `create_task_context()` - Per-task context with fresh environment state
   - Environments handle rendering of outputs and injection vector detection
   - `registry.py` - Environment plugin registration system
   - Key environments:
     - `agentdojo/` - AgentDojo environment integration
     - `playwright.py` - Playwright-based web automation environment

4. **Attack System** (`src/prompt_siren/attacks/`)
   - `abstract.py` - Base attack abstractions
   - `InjectionAttack` type defines attack payloads (in `types.py`)
   - `registry.py` - Attack plugin registration system
   - Attack implementations:
     - `nanogcg/gcg.py` - GCG (Greedy Coordinate Gradient) attack
     - `template_string_attack.py` - Template string-based attacks
     - `dict_attack.py` - Dictionary-based attacks from config or files
     - `mini_goat_attack.py` - Mini-GOAT attack implementation
     - `target_string_attack.py` - Target string-based attacks
   - `attack_utils.py` provides common utilities

5. **Configuration System** (`src/prompt_siren/config/`)
   - `experiment_config.py` - Pydantic-based configuration schema defining ExperimentConfig
   - `export.py` - Configuration export functionality
   - `exceptions.py` - Configuration-specific exceptions
   - `registry_bridge.py` - Bridge between Hydra config and plugin systems
   - `default/` - Default Hydra configuration files
   - Integrates with Hydra for powerful experiment orchestration

6. **Task System** (`src/prompt_siren/tasks.py`)
   - `Task` - Defines individual tasks with evaluators
   - `TaskCouple` - Pairs benign and malicious tasks for evaluation
   - `TaskResult` - Contains execution results and environment state
   - `TaskEvaluator` - Protocol for task evaluation functions
   - `EvaluationResult` - Stores evaluation scores from evaluators

7. **Task Runner** (`src/prompt_siren/run.py`)
   - `run_tasks` orchestrates concurrent task execution
   - Manages batch and task context lifecycle using two-level context management
   - Manages concurrency limits and result collection
   - Integrates agents, datasets, environments, and attacks
   - Handles task evaluation and result aggregation

8. **CLI Interface** (`src/prompt_siren/cli.py` and `src/prompt_siren/hydra_app.py`)
   - `cli.py` - Click-based CLI with structured subcommands (config, run)
   - `hydra_app.py` - Hydra integration for configuration management
   - Entry point: `prompt_siren` command
   - Subcommands: `config export`, `config validate`, `run benign`, `run attack`
   - Execution mode (benign vs attack) determined by CLI command, not config
   - Supports parameter overrides and multi-run sweeps via `--multirun`
   - Automatic experiment tracking and caching

9. **Telemetry System** (`src/prompt_siren/telemetry/`)
   - OpenTelemetry instrumentation for observability
   - Conversation logging capabilities
   - `formatted_span.py` - Span formatting utilities
   - `workbench_spans.py` - Workbench-specific span management
   - `pydantic_ai_processor/` - PydanticAI telemetry integration

10. **Type System** (`src/prompt_siren/types.py`)
   - Core type definitions for the workbench
   - `InjectionAttack` and related attack types
   - `InjectableUserPromptPart`, `InjectableModelRequest` for injection modeling
   - `AttackConfig`, `AttackMetadata` for attack configuration
   - Content types: `StrContentAttack`, `BinaryContentAttack`

11. **Plugin System** (`src/prompt_siren/registry_base.py`)
    - Base registry functionality for plugin management
    - Supports dynamic component registration via entry points
    - Each subsystem (agents, attacks, datasets, providers) has its own registry
    - Flexible design allows components with or without config classes

12. **Provider System** (`src/prompt_siren/providers/`)
    - `abstract.py` - Defines `AbstractProvider` protocol for model providers
    - `registry.py` - Provider registry and `infer_model` function
    - Provider implementations:
      - `bedrock.py` - AWS Bedrock provider for Anthropic models
      - `llama.py` - Llama API provider
    - Providers handle model creation for specific prefixes (e.g., "bedrock:", "llama:")
    - Read configuration from environment variables (no Pydantic config classes)
    - Automatically integrated via `infer_model` in agent configuration
    - Factory functions take no parameters (unlike other components)

13. **Supporting Modules**
    - `run_persistence.py` - Persistence layer for experiment results
    - `tools_utils.py` - Utilities for tool handling

### Task Selection

Task selection is configured via the `task_ids` field in `ExperimentConfig`:
- `task_ids: null` (default) - Run all tasks appropriate for the execution mode:
  - Benign mode: all unique benign tasks
  - Attack mode: all task couples
- `task_ids: ["task1", "task2"]` - Run specific tasks by ID
  - Task IDs with `:` are treated as couple IDs (e.g., `"benign_id:malicious_id"`)
  - Task IDs without `:` are treated as individual task IDs
- Task selection logic is handled by `get_selected_tasks()` in `hydra_app.py`
- Execution mode (benign vs attack) is determined by CLI command, not stored in config

### Plugin System

The workbench uses Python entry points for extensibility. Components are registered in `pyproject.toml`:

- **Agent plugins**: `prompt_siren.agents` entry point
  - `plain` - Plain agent implementation

- **Attack plugins**: `prompt_siren.attacks` entry point
  - `template_string` - Template string attacks
  - `agentdojo` - Alias for template_string (backwards compatibility)
  - `dict` - Dictionary-based attacks from config
  - `file` - Dictionary-based attacks from file
  - `mini-goat` - Mini-GOAT attacks

- **Dataset plugins**: `prompt_siren.datasets` entry point
  - `agentdojo-workspace` - AgentDojo workspace dataset
  - `swebench` - SWE-bench dataset for code editing benchmarks

- **Provider plugins**: `prompt_siren.providers` entry point
  - `bedrock` - AWS Bedrock provider for Anthropic models
  - `llama` - Llama API provider
  - Note: Providers don't use config classes; they read from environment variables

Plugins can be added by implementing the appropriate protocol and registering via entry points.

### Available Environment Implementations

Environments are not registered as plugins; instead, datasets directly instantiate and own their environment instances:

- **AgentDojoEnv** (`src/prompt_siren/environments/agentdojo_env.py`) - For AgentDojo benchmark suites
  - Used by: AgentDojo datasets
  - Features: Snapshottable environment state, placeholder injection, AgentDojo tool integration

- **PlaywrightEnv** (`src/prompt_siren/environments/playwright.py`) - For web automation tasks
  - Features: Non-snapshottable (uses tool replay), browser automation, webpage interaction

Datasets specify which environment implementation to use and configure it during initialization.

### Key Design Patterns

- **Dependency Injection**: Uses generic types for flexible dependency management
- **Protocol-based Design**: Uses Python protocols for extensibility
- **Async-first**: All core operations are asynchronous
- **Two-level Context Management**: Environments use a two-tier context system:
  - Batch context for expensive resource setup shared across tasks
  - Task context for per-task execution with fresh environment state
- **Plugin Architecture**: Entry points system for component registration

### Testing Structure

- Uses `pytest` with `anyio` for async testing
- `conftest.py` provides mock environments and fixtures
- Test files mirror source structure under `tests/`

## Dependencies

### Core Dependencies (always installed)
- `pydantic-ai-slim[anthropic,bedrock,openai]` - AI framework with model integrations
- `click` - CLI framework for structured subcommands
- `hydra-core` - Configuration management and experiment orchestration
- `hydra-submitit-launcher` - Submitit launcher for Hydra
- `logfire` - Logging and observability
- `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` - Telemetry
- `omegaconf` - Configuration management
- `pyyaml` - YAML parsing
- `aiohttp` - Async HTTP client
- `filelock` - File locking utilities
- `pandas` - Data analysis for results aggregation

### Optional Dependencies

Install optional features with: `pip install 'prompt-siren[feature]'`

| Group | Dependencies | Purpose |
|-------|--------------|---------|
| `[agentdojo]` | `agentdojo>=0.1.35` | AgentDojo dataset, environment, and attacks |
| `[swebench]` | `swebench`, `jinja2>=3.1.6` | SWE-bench dataset for code editing benchmarks |
| `[docker]` | `aiodocker>=0.24.0` | Docker sandbox manager |
| `[playwright]` | `playwright>=1.54.0` | Web automation environment |
| `[all]` | All optional deps | Full installation with all features |

**Examples:**
```bash
# Minimal install (core only)
pip install prompt-siren

# AgentDojo support
pip install 'prompt-siren[agentdojo]'

# SWE-bench with Docker sandbox manager
pip install 'prompt-siren[swebench,docker]'

# Everything
pip install 'prompt-siren[all]'
```

### Requirements
- **Python**: Requires Python 3.10+
- **Package Manager**: Uses `uv` for dependency management
- **Platform**: Linux and macOS only (Windows not supported due to sandbox architecture)

## Useful documentation links

Check out these pages before going to a specific page. Navigate the .md links as they are more LLM-friendly.

- PydanticAI: https://ai.pydantic.dev/llms.txt
- Pydantic: https://docs.pydantic.dev/latest/llms.txt
- Hydra: https://hydra.cc/docs/intro/

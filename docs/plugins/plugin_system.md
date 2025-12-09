# Plugin System

Siren uses an entry point-based plugin system for automatic discovery and configuration of components (agents, attacks, datasets, providers, and sandbox managers).

## Key Features

- **Entry point discovery**: Components are automatically discovered via Python entry points
- **Factory-based design**: Uses factory functions for component creation
- **Composable configuration**: Each component type can have its own specialized Pydantic configuration (except providers)
- **File-based configuration**: Load configurations from YAML/JSON/TOML files
- **Type-safe registry**: Type safety when creating components with proper generics
- **Plugin ecosystem**: External packages can add components by simply defining entry points

## Component Types

### Agents

AI agents that interact with environments and execute tasks.

**Available Agents:**
- `plain`: Basic agent with configurable model and settings

### Attacks

Prompt injection attack strategies for testing model robustness.

**Available Attacks:**
- `agentdojo`: AgentDojo attack with customizable template and user name
- `dict`: Dictionary-based attacks from config
- `file`: Dictionary-based attacks from file
- `mini-goat`: Mini-GOAT attacks

### Datasets

Collections of tasks with associated environment specifications.

**Available Datasets:**
- `agentdojo`: AgentDojo workspace benchmark dataset
- `swebench`: SWE-bench dataset for code editing benchmarks

**Note**: Datasets provide tasks and specify which environment type to use for execution. This separation allows multiple datasets to share the same environment type.

### Providers

Model provider implementations for creating model instances with specific prefixes.

**Available Providers:**
- `bedrock`: AWS Bedrock provider for Anthropic models
- `llama`: Llama API provider

**Note**: Unlike other components, providers don't use Pydantic config classes. They read configuration from environment variables (e.g., `AWS_REGION`, `LLAMA_API_KEY`) following the same pattern as pydantic-ai's built-in providers.

### Environments

Execution contexts for running tasks (rendering, injection detection, resource management).

**Available Environments:**
- `agentdojo`: AgentDojo environment for workspace tasks
- `playwright`: Playwright-based web automation environment
- `bash`: Bash environment for command execution

**Note**: Environments are not registered as plugins. They provide execution context only. Tasks are provided by datasets, and datasets own their environment instances.

### Sandbox Managers

Manage sandboxed execution environments for isolated task execution.

**Available Sandbox Managers:**
- `local-docker`: Local Docker-based sandbox manager

## Creating Plugins

For detailed implementation examples, see [Plugins](README.md).

## Plugin Discovery

The plugin system automatically discovers components at runtime:

1. **Scan entry points**: Looks for entry points in the appropriate group
2. **Load factory functions**: Imports and validates factory functions
3. **Type checking**: Validates factories return the correct component type
4. **Configuration validation**: Validates component configurations using Pydantic

## Usage

Components are automatically available once registered:

```bash
# Use different component types
uv run prompt-siren run benign +dataset=agentdojo agent.config.model=azure:gpt-5
uv run prompt-siren run attack +dataset=agentdojo +attack=template_string

# Override specific component configurations
uv run prompt-siren run benign +dataset=agentdojo agent.type=plain
uv run prompt-siren run attack +dataset=agentdojo +attack=mini-goat
```

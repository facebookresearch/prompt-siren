# Docker Execution Backends

The Prompt Siren Workbench supports Docker execution backends for running containerized tasks.

## Overview

Docker execution backends handle the creation, management, and execution of Docker containers for sandbox environments.

## Local Docker Backend

The local Docker backend executes containers directly on your machine using the Docker daemon.

### Requirements

- Docker installed and running on your machine
- Docker socket accessible (typically `/var/run/docker.sock`)
- `DOCKER_HOST` environment variable set in your `.env` file

### Setup

1. **Install Docker** (if not already installed):
   ```bash
   docker --version
   ```

2. **Configure environment** in `.env`:
   ```bash
   DOCKER_HOST="unix:///var/run/docker.sock"
   ```

3. **Verify Docker is running**:
   ```bash
   docker ps
   ```

### Usage

Local Docker is the default backend and requires no special configuration:

```bash
# Run with local Docker (default)
uv run --env-file .env prompt-siren run benign +dataset=swebench

# Or explicitly specify
uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=local
```

### Advantages

- **Fast**: No network latency, containers run locally
- **Easy debugging**: Direct access to containers via `docker` CLI
- **No quotas**: Limited only by your machine's resources

### Limitations

- Requires Docker daemon running
- Limited by local machine resources
- Cannot run on machines without Docker

## Hydra Configuration

You can set the Docker backend in your Hydra configuration file or via command-line overrides.

### In Configuration File

Create or modify `config.yaml`:

```yaml
defaults:
  - _self_
  - dataset: swebench

sandbox_manager:
  type: local-docker
  config:
    docker_client: local
```

Then run without needing overrides:

```bash
uv run --env-file .env prompt-siren run benign --config-dir=./config
```

### Via Command-Line Overrides

Override the backend at runtime:

```bash
# Explicitly use local Docker
uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=local
```

## End-to-End Example

Here's a complete example of running SWE-bench with Docker:

```bash
# 1. Set up environment
cat > .env <<EOF
DOCKER_HOST="unix:///var/run/docker.sock"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.azure-api.net"
AZURE_OPENAI_API_KEY="your-key"
OPENAI_API_VERSION="2025-04-01-preview"
EOF

# 2. Verify Docker is running
docker ps

# 3. Run with local Docker
uv run --env-file .env prompt-siren run benign +dataset=swebench \
  agent.config.model=azure:gpt-4o \
  'dataset.config.instance_ids=["django__django-11179"]'
```

## Troubleshooting

### Local Docker Issues

**Problem**: `Cannot connect to Docker daemon`
```bash
# Solution: Verify Docker is running
docker ps

# Check DOCKER_HOST is set correctly
echo $DOCKER_HOST
```

**Problem**: `Permission denied` accessing Docker socket
```bash
# Solution: Check socket permissions
ls -l /var/run/docker.sock

# May need to add your user to docker group (requires logout/login)
sudo usermod -aG docker $USER
```

## Plugin System

Local Docker is implemented as a plugin using the Docker client registry system. You can create custom Docker client plugins by:

1. Implementing the `AbstractDockerClient` protocol
2. Creating a factory function
3. Registering via entry points

See [custom_components.md](custom_components.md) for details on creating custom plugins.


## See Also

- [Sandbox Manager Documentation](sandbox_manager.md) - General sandbox manager concepts
- [Configuration Guide](configuration.md) - Hydra configuration details
- [Custom Components](custom_components.md) - Creating custom plugins

# Docker Execution Backends

The Prompt Siren Workbench supports two Docker execution backends for running containerized tasks: **Local Docker** and **DES (Docker Execution Service)**.

## Overview

Docker execution backends handle the creation, management, and execution of Docker containers for sandbox environments. The backend you choose determines where and how your containers run.

### Available Backends

1. **Local Docker** - Runs containers on your local machine using the Docker daemon
2. **DES (Docker Execution Service)** - Runs containers on a remote VM managed by Meta's internal service

## Local Docker Backend

The local Docker backend executes containers directly on your machine using the Docker daemon.

### Requirements

- Docker installed and running on your machine
- Docker socket accessible (typically `/var/run/docker.sock`)
- `DOCKER_HOST` environment variable set in your `.env` file

### Setup

1. **Install Docker** (if not already installed):
   ```bash
   # On devserver, Docker is typically pre-installed
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
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench

# Or explicitly specify
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
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

## DES (Docker Execution Service) Backend

DES is Meta's internal service for running Docker containers on remote VMs. Useful when you don't have access to a local Docker daemon or need isolated execution.

### Requirements

- Access to Meta's internal network
- `des_exec` binary available in your PATH
- No local Docker installation required

### Setup

1. **Verify DES access**:
   ```bash
   which des_exec
   # Should show path to des_exec binary
   ```

2. **No additional environment variables needed** - DES handles authentication automatically

### Usage

Specify DES as the Docker client backend using Hydra configuration:

```bash
# Basic DES usage
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=des

# Configure session timeout (in seconds)
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=des \
  sandbox_manager.config.des_session_timeout=7200

# Pre-load specific Docker images on the VM
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=des \
  'sandbox_manager.config.des_docker_images=["vmvm-registry.fbinfra.net/debian:bookworm-slim","python:3.12"]'
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sandbox_manager.config.docker_client` | string | `"local"` | Docker client backend to use (`"local"` or `"des"`) |
| `sandbox_manager.config.des_session_timeout` | int | `3600` | DES session timeout in seconds |
| `sandbox_manager.config.des_docker_images` | list[str] | `[]` | Docker images to pre-load on the DES VM |

### Advantages

- **No local Docker required**: Runs on remote VMs
- **Isolated execution**: Each session gets a fresh VM
- **Managed infrastructure**: Meta handles VM provisioning and cleanup

### Limitations

- **Network latency**: Remote execution adds overhead
- **Session timeouts**: Long-running tasks may need increased timeout
- **Image pre-loading**: Must specify images to make available on VM

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
    docker_client: des  # Use DES backend
    des_session_timeout: 7200
    des_docker_images:
      - vmvm-registry.fbinfra.net/debian:bookworm-slim
      - python:3.12
```

Then run without needing overrides:

```bash
with-proxy uv run --env-file .env prompt-siren run benign --config-dir=./config
```

### Via Command-Line Overrides

Override the backend at runtime:

```bash
# Switch from local to DES
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=des

# Switch from DES to local
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=local
```

## End-to-End Example

Here's a complete example of running SWE-bench with both backends:

### Local Docker

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
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  agent.config.model=azure:gpt-4o \
  'dataset.config.instance_ids=["django__django-11179"]'
```

### DES (Docker Execution Service)

```bash
# 1. Set up environment (no DOCKER_HOST needed)
cat > .env <<EOF
AZURE_OPENAI_ENDPOINT="https://your-endpoint.azure-api.net"
AZURE_OPENAI_API_KEY="your-key"
OPENAI_API_VERSION="2025-04-01-preview"
EOF

# 2. Verify DES access
which des_exec

# 3. Run with DES
with-proxy uv run --env-file .env prompt-siren run benign +dataset=swebench \
  sandbox_manager.config.docker_client=des \
  sandbox_manager.config.des_session_timeout=7200 \
  'sandbox_manager.config.des_docker_images=["vmvm-registry.fbinfra.net/debian:bookworm-slim"]' \
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

### DES Issues

**Problem**: `des_exec: command not found`
```bash
# Solution: Verify des_exec is in PATH
which des_exec

# If not found, check if it's installed
feature install des_exec  # Or appropriate installation command
```

**Problem**: `DES session timeout`
```bash
# Solution: Increase session timeout
sandbox_manager.config.des_session_timeout=7200  # 2 hours
```

**Problem**: `Docker image not found on DES VM`
```bash
# Solution: Pre-load images
sandbox_manager.config.des_docker_images='["vmvm-registry.fbinfra.net/debian:bookworm-slim","python:3.12"]'
```

## Plugin System

Both Local Docker and DES are implemented as plugins using the Docker client registry system. You can create custom Docker client plugins by:

1. Implementing the `AbstractDockerClient` protocol
2. Creating a factory function
3. Registering via entry points

See [custom_components.md](custom_components.md) for details on creating custom plugins.

## Performance Considerations

### Local Docker
- **Startup time**: Fast (~1-2 seconds per container)
- **Execution**: No network latency
- **Best for**: Development, testing, small datasets

### DES
- **Startup time**: Slower (~30-60 seconds for VM provisioning)
- **Execution**: Network latency for all operations
- **Best for**: Production, large-scale runs, no local Docker available

## See Also

- [Sandbox Manager Documentation](sandbox_manager.md) - General sandbox manager concepts
- [Configuration Guide](configuration.md) - Hydra configuration details
- [Custom Components](custom_components.md) - Creating custom plugins

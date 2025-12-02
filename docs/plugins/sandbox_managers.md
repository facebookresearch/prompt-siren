# Sandbox Manager System

Sandbox managers provide isolated container environments for task execution. Supports both single-container and multi-container tasks with networking.

## Core Concepts

**Two-Level Context Management**:
```python
from prompt_siren.sandbox_managers.sandbox_task_setup import TaskSetup, ContainerSetup, ContainerSpec
from prompt_siren.sandbox_managers.image_spec import PullImageSpec

# Define task setup with agent container
task_setup = TaskSetup(
    task_id="my-task",
    agent_container=ContainerSetup(
        name="agent",
        spec=ContainerSpec(image_spec=PullImageSpec(tag="python:3.12"))
    ),
    service_containers={}  # Optional services
)

async with manager.setup_batch([task_setup]):
    async with manager.setup_task(task_setup) as sandbox_state:
        await manager.exec(sandbox_state.agent_container_id, "python script.py")
    # Docker client closed automatically on exit
```

**Concurrency**: Multiple tasks can run in parallel within a batch. Implementations MUST be async-safe (use `asyncio.Lock` for shared state).

**Resource Management**: Docker client is created in `setup_batch()` and automatically closed when the batch context exits, preventing resource leaks.

## Image Specifications

Sandbox managers use `ImageSpec` to define how to prepare execution environments. Three types are supported:

**1. Pull Pre-Built Images** (`PullImageSpec`):
```python
from prompt_siren.sandbox_managers.image_spec import PullImageSpec

# Simple image pull
spec = PullImageSpec(tag="python:3.12")

# Or from Docker Hub
spec = PullImageSpec(tag="alpine:latest")
```

**2. Build from Dockerfile** (`BuildImageSpec`):
```python
from prompt_siren.sandbox_managers.image_spec import BuildImageSpec

# Simple build
spec = BuildImageSpec(
    context_path="./docker/my-env",
    tag="my-env:latest"
)

# With build args and custom Dockerfile
spec = BuildImageSpec(
    context_path="./docker/python-app",
    dockerfile_path="Dockerfile.dev",  # Relative to context_path
    tag="python-app:dev",
    build_args={"PYTHON_VERSION": "3.12", "ENV": "development"}
)
```

**3. Multi-Stage Build** (`MultiStageBuildImageSpec`):

Multi-stage builds enable caching at multiple levels, ideal for complex environments like SWE-bench where different stages have different cache lifetimes:

```python
from prompt_siren.sandbox_managers.image_spec import (
    MultiStageBuildImageSpec,
    BuildStage
)

# Three-stage build with progressive caching
spec = MultiStageBuildImageSpec(
    stages=[
        # Stage 1: Base OS and system dependencies (rarely changes)
        BuildStage(
            context_path="./cache/base",
            tag="myproject-base:latest",
            cache_key="base_v1"
        ),
        # Stage 2: Python environment (changes per dependency set)
        BuildStage(
            context_path="./cache/env/abc123",
            tag="myproject-env:abc123",
            parent_tag="myproject-base:latest",
            cache_key="env_abc123"
        ),
        # Stage 3: Application code (changes per instance)
        BuildStage(
            context_path="./cache/instance/xyz789",
            tag="myproject-instance:xyz789",
            parent_tag="myproject-env:abc123"
        ),
    ],
    final_tag="myproject-instance:xyz789"
)
```

**Multi-Stage Build Benefits**:
- **Layered caching**: Each stage caches independently (base → env → instance)
- **Incremental builds**: Only rebuild changed stages
- **Disk efficiency**: Share base/env images across multiple instances
- **Used by**: SWE-bench dataset for efficient repository builds

**Multi-Stage Build Behavior**:
- Stages are built sequentially in order
- Each stage's Dockerfile should use `FROM <parent_tag>`
- The `final_tag` is used as the container image
- All intermediate stage images remain available for caching

## Multi-Container Tasks

Tasks can use multiple containers with networking for complex scenarios (e.g., agent + database, agent + attack server):

```python
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    TaskSetup, ContainerSetup, ContainerSpec, NetworkConfig
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec

# Task with agent + database service
task_setup = TaskSetup(
    task_id="web-scraping-task",
    agent_container=ContainerSetup(
        name="agent",
        spec=ContainerSpec(
            image_spec=PullImageSpec(tag="python:3.12"),
            hostname="agent"
        )
    ),
    service_containers={
        "postgres": ContainerSetup(
            name="postgres",
            spec=ContainerSpec(
                image_spec=PullImageSpec(tag="postgres:16"),
                hostname="db",  # Agent can connect to "db" via DNS
                environment={"POSTGRES_PASSWORD": "secret"}
            )
        )
    },
    network_config=NetworkConfig(name="task-net", internal=False)
)

# Containers can communicate via hostnames
async with manager.setup_task(task_setup) as state:
    # Agent connects to database at "db:5432"
    await manager.exec(state.agent_container_id, "psql -h db -U postgres")
```

**Multi-Container Features**:
- **DNS Resolution**: Containers use hostnames for service discovery
- **Network Isolation**: Each task gets its own Docker network
- **Parallel Creation**: Service containers start concurrently
- **Cloning Support**: `clone_sandbox_state()` clones all containers + network

**Image Lifecycle**:
- **Pull specs**: Images are pulled during `setup_batch()` if not already present
- **Build specs**: Images are built during `setup_batch()` with progress logged to Logfire
- **Cleanup**: Built images are NOT automatically cleaned up (users manage via Docker CLI)
- **Caching**: Docker's layer caching applies to builds, speeding up repeated builds

## AbstractSandboxManager Protocol

```python
from pathlib import Path
from prompt_siren.sandbox_managers.abstract import AbstractSandboxManager
from prompt_siren.sandbox_managers.sandbox_task_setup import SandboxTaskSetup
from prompt_siren.sandbox_managers.sandbox_state import SandboxState

class AbstractSandboxManager(Protocol):
    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        """Prepare all images (pull/build) and shared resources for batch."""

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        """Create containers and network for task. MUST be async-safe for parallel execution."""

    async def exec(
        self, container_id: str, cmd: str | list[str],
        stdin: str | bytes | None = None, cwd: str | None = None,
        env: dict[str, str] | None = None, user: str | None = None,
        timeout: int | None = None, shell_path: Path | None = None
    ) -> ExecOutput:
        """Execute command in container."""

    async def clone_sandbox_state(self, source_state: SandboxState) -> SandboxState:
        """Clone all containers and network for state snapshots."""
```

## Docker Implementation

**Configuration**:
```python
from prompt_siren.sandbox_managers.docker import DockerSandboxConfig, create_docker_sandbox_manager

config = DockerSandboxConfig(
    network_enabled=False,  # Disable network (default)
    batch_id_prefix="workbench"
)
manager = create_docker_sandbox_manager(config)
```

**Key Features**:
- **Multi-container support**: Agent + services with DNS-based networking
- **Unique naming**: `{batch_id}-{execution_id}-{container_name}-{task_id}-{uuid}`
- **Default command**: Containers run `sleep infinity` to stay alive for exec commands
- **Network isolation**: `network_enabled=False` creates internal-only networks
- **Cloning**: Commits containers to temp images, creates independent copies with new network
- **Auto-cleanup**: Docker client created in `setup_batch()` and closed automatically on exit
- **Resource tracking**: TaskSandboxContext tracks all containers/networks via execution_id
- **Image caching**: Sequential image building prevents race conditions

## Creating Custom Sandbox Managers

**1. Define config and implement protocol**:
```python
from prompt_siren.sandbox_managers.abstract import AbstractSandboxManager
from prompt_siren.sandbox_managers.sandbox_task_setup import SandboxTaskSetup
from prompt_siren.sandbox_managers.sandbox_state import SandboxState

class MyCustomSandboxConfig(BaseModel):
    api_endpoint: str

class MyCustomSandboxManager:
    def __init__(self, config: MyCustomSandboxConfig):
        self._lock = asyncio.Lock()  # Required for async safety!
        self._contexts: dict[str, Any] = {}

    @asynccontextmanager
    async def setup_batch(self, task_setups: Sequence[SandboxTaskSetup]) -> AsyncIterator[None]:
        # Build all images
        yield

    @asynccontextmanager
    async def setup_task(self, task_setup: SandboxTaskSetup) -> AsyncIterator[SandboxState]:
        # Create containers and network
        yield SandboxState(...)

    async def exec(self, container_id: str, cmd: str | list[str], ...) -> ExecOutput:
        # Execute command
        pass

    async def clone_sandbox_state(self, source_state: SandboxState) -> SandboxState:
        # Clone all containers
        pass
```

**2. Register**:
```toml
# pyproject.toml entry point:
[project.entry-points."prompt_siren.sandbox_managers"]
my_custom = "my_package.sandbox:create_my_custom_sandbox"
```

## Best Practices

1. **Async Safety**: Use `asyncio.Lock` for all shared state modifications
2. **Unique Naming**: Include batch_id, execution_id, and UUID for all resources
3. **Graceful Cleanup**: Catch and log exceptions, never propagate cleanup errors
4. **Resource Isolation**: Cleanup affects only specified containers, not others
5. **Sequential Building**: Build images sequentially in `setup_batch()` to avoid races
6. **Parallel Creation**: Create containers in parallel during `setup_task()`

## Testing

**Unit tests** with mocks:
```python
@pytest.mark.anyio
async def test_parallel_creation(manager):
    # Test thread safety
    ids = await asyncio.gather(*[manager.create_task_container(...) for _ in range(3)])
    assert len(set(ids)) == 3  # All unique
```

**Integration tests** with real Docker (mark with `@pytest.mark.docker_integration`):
```bash
uv run pytest -m "not docker_integration"  # Skip (CI/CD default)
uv run pytest -m docker_integration        # Run with Docker
```

## Adding Custom Environments

**Important**: Environments are **not** registered as plugins. Instead, they are **owned and instantiated by datasets**. When creating a custom dataset, you specify which environment implementation to use.

This design allows:
- Multiple datasets to share the same environment type with different configurations
- Environment selection to be an implementation detail of the dataset
- Cleaner separation between data (tasks) and execution context (environment)

### When to Create a Custom Environment

Create a custom environment when:
- You need a new execution context (e.g., connecting to an external API, custom sandbox)
- You need different dependency types than existing environments
- You need custom rendering or injection detection logic

**Note**: If you're just adding new tasks, consider creating a custom dataset that reuses an existing environment (like `AgentDojoEnv` or `PlaywrightEnv`).

### Environment Implementation

#### Environment Architecture

Environments provide the **execution context** for tasks. They are responsible for:
- Rendering outputs with injected attacks
- Detecting injection points in outputs
- Managing execution resources (browsers, servers, etc.)
- Providing default values for injection vectors

**Environment state (env_state)**: The `EnvStateT` type parameter represents the environment state that are passed to tools via `RunContext[EnvStateT]`. These are the actual stateful objects your tools interact with (e.g., `TaskEnvironment` for AgentDojo, `Page` for Playwright).

**Note**: `env_state` corresponds to PydanticAI's `deps` (dependencies) concept - it's the runtime context that gets passed to your tools through `RunContext`. When you see `RunContext[EnvStateT]` in tool signatures, `EnvStateT` is the type of dependencies (`deps`) that PydanticAI will inject.

Environments use a **two-level context management pattern**:
- **Batch context** (`create_batch_context`) - Set up expensive resources once per batch (browsers, servers, etc.)
- **Task context** (`create_task_context`) - Create fresh dependencies for each task execution

This separation allows efficient resource sharing while maintaining isolation between tasks.

#### Snapshottable vs Non-Snapshottable Environments

The workbench distinguishes between two types of environments based on their ability to snapshot state:

**Snapshottable Environments** - Dependencies can be copied to create independent state snapshots:
- Use `SnapshottableAbstractEnvironment` protocol
- Implement `copy_env_state(env_state) -> env_state` to create deep copies
- Examples: Pydantic models, dataclasses, simple Python objects
- Benefit: Fast rollback operations (Mini-GOAT attack uses this)

**Non-Snapshottable Environments** - Dependencies are stateful resources that cannot be copied:
- Use `NonSnapshottableAbstractEnvironment` protocol
- Implement `reset_env_state(env_state) -> env_state` to reset to initial state
- Examples: Browser pages, database connections, live servers
- Behavior: On rollback, environment resets and replays tool history

#### Required Protocol Methods

All environments must implement the following methods:

1. **`get_injectable_ids(raw_output) -> list[InjectionVectorID]`** - Detect which injection vectors are present in the raw output
2. **`get_default_for_injection_vectors(injection_vector_ids) -> InjectionAttacksDict`** - Provide default/benign content for injection vectors
3. **`render(raw_output, attacks?) -> FinalOutputT`** - Render raw output with optional attack injections
4. **`create_batch_context(tasks) -> AsyncIterator[Self]`** - Set up batch-level resources
5. **`create_task_context(task) -> AsyncIterator[EnvStateT]`** - Create a fresh environment state for each task
6. **`copy_env_state(env_state) -> env_state`** (Snapshottable only) - Create deep copy for state snapshotting
7. **`reset_env_state(env_state) -> env_state`** (Non-Snapshottable only) - Reset env_state to initial state

Additionally, environments must define these instance attributes:
- **`all_injection_ids: list[InjectionVectorID]`** - All possible injection vector IDs
- **`name: str`** - Environment name identifier

#### State Isolation

**Critical**: Each execution state maintains its own independent environment state:
- When tools execute, env_state is copied BEFORE execution (for snapshottable envs)
- Previous states remain unchanged when later states modify env_state
- This enables attacks like Mini-GOAT to rollback and try alternative injections

#### Example: Snapshottable Environment

```python
# my_snapshottable_env.py
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing_extensions import Self
from prompt_siren.environments.abstract import SnapshottableAbstractEnvironment
from prompt_siren.tasks import TaskCouple, BenignTask, MaliciousTask
from prompt_siren.types import InjectionAttacksDict, InjectionVectorID, StrContentAttack
from pydantic import BaseModel

# Env state that can be deep-copied
class MyEnvState(BaseModel):
    state: dict[str, str]
    counter: int = 0

@dataclass
class MySnapshottableEnvironment(SnapshottableAbstractEnvironment[MyEnvState, str, str, StrContentAttack]):
    """Snapshottable environment with Pydantic env_state."""

    initial_env_state: MyEnvState
    all_injection_ids: list[InjectionVectorID]  # Required by protocol
    name: str  # Required by protocol

    async def copy_env_state(self, env_state: MyEnvState) -> MyEnvState:
        """Create deep copy for state snapshotting."""
        return env_state.model_copy(deep=True)

    async def get_injectable_ids(self, raw_output: str) -> list[InjectionVectorID]:
        """Find injection points in output."""
        return [id for id in self.all_injection_ids if f"{{{id}}}" in raw_output]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Returns default content for the given injection vectors."""
        return {vector_id: StrContentAttack(content="") for vector_id in injection_vector_ids}

    async def render(self, raw_output: str, attacks: InjectionAttacksDict[StrContentAttack] | None = None) -> str:
        """Render output with optional attacks."""
        if attacks is None:
            return raw_output
        result = raw_output
        for vector_id, attack in attacks.items():
            result = result.replace(f"{{{vector_id}}}", attack.content)
        return result

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[MyEnvState]]
            | Sequence[BenignTask[MyEnvState]]
            | Sequence[MaliciousTask[MyEnvState]]
            | Sequence[BenignTask[MyEnvState] | MaliciousTask[MyEnvState]]
        ),
    ) -> AsyncIterator[Self]:
        """No expensive resources needed for this example."""
        yield self

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[MyEnvState] | BenignTask[MyEnvState] | MaliciousTask[MyEnvState],
    ) -> AsyncIterator[MyEnvState]:
        """Create fresh env_state copy for each task.

        Yields:
            Fresh env_state for this task execution
        """
        env_state = self.initial_env_state.model_copy(deep=True)
        yield env_state
```

#### Example: Non-Snapshottable Environment

```python
# my_non_snapshottable_env.py
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing_extensions import Self
from playwright.async_api import Page, Browser
from prompt_siren.environments.abstract import NonSnapshottableAbstractEnvironment
from prompt_siren.tasks import TaskCouple, BenignTask, MaliciousTask
from prompt_siren.types import InjectionAttacksDict, InjectionVectorID, StrContentAttack

@dataclass
class MyNonSnapshottableEnvironment(NonSnapshottableAbstractEnvironment[Page, Page, str, StrContentAttack]):
    """Non-snapshottable environment using browser pages."""

    browser: Browser
    server_port: int
    all_injection_ids: list[InjectionVectorID]  # Required by protocol
    name: str  # Required by protocol

    async def reset_env_state(self, env_state: Page) -> Page:
        """Reset browser page to initial state."""
        # Navigate back to initial URL and clear state
        await env_state.goto(f"http://localhost:{self.server_port}/index.html")
        await env_state.context.clear_cookies()
        return env_state

    async def get_injectable_ids(self, raw_output: Page) -> list[InjectionVectorID]:
        """Find injection points in HTML output."""
        # Example: check for elements with IDs matching injection vectors
        return [
            injection_id
            for injection_id in self.all_injection_ids
            if await raw_output.query_selector(f"#{injection_id}")
        ]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Returns default content for the given injection vectors."""
        return {vector_id: StrContentAttack(content="") for vector_id in injection_vector_ids}

    async def render(
        self, raw_output: Page, attacks: InjectionAttacksDict[StrContentAttack] | None = None
    ) -> str:
        """Render output with optional attacks."""
        if attacks is not None:
            # Inject attacks into page elements
            await raw_output.evaluate(
                """
                (attack) => {
                    for (const [id, value] of Object.entries(attack)) {
                        const el = document.getElementById(id);
                        if (el) {
                            el.textContent = value;
                        }
                    }
                }
                """,
                {k: v.content for k, v in attacks.items()},
            )
        return await raw_output.content()

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[Page]]
            | Sequence[BenignTask[Page]]
            | Sequence[MaliciousTask[Page]]
            | Sequence[BenignTask[Page] | MaliciousTask[Page]]
        ),
    ) -> AsyncIterator[Self]:
        """Start browser once for all tasks."""
        browser = await self.playwright.chromium.launch()
        try:
            yield self
        finally:
            await browser.close()

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[Page] | BenignTask[Page] | MaliciousTask[Page],
    ) -> AsyncIterator[Page]:
        """Create fresh page for each task.

        Yields:
            Fresh browser page for this task execution
        """
        page = await self.browser.new_page()
        try:
            yield page
        finally:
            await page.close()
```

## Adding Custom Agents

Custom agents in prompt-siren implement a finite state machine design that transitions through different execution states while interacting with a model and executing tools. This makes it easier to track execution progress, intercept injectable content, and apply attacks at precise moments.

**Reference Implementation**: See `src/prompt_siren/agents/plain.py` for a complete working example of the PlainAgent implementation.

### 1. Create Agent Configuration

```python
# my_agent_config.py
from pydantic import BaseModel, Field

class MyCustomAgentConfig(BaseModel):
    """Configuration for my custom agent."""

    model: str = Field(description="Model to use")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    custom_parameter: str = Field(default="default", description="Custom agent parameter")
```

### 2. Implement Agent Class

```python
# my_agent.py
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from prompt_siren.agents.abstract import AbstractAgent
from prompt_siren.agents.states import ExecutionState
from prompt_siren.environments.abstract import AbstractEnvironment
from prompt_siren.types import InjectableUserContent, InjectionAttacksDict


@dataclass(frozen=True)
class MyCustomAgent(AbstractAgent):
    """Custom agent implementation.

    Note: The dataclass should be frozen to ensure immutability.
    The config must be stored in a private attribute (_config) and exposed
    via a property to maintain compatibility with Python's type system.
    """

    _config: MyCustomAgentConfig
    agent_type: ClassVar[str] = "my_custom"

    @property
    def config(self) -> MyCustomAgentConfig:
        """Returns the config of the agent.

        It has to be a property method and not an attribute as otherwise
        Python's type system breaks.
        """
        return self._config

    def get_agent_name(self) -> str:
        """Get a descriptive name for this agent (used for filenames and logging).

        Example: 'my_custom:gpt-5'
        """
        return f"my_custom:{self.config.model}"

    # Implement required abstract methods
    async def run(
        self,
        environment: AbstractEnvironment,
        env_state,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> RunContext:
        """Execute the agent until completion.

        This is the main entry point for agent execution.
        """
        # Your implementation here
        pass

    async def iter(
        self,
        environment: AbstractEnvironment,
        env_state,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState]:
        """Iterate through agent execution states.

        Yields each execution state as the state machine progresses.
        """
        # Your implementation here
        pass
        yield  # Needed to make type checker happy

    async def next_state(
        self,
        *,
        current_state: ExecutionState,
        toolsets: Sequence[AbstractToolset],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> ExecutionState:
        """Execute a single state transition in the state machine.

        This is the core state machine transition function.
        """
        # Your implementation here
        pass

    async def prev_state(
        self,
        *,
        current_state: ExecutionState,
        toolsets: Sequence[AbstractToolset],
    ) -> ExecutionState:
        """Roll back a step in the state machine.

        Returns the previous state with correct environment state.
        """
        # Your implementation here
        pass

    async def resume_iter_from_state(
        self,
        *,
        current_state: ExecutionState,
        toolsets: Sequence[AbstractToolset],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState]:
        """Resume execution from a previously saved state.

        Useful for implementing features like attack injection at precise points.
        """
        # Your implementation here
        pass
        yield  # Needed to make type checker happy

    def create_initial_request_state(
        self,
        environment: AbstractEnvironment,
        env_state,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        usage: RunUsage | None = None,
    ):
        """Create the initial state for agent execution.

        Returns either ModelRequestState or InjectableModelRequestState
        depending on whether the user prompt contains injectable content.
        """
        # Your implementation here
        pass
```

### 3. Create Factory Function

```python
# my_agent_factory.py
def create_my_custom_agent(config: MyCustomAgentConfig, context: None = None) -> MyCustomAgent:
    """Factory function to create custom agent.

    Args:
        config: Agent configuration
        context: Optional context parameter (unused by agents, for registry compatibility)

    Returns:
        Configured agent instance
    """
    return MyCustomAgent(_config=config)
```

### 4. Register Agent

```toml
# Register via entry point (recommended)
# In (your or prompt_siren's) pyproject.toml:
[project.entry-points."prompt_siren.agents"]
my_custom = "my_package.my_agent:create_my_custom_agent"
```

```python
# Or register programmatically
# (you need to make sure that this call is executed before you try to instantiate the agent)
from prompt_siren.agents.registry import register_agent
from my_package.my_agent import create_my_custom_agent
from my_package.my_agent_config import MyCustomAgentConfig

register_agent("my_custom", MyCustomAgentConfig, create_my_custom_agent)
```

### 5. Create Configuration Files

```yaml
# my_custom_agent.yaml
defaults:
  # @package _global_
  - _self_

name: "my_custom_agent_experiment"

# Use custom agent
agent:
  type: my_custom
  config:
    model: azure:gpt-5
    temperature: 0.8
    max_tokens: 1500
    custom_parameter: "custom_value"

# Include dataset and attack via defaults
defaults:
  - _self_
  - dataset: agentdojo-workspace
  - attack: agentdojo

# Other default settings...
execution:
  concurrency: 1
task_ids: null  # null = all tasks for the execution mode
output:
  trace_dir: "traces"
telemetry:
  trace_console: true
usage_limits: null
```

### 6. Use Custom Agent

```bash
# Use in experiments with config file
uv run prompt-siren run benign --config-dir=. --config-name=my_custom_agent

# Use in attack mode
uv run prompt-siren run attack --config-dir=. --config-name=my_custom_agent

# Override agent type and parameters via command line
uv run prompt-siren run benign +dataset=agentdojo-workspace agent.type=my_custom agent.config.temperature=0.5

# Parameter sweep with custom agent
uv run prompt-siren run benign --multirun +dataset=agentdojo-workspace agent.type=my_custom agent.config.temperature=0.5,0.8,1.0
```

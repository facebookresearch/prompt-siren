## Adding Custom Providers

Providers handle the creation of model instances for specific provider prefixes (e.g., `bedrock:`, `llama:`). They allow you to add support for new model providers that aren't built into pydantic-ai.

**Key Characteristics of Providers:**
- **No configuration class**: Unlike other components, providers don't use Pydantic config classes.
- **Environment-based configuration**: Providers read their settings from environment variables (following pydantic-ai conventions)
- **Prefix-based routing**: Each provider handles all models with a specific prefix (e.g., "bedrock:" → BedrockProvider)
- **Transparent integration**: Once registered, providers work seamlessly with the existing agent configuration

### 1. Implement Provider Class

Providers must implement the `AbstractProvider` protocol with a single method: `create_model(model_string: str) -> Model`.

```python
# my_provider.py
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from openai import AsyncOpenAI
import os

class MyCustomProvider:
    """Provider for MyCustom API.

    Configuration is read from the MY_CUSTOM_API_KEY environment variable.
    """

    def __init__(self):
        """Initialize the provider.

        Read configuration from environment variables here.
        This follows the same pattern as pydantic-ai's built-in providers
        (e.g., OpenAI reads OPENAI_API_KEY, Anthropic reads ANTHROPIC_API_KEY).
        """
        api_key = os.getenv("MY_CUSTOM_API_KEY")
        if not api_key:
            raise ValueError(
                "MY_CUSTOM_API_KEY environment variable must be set to use MyCustomProvider"
            )

        # Initialize your client
        self._client = AsyncOpenAI(
            base_url="https://api.mycustom.com/v1/",
            api_key=api_key,
        )

    def create_model(self, model_string: str) -> Model:
        """Create a model instance for the given model string.

        Args:
            model_string: Full model string (e.g., "mycustom:my-model-v1")
                         This includes the provider prefix.

        Returns:
            Model instance ready to use with pydantic-ai

        Raises:
            ValueError: If the model string is not valid for this provider
        """
        # Extract model name from the full string
        # model_string format: "mycustom:my-model-v1"
        if ":" not in model_string:
            raise ValueError(f"Invalid model string format: {model_string}")

        prefix, model_name = model_string.split(":", 1)

        if prefix != "mycustom":
            raise ValueError(
                f"MyCustomProvider only handles 'mycustom:' prefix, got '{prefix}:'"
            )

        # Create and return the appropriate Model instance
        # This example uses OpenAIChatModel, but you can use any Model type
        return OpenAIChatModel(model_name, provider=self._client)
```

### 2. Provider Registration (Optional Factory Function)

Providers can be registered directly via their class or through a factory function, as long as they're callable with **no parameters** (unlike other components that receive config).

**Option A: Use the class directly (if constructor takes no required parameters)**
```python
# my_provider.py - no factory function needed!
# The MyCustomProvider class above can be used directly since __init__ takes no parameters
```

**Option B: Use an explicit factory function (if you need custom initialization logic)**
```python
# my_provider.py (continued)

def create_my_custom_provider() -> MyCustomProvider:
    """Factory function for creating a MyCustom provider.

    Use this pattern if you need custom initialization logic beyond
    what the constructor provides.

    Returns:
        Configured MyCustomProvider instance
    """
    return MyCustomProvider()
```

**Note:** Providers don't use Pydantic config classes - they read configuration from environment variables instead.

### 3. Register Provider

Register via entry point (recommended) or programmatically. The entry point can reference either the class directly or a factory function.

```toml
# In pyproject.toml - Option A: Reference the class directly
[project.entry-points."prompt_siren.providers"]
mycustom = "my_package.my_provider:MyCustomProvider"

# Or Option B: Reference a factory function
mycustom = "my_package.my_provider:create_my_custom_provider"
```

```python
# Or register programmatically (if needed for testing)
from prompt_siren.providers import register_provider

# Option A: Register the class directly
register_provider("mycustom", MyCustomProvider)

# Option B: Register via factory function
register_provider("mycustom", create_my_custom_provider)
```

### 4. Use Custom Provider

Once registered, models with your prefix are automatically routed to your provider:

```yaml
# config.yaml
agent:
  type: plain
  config:
    model: mycustom:my-model-v1  # Will use MyCustomProvider
    temperature: 0.7
```

```bash
# Set environment variable
export MY_CUSTOM_API_KEY=your-api-key-here

# Run experiment
uv run prompt-siren run benign +dataset=agentdojo-workspace agent.config.model=mycustom:my-model-v1
```

### Provider Design Guidelines

**Configuration Management:**
- Read all configuration from environment variables
- Follow established conventions (e.g., `<PROVIDER>_API_KEY`)
- Provide clear error messages when environment variables are missing
- Document required environment variables in docstrings

**Error Handling:**
- Validate the model string format in `create_model`
- Raise `ValueError` with descriptive messages for invalid inputs
- Handle missing API keys gracefully with helpful error messages

**Model Creation:**
- Return pydantic-ai `Model` instances
- You can wrap existing providers or create custom implementations
- Ensure the returned model is compatible with pydantic-ai's Agent interface

**Testing:**
- Test provider registration and discovery
- Test model creation with valid and invalid model strings
- Mock API calls to avoid requiring actual API keys in tests

# Usage Limits Configuration

Siren supports configuring usage limits using PydanticAI's `UsageLimits` to control resource consumption during experiments.

## Configuration

```yaml
# No limits (default)
usage_limits: null

# With limits
usage_limits:
  request_limit: 10
  total_tokens_limit: 50000
  count_tokens_before_request: true
```

## Command Line Usage

```bash
# Set limits
uv run prompt-siren usage_limits.request_limit=10 usage_limits.total_tokens_limit=50000

# Disable limits
uv run prompt-siren usage_limits=null
```

## Behavior

When limits are exceeded, PydanticAI raises `UsageLimitExceeded`, stopping the current task and continuing with remaining tasks.

For more details on what parameters can be set, see [PydanticAI Usage Documentation](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits).

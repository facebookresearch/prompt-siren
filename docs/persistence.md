# Results saving System

Siren includes a comprehensive persistence layer for saving task execution results, generated attacks, and conversation logs. This system is designed for reproducibility, organization, and concurrent execution safety.

## Overview

The results saving automatically saves:
- **Task execution results** - Benign and attack evaluation scores
- **Conversation logs** - Complete message history and model responses
- **Generated attacks** - Attack vectors produced during execution
- **Configuration snapshots** - Hydra-compatible YAML configs for reproducibility
- **Telemetry metadata** - Trace IDs, span IDs, token usage

## Directory Structure

Results are organized by configuration components:

```
outputs/
├── index.jsonl                           # Global index of all executions
├── index.jsonl.lock                      # File lock for concurrent writes
└── {dataset_type}/
    └── {agent_type}/
        └── {attack_type}/
            └── {config_hash}/
                ├── config.yaml           # Configuration snapshot
                ├── {timestamp}_{task_id}.json
                ├── {timestamp}_{task_id}.json
                └── ...
```

### Example Directory

```
outputs/
├── index.jsonl
└── agentdojo/
    └── plain/
        ├── gcg/
        │   └── a1b2c3d4/
        │       ├── config.yaml
        │       ├── 20251014_153022_user_0.json
        │       └── 20251014_153045_user_1.json
        └── benign/
            └── 576f167c/
                ├── config.yaml
                └── 20251014_154012_user_0.json
```

## Configuration Hash

Each unique configuration receives an 8-character hash derived from:
- Dataset type and configuration
- Agent type and configuration
- Attack type and configuration (or `None` for benign runs)

The hash ensures:
- **Deterministic identification** - Same configs always produce the same hash
- **Collision avoidance** - Different configs produce different hashes
- **Reproducibility** - Easy to find results for specific configurations

### Hash Computation

```python
from prompt_siren.run_persistence import compute_config_hash

config_hash = compute_config_hash(
    dataset_config=DatasetConfig(type="agentdojo", config={...}),
    agent_config=AgentConfig(type="plain", config={...}),
    attack_config=AttackConfig(type="gcg", config={...}),  # or None
)
```

## File Formats

### config.yaml

Hydra-compatible configuration file created once per unique config:

```yaml
# Config hash: a1b2c3d4
# Created: 2025-10-14T15:30:22.123456Z

dataset:
  type: agentdojo
  config:
    suite_name: workspace
    version: v1.2.2

agent:
  type: plain
  config:
    model: claude-3-5-sonnet
    temperature: 0.0

attack:
  type: gcg
  config:
    learning_rate: 0.1
    epochs: 100
```

### Execution Files

Individual task results saved as JSON (one file per task execution):

```json
{
  "execution_id": "abc12345",
  "task_id": "user:task_123",
  "dataset_type": "agentdojo",
  "dataset_config": {
    "suite_name": "workspace",
    "version": "v1.2.2"
  },
  "agent": "plain_agent",
  "config_hash": "a1b2c3d4",
  "timestamp": "2025-10-14T15:30:22.123456Z",
  "trace_id": "0000000000000000000000075bcd15",
  "span_id": "000000003ade68b1",
  "messages": [
    ...
  ],
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "requests": 1,
    "tool_calls": 2
  },
  "results": {
    "benign_score": 1.0,
    "attack_score": 0.5
  },
  "attacks": {
    "vector1": {
      "type": "str_content",
      "content": "malicious payload here"
    }
  }
}
```

### index.jsonl

Global index with one JSON entry per line for fast querying:

```jsonl
{"execution_id":"abc12345","task_id":"user:0","timestamp":"2025-10-14T15:30:22Z","dataset":"agentdojo","dataset_config":{"suite_name":"workspace","version":"v1.2.2"},"agent_type":"plain","agent_name":"plain_agent","attack_type":"gcg","config_hash":"a1b2c3d4","benign_score":1.0,"attack_score":0.5,"path":"outputs/agentdojo/plain/gcg/a1b2c3d4/20251014_153022_user_0.json"}
{"execution_id":"def67890","task_id":"user:1","timestamp":"2025-10-14T15:30:45Z","dataset":"agentdojo","dataset_config":{"suite_name":"workspace","version":"v1.2.2"},"agent_type":"plain","agent_name":"plain_agent","attack_type":"gcg","config_hash":"a1b2c3d4","benign_score":1.0,"attack_score":1.0,"path":"outputs/agentdojo/plain/gcg/a1b2c3d4/20251014_153045_user_1.json"}
```

## Usage

### Basic Usage with run_tasks

```python
from pathlib import Path
from prompt_siren.run import run_tasks
from prompt_siren.run_persistence import ExecutionPersistence

# Create persistence instance
persistence = ExecutionPersistence.create(
    base_dir=Path("outputs"),
    dataset_config=dataset_config,
    agent_config=agent_config,
    attack_config=attack_config,  # or None for benign runs
)

# Run tasks with persistence
results = await run_tasks(
    tasks=task_list,
    agent=agent,
    toolsets=toolsets,
    env=environment,
    persistence=persistence,  # Enable persistence
    max_concurrency=10,
)
```

## Concurrent Safety

The persistence system is designed for concurrent and multi-process execution.

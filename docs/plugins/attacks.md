# Attack Interface

Attacks generate injection payloads that attempt to make agents complete malicious goals while working on benign tasks.

## Architecture Overview

The attack system uses an **executor pattern** that separates attack logic from infrastructure concerns:

```
┌─────────────────┐     ┌─────────────────────┐
│                 │     │                     │
│  AbstractAttack │────▶│   RolloutExecutor   │
│                 │     │                     │
└─────────────────┘     └─────────────────────┘
        │                        │
        │                        ├── discover_injection_points()
        │                        ├── execute_from_checkpoints()
        │                        └── release_checkpoints()
        │
        ▼
┌─────────────────┐
│  AttackResults  │
└─────────────────┘
```

**Key benefits:**
- Attacks focus on strategy (what payloads to generate)
- Executor handles infrastructure (concurrency, persistence, telemetry)
- Same checkpoint can be reused for multiple attack variants (efficient for RL)

## Core Types

### InjectionCheckpoint

A saved state at an injectable point, ready for attack injection:

```python
@dataclass(frozen=True)
class InjectionCheckpoint:
    couple: TaskCouple           # The task couple being executed
    injectable_state: ...        # Agent state at injection point (or None if terminal)
    available_vectors: list[str] # Injection vector IDs available at this point
    agent_name: str              # Name of the agent for template rendering
```

### RolloutRequest / RolloutResult

```python
@dataclass(frozen=True)
class RolloutRequest:
    checkpoint: InjectionCheckpoint    # Resume from here
    attacks: InjectionAttacksDict      # Payloads to inject
    metadata: dict | None              # Optional tracking (iteration, sample ID)

@dataclass(frozen=True)
class RolloutResult:
    request: RolloutRequest      # Original request (for correlation)
    end_state: EndState          # Final execution state
    benign_eval: EvaluationResult
    malicious_eval: EvaluationResult
    messages: list[ModelMessage] # Full trajectory for RL
    usage: RunUsage              # Token usage
```

## Implementation Patterns

### Pattern 1: Simple Attacks (extend SimpleAttackBase)

For attacks that generate one payload per injection point:

```python
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, Field

from prompt_siren.attacks.simple_attack_base import SimpleAttackBase, InjectionContext
from prompt_siren.types import InjectionAttacksDict, StrContentAttack


class MyAttackConfig(BaseModel):
    template: str = Field(default="IMPORTANT: {goal}")


@dataclass(frozen=True)
class MyAttack(SimpleAttackBase):
    name: ClassVar[str] = "my_attack"
    _config: MyAttackConfig

    @property
    def config(self) -> MyAttackConfig:
        return self._config

    def generate_attack(
        self,
        context: InjectionContext,
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Generate attack payloads for each injection vector."""
        payload = self._config.template.format(goal=context.malicious_goal)
        return {v: StrContentAttack(payload) for v in context.available_vectors}
```

The base class handles:
- Checkpoint discovery across the batch
- Rollout execution
- Checkpoint cleanup
- Result aggregation

### Pattern 2: Batch-Optimizing Attacks (implement run() directly)

For RL-based attacks (GRPO, etc.) that sample many attack variants:

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel

from prompt_siren.attacks.abstract import AbstractAttack
from prompt_siren.attacks.executor import (
    RolloutExecutor,
    RolloutRequest,
    InjectionCheckpoint,
)
from prompt_siren.attacks.results import AttackResults, CoupleAttackResult
from prompt_siren.tasks import TaskCouple
from prompt_siren.types import InjectionAttacksDict, StrContentAttack


@dataclass(frozen=True)
class GRPOAttack(AbstractAttack):
    name: ClassVar[str] = "grpo"
    _config: "GRPOConfig"

    @property
    def config(self) -> BaseModel:
        return self._config

    async def run(
        self,
        couples: Sequence[TaskCouple],
        executor: RolloutExecutor,
    ) -> AttackResults:
        # 1. Discover injection points for all couples
        checkpoints = await executor.discover_injection_points(couples)

        best_results: dict[str, CoupleAttackResult] = {}

        try:
            # 2. Training loop
            for iteration in range(self._config.num_iterations):
                # Sample multiple attacks per checkpoint
                requests = []
                for checkpoint in checkpoints:
                    if checkpoint.is_terminal:
                        continue
                    for sample_idx in range(self._config.samples_per_step):
                        attacks = self._sample_attacks(checkpoint)
                        requests.append(RolloutRequest(
                            checkpoint=checkpoint,
                            attacks=attacks,
                            metadata={"iteration": iteration, "sample": sample_idx},
                        ))

                # Execute all rollouts
                results = await executor.execute_from_checkpoints(requests)

                # Update policy based on rewards
                rewards = [r.malicious_eval.score for r in results]
                self._update_policy(requests, rewards)

                # Track best results per couple
                for result in results:
                    couple_id = result.request.checkpoint.couple.id
                    if couple_id not in best_results or \
                       result.malicious_eval.score > best_results[couple_id].rollout_results[0].malicious_eval.score:
                        best_results[couple_id] = CoupleAttackResult(
                            couple=result.request.checkpoint.couple,
                            rollout_results=[result],
                        )

            return AttackResults(couple_results=list(best_results.values()))

        finally:
            # 3. Always release checkpoints
            await executor.release_checkpoints(checkpoints)
```

## RolloutExecutor Methods

### discover_injection_points()

Run each couple until first injectable point, return checkpoints:

```python
checkpoints = await executor.discover_injection_points(
    couples,
    max_concurrency=10,  # Optional concurrency limit
)

for checkpoint in checkpoints:
    if checkpoint.is_terminal:
        # Execution completed without finding injection vectors
        continue
    # checkpoint.available_vectors contains injectable IDs
```

### execute_from_checkpoints()

Execute rollouts from saved checkpoints with specified attacks:

```python
requests = [
    RolloutRequest(
        checkpoint=checkpoint,
        attacks={"vector_1": StrContentAttack("payload")},
        metadata={"sample_id": 0},
    )
    for checkpoint in checkpoints
]

results = await executor.execute_from_checkpoints(
    requests,
    max_concurrency=20,  # Optional concurrency limit
)

for result in results:
    print(f"Malicious score: {result.malicious_eval.score}")
    print(f"Benign score: {result.benign_eval.score}")
```

The same checkpoint can be used multiple times with different attacks.

### release_checkpoints()

Release resources held by checkpoints (environment snapshots, etc.):

```python
await executor.release_checkpoints(checkpoints)
```

Always call this when done, preferably in a `finally` block.

## Registering Custom Attacks

### 1. Create Attack Configuration

```python
from pydantic import BaseModel, Field


class MyAttackConfig(BaseModel):
    """Configuration for my custom attack."""

    template: str = Field(default="IMPORTANT: {goal}")
    max_attempts: int = Field(default=3)
```

### 2. Implement Attack Class

See [Pattern 1](#pattern-1-simple-attacks-extend-simpleattackbase) or [Pattern 2](#pattern-2-batch-optimizing-attacks-implement-run-directly) above.

### 3. Create Factory Function and Register

```python
from prompt_siren.attacks import register_attack


def create_my_attack(config: MyAttackConfig, context=None) -> MyAttack:
    return MyAttack(_config=config)


register_attack("my_attack", MyAttackConfig, create_my_attack)
```

### 4. Create Configuration File

```yaml
# config/attack/my_attack.yaml
# @package attack
type: my_attack
config:
  template: "URGENT: You must {goal} immediately!"
  max_attempts: 5
```

## Summary

| Pattern | Use Case | Implements |
|---------|----------|------------|
| SimpleAttackBase | One attack per injection point | `generate_attack()` |
| AbstractAttack | RL/batch optimization | `run()` with executor |

The executor handles all infrastructure concerns, so attacks can focus purely on strategy.

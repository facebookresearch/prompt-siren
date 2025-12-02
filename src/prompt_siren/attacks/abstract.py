# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc
from collections.abc import Sequence
from typing import ClassVar, Protocol, TypeVar

from pydantic import BaseModel
from pydantic_ai import InstrumentationSettings
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from ..agents.abstract import AbstractAgent
from ..agents.states import EndState
from ..environments.abstract import AbstractEnvironment
from ..tasks import BenignTask, MaliciousTask
from ..types import InjectionAttack, InjectionAttacksDict

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class AbstractAttack(Protocol[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    name: ClassVar[str]

    @property
    def config(self) -> BaseModel:
        """Returns the config of the attack.

        It has to be a property method and not an attribute as otherwise Python's type system breaks.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def attack(
        self,
        agent: AbstractAgent,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        message_history: Sequence[ModelMessage],
        env_state: EnvStateT,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        # TODO: make this take TaskCouple as input instead
        benign_task: BenignTask[EnvStateT],
        malicious_task: MaliciousTask[EnvStateT],
        usage_limits: UsageLimits,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> tuple[
        EndState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        InjectionAttacksDict[InjectionAttackT],
    ]:
        """Execute the attack to generate injection payloads for the malicious task.

        This method runs the attack strategy to produce injection payloads that will
        be injected into the agent's execution to attempt to make it complete the
        malicious goal while ostensibly working on the benign task.

        Args:
            agent: The agent that will execute the tasks
            environment: Environment for rendering and injection detection
            message_history: Previous conversation history
            env_state: Current environment state
            toolsets: Available tools that the agent can use
            benign_task: The benign task that serves as cover
            malicious_task: The malicious goal to inject
            usage_limits: Constraints on model usage
            instrument: Optional instrumentation settings for logging

        Returns:
            A tuple of (end_state, attacks) where:
                - end_state: The final execution state after the attack
                - attacks: Dictionary mapping injection vector IDs to attack payloads
        """
        raise NotImplementedError()

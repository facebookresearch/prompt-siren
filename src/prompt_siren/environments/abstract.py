# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable, TypeAlias, TypeVar

from typing_extensions import Self

from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict, InjectionVectorID

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT", contravariant=True)
FinalOutputT = TypeVar("FinalOutputT", covariant=True)
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@runtime_checkable
class Snapshottable(Protocol[EnvStateT]):
    """Protocol for environments that support env_state snapshotting.

    Snapshottable environments create independent copies of env_state at each state
    transition, allowing historical states to maintain their original environment.
    This enables efficient rollback without re-executing tools.
    """

    @abstractmethod
    async def copy_env_state(self, env_state: EnvStateT) -> EnvStateT:
        """Create an independent copy of env_state for state snapshotting.

        This method is called before tool execution to preserve the current state.
        The returned copy should be completely independent from the original -
        modifications to one should not affect the other.

        Note: env_state corresponds to PydanticAI's 'deps' - the runtime context passed to tools.
        This method creates a snapshot of the environment state so each execution state maintains
        its own independent copy.

        Args:
            env_state: The state to copy

        Returns:
            A deep copy of env_state that can be modified independently

        Example:
            For Pydantic models: `return env_state.model_copy(deep=True)`
            For Docker containers: clone via commit + run
        """
        ...


@runtime_checkable
class NonSnapshottable(Protocol[EnvStateT]):
    """Protocol for environments that use tool replay instead of snapshotting.

    Non-snapshottable environments (e.g., live browsers, external services) cannot
    easily create independent copies. Instead, they reset to a clean state and
    replay tools from message history when rolling back.
    """

    @abstractmethod
    async def reset_env_state(self, env_state: EnvStateT) -> EnvStateT:
        """Reset env_state to initial state before replaying tools.

        Called before run_tool_history in prev_state to ensure tools replay on a
        clean state. This should return a fresh environment state in their initial state.

        Note: env_state corresponds to PydanticAI's 'deps' - the runtime context passed to tools.
        This method resets the environment state to its initial state so tool history can be replayed.

        Args:
            env_state: The state to reset

        Returns:
            Fresh environment state (may be same object after reset,
            or a new object - implementation dependent)

        Example:
            For browsers: navigate to initial URL, clear cookies, return same page
            For external services: reset state via API calls, return new connection
        """
        ...


class _AbstractEnvironment(Protocol[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Base protocol with common methods for all environments.

    Do not use this directly - use AbstractEnvironment instead, which is a union
    of SnapshottableAbstractEnvironment and NonSnapshottableAbstractEnvironment.
    """

    all_injection_ids: list[InjectionVectorID]
    name: str

    @abstractmethod
    async def get_injectable_ids(self, raw_output: RawOutputT) -> list[InjectionVectorID]:
        """Given a list of injection IDs, returns the list of these IDs that
        can be injected in the `raw_output` passed.

        Args:
            raw_output: the raw tool call output to look for placeholders into.
            injection_ids: the injction vector ids to look for in the raw output.

        Returns:
            The list of vector ids found in the raw output.

        """
        raise NotImplementedError()

    @abstractmethod
    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[InjectionAttackT]:
        """Returns the default content for the given injection vector.

        Args:
            injection_vector_id: the ID of the injection vector.

        Returns:
            The default value for the given injection vector.
        """
        ...

    @abstractmethod
    async def render(
        self,
        raw_output: RawOutputT,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
    ) -> FinalOutputT:
        """Renders the raw_output into a final output by replacing the injection placeholders
        with the given attack.

        "hello, {injectin_id_1}."

        If no attack is passed, the method is also resposible for replacing the placeholders.


        Args:
            raw_output: the output to render.
            attack: the attacks to render inside the raw output.  If no attack is provided,
            then the output is rendered without attack.

        Returns:
            The rendered raw_output with attacks in place.

        """
        raise NotImplementedError()

    @asynccontextmanager
    @abstractmethod
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[EnvStateT]]
            | Sequence[BenignTask[EnvStateT]]
            | Sequence[MaliciousTask[EnvStateT]]
            | Sequence[BenignTask[EnvStateT] | MaliciousTask[EnvStateT]]
        ),
    ) -> AsyncIterator[Self]:
        """Creates batch-level context for expensive resource setup (browsers, servers, etc.).

        This context manager should set up long-lived resources that are shared across
        multiple task executions within a batch.

        Args:
            tasks: The list of tasks to be executed in this batch. Can be task couples
                (for attack mode) or individual benign/malicious tasks (for benign mode).
        """
        raise NotImplementedError()
        yield  # needed to make type checker happy about the @asynccontextmanager decorator

    @asynccontextmanager
    @abstractmethod
    async def create_task_context(
        self,
        task: TaskCouple[EnvStateT] | BenignTask[EnvStateT] | MaliciousTask[EnvStateT],
    ) -> AsyncIterator[EnvStateT]:
        """Creates per-task execution context with a fresh environment state.

        This context manager should create a fresh environment state for each individual task execution.
        Returns only the env_state - the environment is already available to the caller.

        Note: env_state corresponds to PydanticAI's 'deps' - the runtime context passed to tools
        via RunContext[EnvStateT]. These are the actual stateful objects your tools interact with.

        Args:
            task: The task being executed (TaskCouple for attack mode, BenignTask/MaliciousTask for benign mode)

        Yields:
            Fresh environment state (env_state/deps) for this task execution
        """
        raise NotImplementedError()
        yield  # needed to make type checker happy about the @asynccontextmanager decorator


class SnapshottableAbstractEnvironment(
    _AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    Snapshottable[EnvStateT],
    Protocol,
):
    """Environment that supports state snapshotting via copy_env_state.

    Use this for environments where env_state can be efficiently copied (e.g., Pydantic models,
    in-memory state). Snapshottable environments allow fast rollback without re-executing tools.
    """


class NonSnapshottableAbstractEnvironment(
    _AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    NonSnapshottable[EnvStateT],
    Protocol,
):
    """Environment that uses tool replay instead of snapshotting.

    Use this for environments where env_state cannot be easily copied (e.g., live browser pages,
    external services). Non-snapshottable environments reset state and replay tools on rollback.
    """


AbstractEnvironment: TypeAlias = (
    SnapshottableAbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    | NonSnapshottableAbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
)

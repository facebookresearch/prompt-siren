# Copyright (c) Meta Platforms, Inc. and affiliates.
import dataclasses
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Generic, Protocol, runtime_checkable, TypeVar

from typing_extensions import assert_never, Self

from ..sandbox_managers.abstract import AbstractSandboxManager
from ..sandbox_managers.image_spec import PullImageSpec
from ..sandbox_managers.sandbox_state import ContainerID, SandboxState
from ..sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    SandboxTaskSetup,
    TaskSetup,
)
from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttacksDict, InjectionVectorID, StrContentAttack
from .abstract import SnapshottableAbstractEnvironment


@dataclass(frozen=True)
class BashEnvState:
    """Dependencies for bash environment."""

    sandbox_state: SandboxState
    sandbox_manager: AbstractSandboxManager

    @property
    def agent_container_id(self) -> ContainerID:
        """The agent container where tools execute."""
        return self.sandbox_state.agent_container_id


@runtime_checkable
class BenignTaskBashEnvMetadataProtocol(Protocol):
    """Protocol defining the metadata interface for benign tasks in BashEnvironment.

    Datasets using BashEnvironment should provide metadata objects that implement this
    protocol. The protocol allows datasets to include additional dataset-specific fields
    beyond these required attributes.

    Attributes:
        agent_container_spec: Specification for the container where the agent executes.
        service_containers: Dictionary mapping service names to their container specs.
            These are auxiliary containers (e.g., databases, servers) that the agent
            may interact with during task execution.
    """

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec]


@runtime_checkable
class MaliciousTaskBashEnvMetadataProtocol(Protocol):
    """Protocol defining the metadata interface for malicious tasks in BashEnvironment.

    Attributes:
        agent_container_spec: Specification for the container where the agent executes.
        service_containers: Dictionary mapping service names to their container specs.
            These typically include attacker-controlled services (e.g., malicious servers).
    """

    agent_container_spec: ContainerSpec
    service_containers: dict[str, ContainerSpec]

    def get_pair_image_tag(self, benign_task_id: str, malicious_task_id: str) -> str | None:
        """Get the pre-built pair image tag for this malicious task combined with a benign task.

        If the malicious task requires modifications to the benign container, this returns
        the tag for the pre-built combined image. Otherwise returns None.

        Args:
            benign_task_id: The ID of the benign task this is being paired with
            malicious_task_id: The ID of the malicious task

        Returns:
            The pre-built pair image tag, or None if no modification is needed
        """
        ...


def _create_benign_task_setup(
    task: BenignTask[BashEnvState] | MaliciousTask[BashEnvState],
) -> TaskSetup:
    """Creates setup for benign or malicious task with optional service containers.

    For benign tasks, uses the benign metadata directly.
    For malicious tasks run in isolation (not as part of a couple), uses the malicious
    task's own agent_container_spec.
    """
    match task:
        case BenignTask():
            if not isinstance(task.metadata, BenignTaskBashEnvMetadataProtocol):
                raise RuntimeError(
                    f"BenignTask metadata does not implement BenignTaskBashEnvMetadataProtocol. "
                    f"Got type: {type(task.metadata).__name__}. "
                    f"Expected protocol attributes: agent_container_spec, service_containers"
                )
            meta = task.metadata
        case MaliciousTask():
            if not isinstance(task.metadata, MaliciousTaskBashEnvMetadataProtocol):
                raise RuntimeError(
                    f"MaliciousTask metadata does not implement MaliciousTaskBashEnvMetadataProtocol. "
                    f"Got type: {type(task.metadata).__name__}. "
                    f"Expected protocol attributes: agent_container_spec, service_containers, get_pair_image_tag"
                )
            meta = task.metadata
        case _:
            assert_never(task)

    # Convert service containers dict to ContainerSetup dict
    service_containers = {
        name: ContainerSetup(name=name, spec=spec) for name, spec in meta.service_containers.items()
    }

    # Create network config if there are service containers
    network_config = None
    if service_containers:
        safe_task_id = task.id.replace(":", "-").replace("/", "-")
        network_config = NetworkConfig(name=f"net-{safe_task_id}", internal=True)

    return TaskSetup(
        task_id=task.id,
        agent_container=ContainerSetup(
            name="agent",
            spec=meta.agent_container_spec,
        ),
        service_containers=service_containers,
        network_config=network_config,
    )


def _create_task_couple_setup(task: TaskCouple[BashEnvState]) -> TaskSetup:
    """Creates multi-container setup for task couple with agent + service containers.

    Merges benign and malicious service containers. If there are naming conflicts,
    malicious service containers take precedence.

    Uses pre-built pair images when available (via get_pair_image_tag).
    """
    if not isinstance(task.benign.metadata, BenignTaskBashEnvMetadataProtocol):
        raise RuntimeError(
            f"TaskCouple benign task metadata does not implement BenignTaskBashEnvMetadataProtocol. "
            f"Got type: {type(task.benign.metadata).__name__}. "
            f"Expected protocol attributes: agent_container_spec, service_containers"
        )
    benign_meta = task.benign.metadata

    if not isinstance(task.malicious.metadata, MaliciousTaskBashEnvMetadataProtocol):
        raise RuntimeError(
            f"TaskCouple malicious task metadata does not implement MaliciousTaskBashEnvMetadataProtocol. "
            f"Got type: {type(task.malicious.metadata).__name__}. "
            f"Expected protocol attributes: agent_container_spec, service_containers, get_pair_image_tag"
        )
    malicious_meta = task.malicious.metadata

    # Merge service containers (malicious takes precedence over benign)
    merged_service_containers = {
        name: ContainerSetup(name=name, spec=spec)
        for name, spec in {
            **benign_meta.service_containers,
            **malicious_meta.service_containers,
        }.items()
    }

    # Sanitize task ID for network name (Docker doesn't allow ":")
    safe_task_id = task.id.replace(":", "-").replace("/", "-")

    # Determine the agent container spec - use pre-built pair image if available
    pair_image_tag = malicious_meta.get_pair_image_tag(task.benign.id, task.malicious.id)
    if pair_image_tag is not None:
        # Use pre-built pair image
        agent_container_spec = ContainerSpec(image_spec=PullImageSpec(tag=pair_image_tag))
        agent_container = ContainerSetup(
            name="agent",
            spec=agent_container_spec,
        )
    else:
        # No modification needed, use benign spec directly
        agent_container = ContainerSetup(
            name="agent",
            spec=benign_meta.agent_container_spec,
        )

    return TaskSetup(
        task_id=task.id,
        agent_container=agent_container,
        service_containers=merged_service_containers,
        network_config=NetworkConfig(name=f"net-{safe_task_id}", internal=True),
    )


def _create_task_setup_from_task(
    task: TaskCouple[BashEnvState] | BenignTask[BashEnvState] | MaliciousTask[BashEnvState],
) -> SandboxTaskSetup:
    """Convert task to unified setup structure.

    For TaskCouple: creates multi-container setup with benign + attack containers
    For single tasks: creates single-container setup with just benign container
    """
    if isinstance(task, TaskCouple):
        return _create_task_couple_setup(task)
    return _create_benign_task_setup(task)


# Type of the sandbox manager
SandboxManagerT = TypeVar("SandboxManagerT", bound=AbstractSandboxManager)

# Types of the metadata
BenignMetadataT = TypeVar("BenignMetadataT", bound=BenignTaskBashEnvMetadataProtocol)
MaliciousMetadataT = TypeVar("MaliciousMetadataT", bound=MaliciousTaskBashEnvMetadataProtocol)


class BashEnvironment(
    Generic[SandboxManagerT, BenignMetadataT, MaliciousMetadataT],
    SnapshottableAbstractEnvironment[BashEnvState, str, str, StrContentAttack],
):
    """Generic bash execution environment for running agent tasks in sandboxed containers.

    This environment orchestrates Docker-based sandboxes via a SandboxManager, supporting
    both simple single-container tasks and complex multi-container setups with service
    dependencies (databases, web servers, etc.).

    Generic Type Parameters:
        SandboxManagerT: Type of sandbox manager (e.g., DockerSandboxManager)
        BenignMetadataT: Dataset-specific benign task metadata type
        MaliciousMetadataT: Dataset-specific malicious task metadata type

    Note on Generics:
        The generic type parameters are primarily for **documentation and IDE support**.
        Python's type system doesn't enforce these at runtime - the actual validation
        happens through runtime protocol checks (isinstance with @runtime_checkable).
        These generics help type checkers and IDEs understand which metadata types are
        expected when working with a specific BashEnvironment instance.

    Protocol-Based Design:
        Rather than hard-coding specific metadata classes, this environment uses Protocols
        (BenignTaskBashEnvMetadataProtocol and MaliciousTaskBashEnvMetadataProtocol) to
        define the required metadata interface. This allows different datasets to provide
        their own metadata implementations with additional dataset-specific fields, as long
        as they implement the required protocol attributes.
    """

    name: str = "bash"
    all_injection_ids: list[InjectionVectorID]
    _sandbox_manager: SandboxManagerT

    def __init__(
        self,
        sandbox_manager: SandboxManagerT,
        all_injection_ids: list[InjectionVectorID],
    ) -> None:
        self._sandbox_manager = sandbox_manager
        self.all_injection_ids = all_injection_ids

    async def copy_env_state(self, env_state: BashEnvState) -> BashEnvState:
        """Clone sandbox state (both containers and network if present)."""
        new_sandbox_state = await env_state.sandbox_manager.clone_sandbox_state(
            env_state.sandbox_state
        )
        return dataclasses.replace(env_state, sandbox_state=new_sandbox_state)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[BashEnvState]]
            | Sequence[BenignTask[BashEnvState]]
            | Sequence[MaliciousTask[BashEnvState]]
            | Sequence[BenignTask[BashEnvState] | MaliciousTask[BashEnvState]]
        ),
    ) -> AsyncIterator[Self]:
        # Convert tasks to TaskSetup objects
        task_setups = [_create_task_setup_from_task(task) for task in tasks]
        async with self._sandbox_manager.setup_batch(task_setups):
            yield self

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[BashEnvState] | BenignTask[BashEnvState] | MaliciousTask[BashEnvState],
    ) -> AsyncIterator[BashEnvState]:
        # Recreate task setup from task (cheap object construction)
        task_setup = _create_task_setup_from_task(task)
        # Setup task returns SandboxState with all container IDs
        async with self._sandbox_manager.setup_task(task_setup) as sandbox_state:
            yield BashEnvState(sandbox_state, self._sandbox_manager)

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        defaults: InjectionAttacksDict[StrContentAttack] = {}
        for vector_id in injection_vector_ids:
            defaults[vector_id] = StrContentAttack(content="")
        return defaults

    async def get_injectable_ids(self, raw_output: str) -> list[InjectionVectorID]:
        return [
            injection_id for injection_id in self.all_injection_ids if injection_id in raw_output
        ]

    async def render(
        self,
        raw_output: str,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> str:
        defaults = await self.get_default_for_injection_vectors(self.all_injection_ids)

        # Create empty dict if attacks is None
        attacks = attacks or {}

        # When an attack is missing, render the default value
        attacks = defaults | attacks

        final_output = raw_output
        for injection_id, attack in attacks.items():
            final_output = final_output.replace(injection_id, attack.content)

        return final_output

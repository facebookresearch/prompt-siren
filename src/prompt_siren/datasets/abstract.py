# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Abstract base classes and protocols for datasets.

Datasets provide collections of tasks (benign, malicious, and task couples) along with
the environment instance for executing those tasks. Each dataset owns its execution
environment, making environment selection an implementation detail.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable, TypeVar

from pydantic_ai.toolsets import FunctionToolset

from ..environments.abstract import AbstractEnvironment
from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttack

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT", contravariant=True)
FinalOutputT = TypeVar("FinalOutputT", covariant=True)
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


@runtime_checkable
class AbstractDataset(Protocol[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Protocol for datasets that provide tasks and their execution environment.

    A dataset encapsulates:
    1. A collection of tasks (benign, malicious, and task couples)
    2. The environment instance for executing those tasks

    Each dataset owns and manages its environment instance, making environment
    selection an implementation detail of the dataset.
    """

    name: str
    """Unique identifier for this dataset"""

    @property
    @abstractmethod
    def system_prompt(self) -> str | None: ...

    @property
    @abstractmethod
    def environment(
        self,
    ) -> AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """The environment instance for executing this dataset's tasks.

        Returns:
            Environment instance configured for this dataset's execution context
        """
        ...

    @property
    @abstractmethod
    def default_toolsets(self) -> list[FunctionToolset[EnvStateT]]:
        """Returns the default toolsets for this dataset.

        Returns:
            List of toolsets that agents can use with this dataset's tasks
        """
        ...

    @property
    @abstractmethod
    def benign_tasks(self) -> list[BenignTask[EnvStateT]]:
        """Returns all benign tasks in this dataset.

        Returns:
            List of unique benign tasks
        """
        ...

    @property
    @abstractmethod
    def malicious_tasks(self) -> list[MaliciousTask[EnvStateT]]:
        """Returns all malicious tasks in this dataset.

        Returns:
            List of unique malicious tasks
        """
        ...

    @property
    @abstractmethod
    def task_couples(self) -> list[TaskCouple[EnvStateT]]:
        """Returns valid task couples for attack evaluation.

        Dataset defines which pairings are valid for this benchmark.

        Returns:
            List of task couples (dataset-defined valid pairings)
        """
        ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Base class for the browser dataset.

This module provides shared browser dataset functionality:
- Container setup and management
- Task definitions (benign, malicious, couples)
- Injection handling
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from playwright.async_api import Page
from pydantic_ai.toolsets import FunctionToolset

from ...environments.abstract import AbstractEnvironment
from ...environments.browser_env import (
    BrowserEnvironment,
    BrowserEnvState,
    BrowserTaskMetadata,
)
from ...sandbox_managers.abstract import AbstractSandboxManager
from ...sandbox_managers.image_spec import BuildImageSpec, ImageBuildSpec
from ...tasks import BenignTask, MaliciousTask, TaskCouple
from ...types import InjectionAttacksDict, StrContentAttack
from ..abstract import AbstractDataset
from .config import BrowserDatasetConfig, SiteName
from .injection import get_vectors_for_sites
from .malicious_tasks import MALICIOUS_TASKS
from .sites import ANSWER_BENIGN_TASKS, GITEA_BENIGN_TASKS

# Output type varies by observation modality
OutputT = TypeVar("OutputT")

# All tasks (flat lists)
ALL_BENIGN_TASKS: list[BenignTask[BrowserEnvState]] = GITEA_BENIGN_TASKS + ANSWER_BENIGN_TASKS
ALL_MALICIOUS_TASKS: list[MaliciousTask[BrowserEnvState]] = MALICIOUS_TASKS


def _compute_sites_with_tasks() -> frozenset[SiteName]:
    """Compute all sites that have tasks defined (called once at module load)."""
    sites: set[SiteName] = set()

    for task in ALL_BENIGN_TASKS + ALL_MALICIOUS_TASKS:
        if isinstance(task.metadata, BrowserTaskMetadata):
            sites.update(task.metadata.sites)

    return frozenset(sites)


# Computed once at module load
SITES_WITH_TASKS: frozenset[SiteName] = _compute_sites_with_tasks()


# Type alias for render function
RenderFn = Callable[[Page, InjectionAttacksDict[StrContentAttack] | None], Awaitable[OutputT]]


@dataclass(frozen=True)
class BaseBrowserDataset(
    AbstractDataset[BrowserEnvState, Any, OutputT, StrContentAttack],
    Generic[OutputT],
):
    """Base class for browser datasets."""

    name: str
    _environment: BrowserEnvironment[OutputT] | None = None
    _benign_tasks: list[BenignTask[BrowserEnvState]] = field(default_factory=list)
    _malicious_tasks: list[MaliciousTask[BrowserEnvState]] = field(default_factory=list)
    _task_couples: list[TaskCouple[BrowserEnvState]] = field(default_factory=list)
    _toolsets: list[FunctionToolset[BrowserEnvState]] = field(default_factory=list)
    _system_prompt: str | None = None

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def environment(
        self,
    ) -> AbstractEnvironment[BrowserEnvState, Any, OutputT, StrContentAttack]:
        if self._environment is None:
            raise RuntimeError(
                "Cannot access environment: dataset was created for image building only "
                "(sandbox_manager was None). Create the dataset with a sandbox_manager "
                "to enable task execution."
            )
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[BrowserEnvState]]:
        return self._toolsets

    @property
    def benign_tasks(self) -> list[BenignTask[BrowserEnvState]]:
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[BrowserEnvState]]:
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[BrowserEnvState]]:
        return self._task_couples

    @classmethod
    def get_image_build_specs(
        cls, config: BrowserDatasetConfig, build_context_dir: str
    ) -> list[ImageBuildSpec]:
        """Return all image specifications needed by this dataset.

        This classmethod is used by the build_images script to pre-build all Docker
        images needed for browser tasks. It iterates through the site container
        specs and returns ImageBuildSpec for sites that need building.

        Args:
            config: Browser dataset configuration specifying which sites to build.
            build_context_dir: Directory to store build contexts and generated scripts (unused).

        Returns:
            List of ImageBuildSpec objects for sites that need to be built
            (excludes PullImageSpec sites that are pulled from registries).
        """
        _ = build_context_dir
        # Get all site container specs from config and filter to only BuildImageSpec
        # (skip PullImageSpec as those are pulled from registries)
        site_specs = config.get_all_site_container_specs()
        return [
            container_spec.image_spec
            for container_spec in site_specs.values()
            if isinstance(container_spec.image_spec, BuildImageSpec)
        ]


def create_browser_environment(
    config: BrowserDatasetConfig,
    sandbox_manager: AbstractSandboxManager,
    render_fn: RenderFn[OutputT],
    *,
    name: str = "browser",
) -> BrowserEnvironment[OutputT]:
    """Create a browser environment with the given render function.

    Args:
        config: Browser dataset configuration
        sandbox_manager: Sandbox manager for container lifecycle
        render_fn: Function to render Page to observation format
        name: Name identifier for the environment

    Returns:
        Configured BrowserEnvironment
    """
    # Use pre-computed sites from module load
    sites = SITES_WITH_TASKS

    return BrowserEnvironment(
        name=name,
        all_injection_ids=get_vectors_for_sites(list(sites)),
        sandbox_manager=sandbox_manager,
        browser_container_spec=config.browser.to_container_spec(),
        site_container_specs=config.get_all_site_container_specs(),
        render_fn=render_fn,
    )

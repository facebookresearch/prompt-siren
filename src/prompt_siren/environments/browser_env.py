# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser environment for web agent tasks.

This environment provides a browser-based execution context for web agent tasks,
with fresh containers created per task for complete isolation.

Container Management:
    Follows the same pattern as SWE-bench/BashEnvironment:
    - create_batch_context(): Pulls/prepares all container images upfront
    - create_task_context(): Creates fresh browser + site containers per task
    - Containers are cleaned up automatically when task context exits

    This provides true parallel execution support - each task gets its own
    isolated set of containers with fresh state.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, get_args, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field, HttpUrl
from typing_extensions import Self

try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright, Request, Route
except ImportError as e:
    raise ImportError(
        "Browser environment requires the 'playwright' optional dependency. "
        "Install with: pip install 'prompt-siren[browser]'"
    ) from e

from ..sandbox_managers.abstract import AbstractSandboxManager
from ..sandbox_managers.sandbox_state import ContainerID, SandboxState
from ..sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    TaskSetup,
)
from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttacksDict, InjectionVectorID, StrContentAttack
from .abstract import NonSnapshottableAbstractEnvironment

logger = logging.getLogger(__name__)

# Output type varies by observation modality
OutputT = TypeVar("OutputT")

# Type alias for render function (attacks can be None for benign rendering)
RenderFn = Callable[[Page, InjectionAttacksDict[StrContentAttack] | None], Awaitable[OutputT]]

# Valid site names for browser environment
SiteName = Literal["gitea", "answer", "wikijs", "classifieds"]


async def _setup_page_with_capture(
    browser: Browser,
    start_url: str,
) -> tuple[Page, list[CapturedRequest]]:
    """Create a new page with request capture and navigate to start URL.

    Args:
        browser: Browser instance to create page from
        start_url: URL to navigate to after page creation

    Returns:
        Tuple of (page, captured_requests list)

    Raises:
        Exception: Re-raises any exception after cleaning up the page resource.
    """
    page = await browser.new_page()
    captured_requests: list[CapturedRequest] = []

    try:

        async def capture_handler(route: Route, request: Request) -> None:
            captured_requests.append(
                CapturedRequest(
                    url=request.url,
                    method=request.method,
                    post_data=request.post_data,
                )
            )
            await route.continue_()

        await page.route("**/*", capture_handler)
        await page.goto(start_url)

        return page, captured_requests
    except Exception:
        await page.close()
        raise


class InjectionError(Exception):
    """Error raised when injection fails."""


async def apply_injections(
    page: Page,
    attacks: InjectionAttacksDict[StrContentAttack] | None,
) -> None:
    """Apply injection attacks to the page DOM.

    This modifies DOM text nodes to replace placeholders with attack content.
    Used by dataset render functions to inject attack payloads before rendering.

    Placeholders use the format ``{vector_id}`` where ``vector_id`` is the key
    in the attacks dictionary. For example, if attacks contains
    ``{"gitea_issue_body": StrContentAttack(content="malicious")}``, then
    all occurrences of ``{gitea_issue_body}`` in DOM text nodes will be
    replaced with ``malicious``.

    Note: Only text nodes are modified. Placeholders in HTML attributes or
    script content will not be replaced.

    Args:
        page: Playwright page to modify
        attacks: Injection attacks to apply, mapping vector IDs to attack content.
            Does nothing if None or empty.

    Raises:
        InjectionError: If JavaScript evaluation fails (e.g., page navigated away)
    """
    if not attacks:
        return

    for vector_id, attack in attacks.items():
        placeholder = f"{{{vector_id}}}"
        try:
            await page.evaluate(
                """([placeholder, replacement]) => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    while (walker.nextNode()) {
                        if (walker.currentNode.textContent.includes(placeholder)) {
                            walker.currentNode.textContent =
                                walker.currentNode.textContent.replace(placeholder, replacement);
                        }
                    }
                }""",
                [placeholder, attack.content],
            )
        except Exception as e:
            raise InjectionError(
                f"Failed to apply injection for vector '{vector_id}' on page {page.url}"
            ) from e


class CapturedRequest(TypedDict):
    """Captured HTTP request from browser."""

    url: str
    method: str
    post_data: str | None


@dataclass
class BrowserEnvState:
    """Environment state for browser-based tasks.

    Attributes:
        page: The Playwright page object for browser interactions
        browser: The Playwright browser connection (for closing on reset)
        playwright: The Playwright instance (for reconnecting on reset)
        sandbox_state: Container IDs and network info for exec access
        sandbox_manager: Manager for executing commands in containers
        task_setup: Task setup for recreating containers on reset
        start_url: Initial URL for navigation on reset

    Properties:
        captured_requests: Read-only sequence of captured HTTP requests (for attack evaluation)
    """

    page: Page
    browser: Browser
    playwright: Playwright
    sandbox_state: SandboxState
    sandbox_manager: AbstractSandboxManager
    task_setup: TaskSetup
    start_url: str
    _captured_requests: list[CapturedRequest] = field(default_factory=list)

    @property
    def captured_requests(self) -> tuple[CapturedRequest, ...]:
        """Read-only tuple of captured requests for attack evaluation."""
        return tuple(self._captured_requests)

    @property
    def browser_container_id(self) -> ContainerID:
        """The browser container ID.

        Useful for evaluators that need to execute commands in the browser container
        via sandbox_manager.exec().
        """
        return self.sandbox_state.agent_container_id

    def get_site_container_id(self, site_name: SiteName) -> ContainerID | None:
        """Get container ID for a specific site.

        Useful for evaluators that need to inspect site state (database, files, logs)
        via sandbox_manager.exec().

        Args:
            site_name: Name of the site (gitea, answer, wikijs, classifieds)

        Returns:
            Container ID if the site container exists, None otherwise
        """
        return self.sandbox_state.service_containers.get(site_name)


class BrowserTaskMetadata(BaseModel):
    """Metadata for browser-based tasks.

    All browser tasks specify which site(s) they interact with
    and the starting URL for the task.
    """

    sites: list[SiteName] = Field(min_length=1)
    """Sites this task interacts with (at least one required)."""
    start_url: HttpUrl
    """Starting URL for this task (validated as HTTP/HTTPS URL)."""


class BrowserEnvironment(
    NonSnapshottableAbstractEnvironment[BrowserEnvState, Page, OutputT, StrContentAttack],
    Generic[OutputT],
):
    """Browser environment with fresh containers per task.

    Follows the SWE-bench pattern:
    - create_batch_context(): Prepares all container images
    - create_task_context(): Creates fresh browser + site containers per task
    - Complete isolation between tasks (no shared state)
    - Supports true parallel execution

    Uses tool replay (NonSnapshottable) since Page objects cannot be cloned.

    The OutputT type parameter determines the observation format:
    - BinaryContent for screenshot-based observations
    - str for accessibility tree or HTML observations
    """

    name: str
    all_injection_ids: list[InjectionVectorID]

    _sandbox_manager: AbstractSandboxManager
    _browser_container_spec: ContainerSpec
    _site_container_specs: dict[str, ContainerSpec]
    _render_fn: RenderFn[OutputT]

    # Task setups prepared during batch context
    _task_setups: list[TaskSetup]

    def __init__(
        self,
        *,
        name: str,
        all_injection_ids: list[InjectionVectorID],
        sandbox_manager: AbstractSandboxManager,
        browser_container_spec: ContainerSpec,
        site_container_specs: dict[str, ContainerSpec],
        render_fn: RenderFn[OutputT],
    ) -> None:
        """Initialize browser environment.

        Args:
            name: Name identifier for the environment
            all_injection_ids: List of injection vector IDs supported
            sandbox_manager: Sandbox manager for container lifecycle
            browser_container_spec: Spec for browser container (Chromium with CDP)
            site_container_specs: Specs for site containers (Gitea, Answer, etc.)
            render_fn: Function to render Page to observation format (OutputT)
        """
        self.name = name
        self.all_injection_ids = all_injection_ids

        self._sandbox_manager = sandbox_manager
        self._browser_container_spec = browser_container_spec
        self._site_container_specs = site_container_specs
        self._render_fn = render_fn

        self._task_setups = []

    def _get_cdp_endpoint(self, sandbox_state: SandboxState) -> str:
        """Get the CDP endpoint URL from sandbox state.

        Uses the dynamically allocated host port from sandbox_state.agent_port_bindings.

        Args:
            sandbox_state: Sandbox state with port bindings from container creation

        Returns:
            CDP endpoint URL (e.g., "http://localhost:32768")

        Raises:
            RuntimeError: If no CDP port binding is found
        """
        # Get the container port (e.g., 9222) from the spec
        if not self._browser_container_spec.ports:
            raise RuntimeError("Browser container spec must have ports defined")
        container_port = next(iter(self._browser_container_spec.ports.values()))

        # Look up the actual host port from the sandbox state
        if container_port not in sandbox_state.agent_port_bindings:
            raise RuntimeError(
                f"CDP port {container_port} not found in agent_port_bindings. "
                f"Available bindings: {sandbox_state.agent_port_bindings}"
            )

        host_port = sandbox_state.agent_port_bindings[container_port]
        return f"http://localhost:{host_port}"

    async def reset_env_state(self, env_state: BrowserEnvState) -> BrowserEnvState:
        """Reset env_state by recreating containers from scratch.

        For browser environment, this:
        1. Closes the browser connection
        2. Replaces the sandbox with fresh containers (old sandbox cleaned in background)
        3. Reconnects browser via CDP
        4. Creates new page with request capture
        5. Navigates to start URL

        This ensures complete state reset including site container state
        (databases, files, etc.) for proper tool replay.
        """
        # Close browser connection first
        await env_state.browser.close()

        # Replace sandbox with fresh containers (old sandbox cleaned in background)
        new_sandbox_state = await env_state.sandbox_manager.replace_sandbox(
            env_state.sandbox_state, env_state.task_setup
        )

        # Connect to browser via CDP using dynamically allocated port
        cdp_endpoint = self._get_cdp_endpoint(new_sandbox_state)

        # Reconnect browser using existing Playwright instance
        new_browser = await env_state.playwright.chromium.connect_over_cdp(cdp_endpoint)

        # Create new page with request capture and navigate to start URL
        new_page, new_captured_requests = await _setup_page_with_capture(
            new_browser, env_state.start_url
        )

        return BrowserEnvState(
            page=new_page,
            browser=new_browser,
            playwright=env_state.playwright,
            sandbox_state=new_sandbox_state,
            sandbox_manager=env_state.sandbox_manager,
            task_setup=env_state.task_setup,
            start_url=env_state.start_url,
            _captured_requests=new_captured_requests,
        )

    async def get_injectable_ids(self, raw_output: Page) -> list[InjectionVectorID]:
        """Detect which injection vectors are present in the page."""
        page_content = await raw_output.content()
        return [
            vector_id for vector_id in self.all_injection_ids if f"{{{vector_id}}}" in page_content
        ]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Returns default content for each vector (benign placeholder text)."""
        return {vid: StrContentAttack(content="[No content]") for vid in injection_vector_ids}

    async def render(
        self,
        raw_output: Page,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> OutputT:
        """Render page with injections using the configured render function."""
        # Get defaults for any detected vectors
        vector_ids = await self.get_injectable_ids(raw_output)
        defaults = await self.get_default_for_injection_vectors(vector_ids)

        # Merge defaults with provided attacks (attacks override defaults)
        effective_attacks = defaults | (attacks or {})

        return await self._render_fn(raw_output, effective_attacks)

    def _extract_sites_from_single_task(
        self,
        task: BenignTask[BrowserEnvState] | MaliciousTask[BrowserEnvState],
    ) -> list[SiteName]:
        """Extract valid sites from a single task's metadata.

        Returns sites in the order specified in metadata, filtered to valid SiteNames.

        Raises:
            TypeError: If task metadata is not BrowserTaskMetadata.
        """
        metadata = task.metadata
        if not isinstance(metadata, BrowserTaskMetadata):
            raise TypeError(
                f"Task {task.id} has unexpected metadata type {type(metadata).__name__} "
                f"(expected BrowserTaskMetadata). This indicates a bug in task definition."
            )

        valid_sites = get_args(SiteName)
        return [site for site in metadata.sites if site in valid_sites]

    def _get_sites_from_task(
        self,
        task: TaskCouple[BrowserEnvState]
        | BenignTask[BrowserEnvState]
        | MaliciousTask[BrowserEnvState],
        *,
        include_malicious: bool = True,
    ) -> list[SiteName]:
        """Extract sites required by a task.

        Args:
            task: The task or task couple to extract sites from
            include_malicious: For TaskCouples, whether to include malicious task sites.
                              Set to False to get only benign task sites (for URL resolution).

        Returns:
            Ordered list of sites. First element is the primary site (from benign task).
            For TaskCouples with include_malicious=True, includes sites from both tasks.
        """
        # Collect tasks to check
        tasks_to_check: list[BenignTask[BrowserEnvState] | MaliciousTask[BrowserEnvState]]
        if isinstance(task, TaskCouple):
            tasks_to_check = [task.benign]
            if include_malicious:
                tasks_to_check.append(task.malicious)
        else:
            tasks_to_check = [task]

        # Extract sites preserving order (first task's sites come first)
        seen: set[SiteName] = set()
        result: list[SiteName] = []
        for t in tasks_to_check:
            for site in self._extract_sites_from_single_task(t):
                if site not in seen:
                    seen.add(site)
                    result.append(site)

        return result

    def _create_task_setup(
        self,
        task: TaskCouple[BrowserEnvState]
        | BenignTask[BrowserEnvState]
        | MaliciousTask[BrowserEnvState],
    ) -> TaskSetup:
        """Create TaskSetup for a single task with browser + required site containers."""
        task_id = task.id
        sites = self._get_sites_from_task(task)

        # Build service containers from required sites
        service_containers: dict[str, ContainerSetup] = {}
        for site in sites:
            if site not in self._site_container_specs:
                raise ValueError(
                    f"Task {task_id} requires site '{site}' but no container spec is configured. "
                    f"Available sites: {list(self._site_container_specs.keys())}"
                )

            db_key = f"{site}-db"
            if db_key in self._site_container_specs and db_key not in service_containers:
                service_containers[db_key] = ContainerSetup(
                    name=db_key,
                    spec=self._site_container_specs[db_key],
                )

            service_containers[site] = ContainerSetup(
                name=site,
                spec=self._site_container_specs[site],
            )

        # Sanitize task ID for network name
        safe_task_id = task_id.replace(":", "-").replace("/", "-")

        return TaskSetup(
            task_id=task_id,
            agent_container=ContainerSetup(
                name="browser",
                spec=self._browser_container_spec,
            ),
            service_containers=service_containers,
            network_config=NetworkConfig(name=f"browser-net-{safe_task_id}", internal=False),
        )

    def _get_start_url(
        self,
        task: TaskCouple[BrowserEnvState]
        | BenignTask[BrowserEnvState]
        | MaliciousTask[BrowserEnvState],
    ) -> str:
        """Get the starting URL for a task from its metadata."""
        actual_task = task.benign if isinstance(task, TaskCouple) else task
        metadata = actual_task.metadata
        if not isinstance(metadata, BrowserTaskMetadata):
            raise ValueError(f"Task {actual_task.id} must have BrowserTaskMetadata")
        return str(metadata.start_url)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[BrowserEnvState]]
            | Sequence[BenignTask[BrowserEnvState]]
            | Sequence[MaliciousTask[BrowserEnvState]]
            | Sequence[BenignTask[BrowserEnvState] | MaliciousTask[BrowserEnvState]]
        ),
    ) -> AsyncIterator[Self]:
        """Prepare container images for batch execution.

        This context manager prepares all required images upfront via setup_batch().
        Actual containers are created per-task in create_task_context().
        """
        # Create task setups for all tasks
        self._task_setups = [self._create_task_setup(task) for task in tasks]

        async with self._sandbox_manager.setup_batch(self._task_setups):
            try:
                yield self
            finally:
                self._task_setups = []

    @asynccontextmanager
    async def create_task_context(
        self,
        task: TaskCouple[BrowserEnvState]
        | BenignTask[BrowserEnvState]
        | MaliciousTask[BrowserEnvState],
    ) -> AsyncIterator[BrowserEnvState]:
        """Create per-task context with fresh containers.

        Creates fresh browser + site containers for complete isolation.
        Supports true parallel execution - each task gets its own containers.

        Note: Uses async_playwright().start() instead of context manager so that
        the Playwright instance can be passed to BrowserEnvState for use in
        reset_env_state() to reconnect to recreated containers.
        """
        task_setup = self._create_task_setup(task)

        async with self._sandbox_manager.setup_task(task_setup) as sandbox_state:
            # Connect to browser via CDP using dynamically allocated port
            cdp_endpoint = self._get_cdp_endpoint(sandbox_state)

            # Get starting URL from task metadata
            start_url = self._get_start_url(task)

            # Use .start() instead of context manager so we can pass pw to env_state
            pw = await async_playwright().start()
            try:
                browser = await pw.chromium.connect_over_cdp(cdp_endpoint)

                try:
                    # Create page with request capture and navigate to start URL
                    page, captured_requests = await _setup_page_with_capture(browser, start_url)

                    yield BrowserEnvState(
                        page=page,
                        browser=browser,
                        playwright=pw,
                        sandbox_state=sandbox_state,
                        sandbox_manager=self._sandbox_manager,
                        task_setup=task_setup,
                        start_url=start_url,
                        _captured_requests=captured_requests,
                    )
                finally:
                    await browser.close()
            finally:
                await pw.stop()

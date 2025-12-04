# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypedDict, TypeVar

from aiohttp.web import Application, AppRunner, Request, Response, TCPSite
from pydantic import BaseModel, Field

try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright
except ImportError as e:
    raise ImportError(
        "Playwright environment requires the 'playwright' optional dependency. "
        "Install with: pip install 'prompt-siren[playwright]'"
    ) from e
from pydantic_ai import RunContext
from pydantic_ai.messages import BinaryContent
from typing_extensions import Self

from ..tasks import BenignTask, MaliciousTask, TaskCouple
from ..types import InjectionAttacksDict, InjectionVectorID, StrContentAttack
from .abstract import AbstractEnvironment, NonSnapshottableAbstractEnvironment

RawOutputT = TypeVar("RawOutputT")


class RequestLog(TypedDict):
    path: str
    method: Literal["POST"]
    content: dict


@dataclass(frozen=True)
class Server:
    runner: AppRunner
    site: TCPSite
    port: int
    logs: list[RequestLog] = field(default_factory=list)

    async def close(self):
        await self.site.stop()
        await self.runner.cleanup()


async def render_html(request: Request, website_dir: Path):
    filename = request.match_info["filename"]  # type: ignore[non-subscriptable]
    filepath = website_dir / filename
    # Security: prevent directory traversal
    if not str(filepath).startswith(str(website_dir)):
        return Response(status=404, text="Not found")
    if not filepath.exists():
        return Response(status=404, text="Not found")
    # Read the HTML file
    with open(filepath, encoding="utf-8") as f:
        html = f.read()
    return Response(text=html, content_type="text/html")


async def _inject_page(page: Page, attack: InjectionAttacksDict[StrContentAttack]) -> None:
    await page.evaluate(
        """
        (attack) => {
            for (const [id, value] of Object.entries(attack)) {
                const el = document.getElementById(id);
                if (el) {
                    el.textContent = value;
                }
            }
        }
        """,
        {k: v.content for k, v in attack.items()},
    )


async def handle_post(request: Request):
    # This will match any POST path
    path = request.path
    data = await request.post()
    return Response(text=f"Received POST on path: {path} with data: {data}")


class PlaywrightEnvironmentConfig(BaseModel):
    """Configuration for Playwright environment."""

    website_dir: Path = Field(description="Path to the directory containing the website files")
    injection_ids: list[str] = Field(description="List of injection vector IDs to use")
    name: str = Field(description="Name identifier for the environment")
    output_type: Literal["screenshot", "html"] = Field(
        default="screenshot",
        description="Type of output to generate: 'screenshot' for images or 'html' for HTML content",
    )


@dataclass
class PlaywrightEnvironment(
    NonSnapshottableAbstractEnvironment[Page, Page, RawOutputT, StrContentAttack],
    Generic[RawOutputT],
):
    website_dir: Path
    all_injection_ids: list[InjectionVectorID]
    name: str

    _pw: Playwright | None = None
    _browser: Browser | None = None
    _server: Server | None = None

    async def reset_env_state(self, env_state: Page) -> Page:
        """Reset the browser page to initial state.

        Navigates back to the initial URL and clears cookies to ensure
        a clean state before tool replay.

        Args:
            env_state: The Page to reset

        Returns:
            The same Page after reset (Playwright pages are stateful)
        """
        if self._server is None:
            raise RuntimeError("Server not initialized - must be in batch context")

        # Navigate to initial URL
        url = f"http://localhost:{self._server.port}/index.html"
        await env_state.goto(url, timeout=30000)  # 30 second timeout

        # Clear cookies and storage
        await env_state.context.clear_cookies()

        return env_state

    async def _init_browser(self) -> None:
        if self._browser is not None:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch()

    async def _init_server(self) -> None:
        app = Application()
        app.router.add_get(
            "/{filename}",
            lambda request: render_html(request, self.website_dir),
        )
        app.router.add_post("/{tail:.*}", handle_post)
        runner = AppRunner(app)
        await runner.setup()
        # TODO: make sure port is not used
        site = TCPSite(runner, "localhost", 42352)
        await site.start()
        self._server = Server(runner, site, site._port)

    @asynccontextmanager
    async def create_batch_context(
        self,
        tasks: (
            Sequence[TaskCouple[Page]]
            | Sequence[BenignTask[Page]]
            | Sequence[MaliciousTask[Page]]
            | Sequence[BenignTask[Page] | MaliciousTask[Page]]
        ),
    ) -> AsyncIterator[Self]:
        """Set up browser and server for batch execution.

        Args:
            tasks: The list of tasks to be executed in this batch (can be used to optimize resource setup).
        """
        await self._init_browser()
        await self._init_server()
        try:
            yield self
        finally:
            # Clean up browser and server
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
            if self._server:
                await self._server.close()

    async def get_injectable_ids(self, raw_output: Page) -> list[InjectionVectorID]:
        return [
            injection_id
            for injection_id in self.all_injection_ids
            if await raw_output.query_selector(f"#{injection_id}")
        ]

    async def get_default_for_injection_vectors(
        self, injection_vector_ids: Sequence[InjectionVectorID]
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Returns default content for the given injection vectors."""
        return {vector_id: StrContentAttack(content="") for vector_id in injection_vector_ids}

    @asynccontextmanager
    async def create_task_context(
        self, task: TaskCouple[Page] | BenignTask[Page] | MaliciousTask[Page]
    ) -> AsyncIterator[Page]:
        """Create per-task context with fresh browser page.

        Args:
            task: The task being executed (used for task-specific environment setup)

        Yields:
            Fresh browser page for this task execution
        """
        if self._browser is None or self._server is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be entered via create_batch_context() before creating task contexts"
            )
        page = await self._browser.new_page()
        url = f"http://localhost:{self._server.port}/index.html"
        await page.goto(url)
        try:
            yield page
        finally:
            await page.close()


class ScreenshotPlayWrightEnvironment(PlaywrightEnvironment[BinaryContent]):
    async def render(
        self,
        raw_output: Page,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> BinaryContent:
        if attacks is not None:
            await _inject_page(raw_output, attacks)
        png_bytes = await raw_output.screenshot(full_page=False)
        return BinaryContent(data=png_bytes, media_type="image/png")


class HTMLPlaywrightEnvironment(PlaywrightEnvironment[str]):
    async def render(
        self,
        raw_output: Page,
        attacks: InjectionAttacksDict[StrContentAttack] | None = None,
    ) -> str:
        if attacks is not None:
            await _inject_page(raw_output, attacks)
        return await raw_output.content()


async def click(
    ctx: RunContext[Page],
    x: int,
    y: int,
    button: Literal["left", "right", "back", "forward", "wheel", "middle"] = "left",
) -> Page:
    page = ctx.deps
    match button:
        case "back":
            await page.go_back()
        case "forward":
            await page.go_forward()
        case "wheel":
            await page.mouse.wheel(x, y)
        case _:
            await page.mouse.click(x, y, button=button)
    return page


async def double_click(ctx: RunContext[Page], x: int, y: int) -> Page:
    page = ctx.deps
    await ctx.deps.mouse.dblclick(x, y)
    return page


async def scroll(ctx: RunContext[Page], x: int, y: int, scroll_x: int, scroll_y: int) -> Page:
    page = ctx.deps
    await ctx.deps.mouse.move(x, y)
    await ctx.deps.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
    return page


async def type_text(ctx: RunContext[Page], text: str) -> Page:
    page = ctx.deps
    await ctx.deps.keyboard.type(text)
    return page


def create_playwright_environment(
    config: PlaywrightEnvironmentConfig,
) -> AbstractEnvironment:
    """Factory function to create a Playwright environment."""
    if config.output_type == "screenshot":
        return ScreenshotPlayWrightEnvironment(
            website_dir=config.website_dir,
            all_injection_ids=config.injection_ids,
            name=config.name,
        )
    # html
    return HTMLPlaywrightEnvironment(
        website_dir=config.website_dir,
        all_injection_ids=config.injection_ids,
        name=config.name,
    )

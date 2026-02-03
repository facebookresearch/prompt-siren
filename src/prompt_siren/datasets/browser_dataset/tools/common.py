# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Common browser tools."""

from typing import Any

from pydantic_ai import RunContext

from ....environments.browser_env import BrowserEnvState


async def press_key(
    ctx: RunContext[BrowserEnvState],
    key: str,
) -> Any:
    """Press a keyboard key.

    Args:
        ctx: The run context containing the browser page
        key: Key to press (e.g., "Enter", "Tab", "Escape", "ArrowDown")

    Returns:
        The current page (rendered as an observation by the environment)
    """
    page = ctx.deps.page
    await page.keyboard.press(key)
    return page


async def goto_url(
    ctx: RunContext[BrowserEnvState],
    url: str,
) -> Any:
    """Navigate to a specific URL.

    Args:
        ctx: The run context containing the browser page
        url: URL to navigate to

    Returns:
        The current page (rendered as an observation by the environment)
    """
    page = ctx.deps.page
    await page.goto(url, timeout=30000)
    return page


async def go_back(
    ctx: RunContext[BrowserEnvState],
) -> Any:
    """Go back to the previous page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        The current page (rendered as an observation by the environment)
    """
    page = ctx.deps.page
    await page.go_back()
    return page


async def go_forward(
    ctx: RunContext[BrowserEnvState],
) -> Any:
    """Go forward to the next page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        The current page (rendered as an observation by the environment)
    """
    page = ctx.deps.page
    await page.go_forward()
    return page

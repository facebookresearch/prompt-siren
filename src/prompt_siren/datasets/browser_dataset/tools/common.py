# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Common browser tools shared across all observation modalities.

These tools are used regardless of whether the agent sees screenshots,
accessibility trees, or HTML.
"""

from pydantic_ai import RunContext

from ....environments.browser_env import BrowserEnvState


async def press_key(
    ctx: RunContext[BrowserEnvState],
    key: str,
) -> str:
    """Press a keyboard key.

    Args:
        ctx: The run context containing the browser page
        key: Key to press (e.g., "Enter", "Tab", "Escape", "ArrowDown")

    Returns:
        Status message describing the key press
    """
    page = ctx.deps.page
    await page.keyboard.press(key)
    return f"Pressed key: {key}"


async def goto_url(
    ctx: RunContext[BrowserEnvState],
    url: str,
) -> str:
    """Navigate to a specific URL.

    Args:
        ctx: The run context containing the browser page
        url: URL to navigate to

    Returns:
        Status message with the final URL
    """
    page = ctx.deps.page
    await page.goto(url, timeout=30000)
    return f"Navigated to: {page.url}"


async def go_back(
    ctx: RunContext[BrowserEnvState],
) -> str:
    """Go back to the previous page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        Status message with the current URL
    """
    page = ctx.deps.page
    await page.go_back()
    return f"Went back. Current URL: {page.url}"


async def go_forward(
    ctx: RunContext[BrowserEnvState],
) -> str:
    """Go forward to the next page in browser history.

    Args:
        ctx: The run context containing the browser page

    Returns:
        Status message with the current URL
    """
    page = ctx.deps.page
    await page.go_forward()
    return f"Went forward. Current URL: {page.url}"

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Coordinate-based browser tools for screenshot observations.

These tools are designed for agents that see screenshots and need to
interact using pixel coordinates.
"""

from typing import Literal

from pydantic_ai import RunContext

from ....environments.browser_env import BrowserEnvState


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if longer than max_len."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def click(
    ctx: RunContext[BrowserEnvState],
    x: int,
    y: int,
    button: Literal["left", "right", "middle"] = "left",
) -> str:
    """Click at a specific position on the page.

    Args:
        ctx: The run context containing the browser page
        x: X coordinate to click
        y: Y coordinate to click
        button: Mouse button to use (left, right, or middle)

    Returns:
        Status message describing the click action
    """
    page = ctx.deps.page
    await page.mouse.click(x, y, button=button)
    return f"Clicked at ({x}, {y}) with {button} button"


async def scroll(
    ctx: RunContext[BrowserEnvState],
    x: int,
    y: int,
    scroll_x: int,
    scroll_y: int,
) -> str:
    """Scroll the page by a specified amount.

    Args:
        ctx: The run context containing the browser page
        x: X coordinate to scroll from
        y: Y coordinate to scroll from
        scroll_x: Horizontal scroll amount (positive = right)
        scroll_y: Vertical scroll amount (positive = down)

    Returns:
        Status message describing the scroll action
    """
    page = ctx.deps.page
    await page.mouse.move(x, y)
    await page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
    return f"Scrolled by ({scroll_x}, {scroll_y}) from position ({x}, {y})"


async def type_text(
    ctx: RunContext[BrowserEnvState],
    text: str,
) -> str:
    """Type text using the keyboard.

    Args:
        ctx: The run context containing the browser page
        text: Text to type

    Returns:
        Status message describing the typing action
    """
    page = ctx.deps.page
    await page.keyboard.type(text)
    return f"Typed: {_truncate(text)}"

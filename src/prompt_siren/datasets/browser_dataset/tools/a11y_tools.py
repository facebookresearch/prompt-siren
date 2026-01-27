# Copyright (c) Meta Platforms, Inc. and affiliates.
"""ARIA role-based browser tools for accessibility tree observations.

These tools are designed for agents that receive accessibility tree
snapshots and interact using ARIA roles and accessible names.
"""

from typing import Literal

from playwright._impl._api_structures import AriaRole
from pydantic_ai import RunContext

from ....environments.browser_env import BrowserEnvState


async def click_element(
    ctx: RunContext[BrowserEnvState],
    role: AriaRole,
    name: str | None = None,
) -> str:
    """Click an element by its ARIA role and accessible name.

    Args:
        ctx: The run context containing the browser page
        role: ARIA role (e.g., "button", "link", "textbox", "heading", "menuitem")
        name: Accessible name of the element (optional for unique roles)

    Returns:
        Status message describing the click action
    """
    page = ctx.deps.page
    locator = page.get_by_role(role, name=name) if name else page.get_by_role(role)
    await locator.click(timeout=5000)
    name_part = f" with name '{name}'" if name else ""
    return f"Clicked {role}{name_part}"


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if longer than max_len."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def fill_element(
    ctx: RunContext[BrowserEnvState],
    role: AriaRole,
    value: str,
    name: str | None = None,
) -> str:
    """Fill a form element by its ARIA role and accessible name.

    Args:
        ctx: The run context containing the browser page
        role: ARIA role (typically "textbox", "searchbox", "combobox")
        value: Value to fill in
        name: Accessible name/label of the element

    Returns:
        Status message
    """
    page = ctx.deps.page
    locator = page.get_by_role(role, name=name) if name else page.get_by_role(role)
    await locator.fill(value, timeout=5000)
    name_part = f" '{name}'" if name else ""
    return f"Filled {role}{name_part} with: {_truncate(value)}"


async def select_option(
    ctx: RunContext[BrowserEnvState],
    name: str,
    option: str,
) -> str:
    """Select an option from a dropdown/combobox by accessible name.

    Args:
        ctx: The run context containing the browser page
        name: Accessible name of the select element
        option: Option text to select

    Returns:
        Status message
    """
    page = ctx.deps.page
    await page.get_by_role("combobox", name=name).select_option(option, timeout=5000)
    return f"Selected '{option}' from '{name}'"


async def check_element(
    ctx: RunContext[BrowserEnvState],
    name: str,
    checked: bool = True,
) -> str:
    """Check or uncheck a checkbox by accessible name.

    Args:
        ctx: The run context containing the browser page
        name: Accessible name of the checkbox
        checked: Whether to check (True) or uncheck (False)

    Returns:
        Status message
    """
    page = ctx.deps.page
    if checked:
        await page.get_by_role("checkbox", name=name).check(timeout=5000)
    else:
        await page.get_by_role("checkbox", name=name).uncheck(timeout=5000)
    return f"{'Checked' if checked else 'Unchecked'} checkbox '{name}'"


async def scroll_page(
    ctx: RunContext[BrowserEnvState],
    direction: Literal["up", "down"] = "down",
    amount: int = 500,
) -> str:
    """Scroll the page up or down.

    Args:
        ctx: The run context
        direction: "up" or "down"
        amount: Pixels to scroll

    Returns:
        Status message
    """
    page = ctx.deps.page
    scroll_y = amount if direction == "down" else -amount
    await page.evaluate(f"window.scrollBy(0, {scroll_y})")
    return f"Scrolled {direction} by {amount} pixels"

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser interaction tools for different observation modalities.

This module provides tools organized by observation type:
- common: Shared tools (navigation, keyboard)
- screenshot_tools: Coordinate-based tools for screenshot observations
- a11y_tools: ARIA role-based tools for accessibility tree observations
- html_tools: CSS selector-based tools for HTML observations
"""

from .a11y_tools import (
    check_element,
    click_element,
    fill_element,
    scroll_page,
    select_option,
)
from .common import go_back, go_forward, goto_url, press_key
from .html_tools import (
    click_selector,
    fill_input,
    get_page_text,
    scroll_to_element,
)
from .screenshot_tools import click, scroll, type_text

__all__ = [
    "check_element",
    # Screenshot tools
    "click",
    # A11y tools
    "click_element",
    # HTML tools
    "click_selector",
    "fill_element",
    "fill_input",
    "get_page_text",
    "go_back",
    "go_forward",
    # Common tools
    "goto_url",
    "press_key",
    "scroll",
    "scroll_page",
    "scroll_to_element",
    "select_option",
    "type_text",
]

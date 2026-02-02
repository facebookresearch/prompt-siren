# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Browser interaction tools.

Tools are split into:
- common: shared tools (navigation, keyboard)
- screenshot_tools: coordinate-based tools used by the browser dataset
"""

from .common import go_back, go_forward, goto_url, press_key
from .screenshot_tools import click, scroll, type_text

__all__ = [
    # Screenshot tools
    "click",
    "go_back",
    "go_forward",
    # Common tools
    "goto_url",
    "press_key",
    "scroll",
    "type_text",
]

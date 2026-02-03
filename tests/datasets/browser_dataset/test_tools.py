# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for browser tools (common + screenshot tools)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prompt_siren.datasets.browser_dataset.tools.common import (
    go_back,
    go_forward,
    goto_url,
    press_key,
)
from prompt_siren.datasets.browser_dataset.tools.screenshot_tools import (
    click,
    scroll,
    type_text,
)
from prompt_siren.environments.browser_env import BrowserEnvState
from pydantic_ai import RunContext

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Create a mock RunContext[BrowserEnvState] with a mocked Playwright page."""
    page = MagicMock()
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.move = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()
    page.evaluate = AsyncMock()
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()

    state = MagicMock(spec=BrowserEnvState)
    state.page = page

    ctx = MagicMock(spec=RunContext)
    ctx.deps = state
    return ctx


class TestScreenshotTools:
    async def test_click_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await click(mock_ctx, 10, 20)

        page.mouse.click.assert_awaited_once_with(10, 20, button="left")
        assert result is page

    async def test_scroll_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await scroll(mock_ctx, 10, 20, scroll_x=0, scroll_y=250)

        page.mouse.move.assert_awaited_once_with(10, 20)
        page.evaluate.assert_awaited_once_with("window.scrollBy(0, 250)")
        assert result is page

    async def test_type_text_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await type_text(mock_ctx, "hello")

        page.keyboard.type.assert_awaited_once_with("hello")
        assert result is page


class TestCommonTools:
    async def test_press_key_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await press_key(mock_ctx, "Enter")

        page.keyboard.press.assert_awaited_once_with("Enter")
        assert result is page

    async def test_goto_url_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await goto_url(mock_ctx, "http://example.com")

        page.goto.assert_awaited_once_with("http://example.com", timeout=30000)
        assert result is page

    async def test_go_back_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await go_back(mock_ctx)

        page.go_back.assert_awaited_once_with()
        assert result is page

    async def test_go_forward_returns_page(self, mock_ctx: MagicMock) -> None:
        page = mock_ctx.deps.page
        result = await go_forward(mock_ctx)

        page.go_forward.assert_awaited_once_with()
        assert result is page

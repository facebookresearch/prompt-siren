# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for browser tools (a11y_tools, html_tools, screenshot_tools, and common)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeout
from prompt_siren.datasets.browser_dataset.tools.a11y_tools import (
    _truncate as a11y_truncate,
    check_element,
    click_element,
    fill_element,
    scroll_page,
    select_option,
)
from prompt_siren.datasets.browser_dataset.tools.common import (
    go_back,
    go_forward,
    goto_url,
    press_key,
)
from prompt_siren.datasets.browser_dataset.tools.html_tools import (
    _truncate as html_truncate,
    click_selector,
    fill_input,
    get_page_text,
    MAX_PAGE_TEXT_LENGTH,
    scroll_to_element,
)
from prompt_siren.datasets.browser_dataset.tools.screenshot_tools import (
    _truncate as screenshot_truncate,
    click,
    scroll,
    type_text,
)
from prompt_siren.environments.browser_env import BrowserEnvState
from pydantic_ai import RunContext

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_env_state() -> MagicMock:
    """Create a mock BrowserEnvState."""
    state = MagicMock(spec=BrowserEnvState)
    state.page = MagicMock()
    return state


@pytest.fixture
def mock_ctx(mock_env_state: MagicMock) -> MagicMock:
    """Create a mock RunContext."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_env_state
    return ctx


class TestTruncate:
    """Tests for _truncate helper function."""

    def test_returns_unchanged_when_shorter_than_max(self):
        """Test that short text is returned unchanged."""
        assert a11y_truncate("hello", max_len=50) == "hello"
        assert html_truncate("hello", max_len=50) == "hello"
        assert screenshot_truncate("hello", max_len=50) == "hello"

    def test_returns_unchanged_when_exactly_max_length(self):
        """Test that text exactly at max length is returned unchanged."""
        text = "x" * 50
        assert a11y_truncate(text, max_len=50) == text
        assert html_truncate(text, max_len=50) == text
        assert screenshot_truncate(text, max_len=50) == text

    def test_truncates_with_ellipsis_when_longer(self):
        """Test that text longer than max is truncated with ellipsis."""
        text = "x" * 60
        result = a11y_truncate(text, max_len=50)
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")

    def test_empty_string(self):
        """Test that empty string is returned unchanged."""
        assert a11y_truncate("", max_len=50) == ""
        assert html_truncate("", max_len=50) == ""
        assert screenshot_truncate("", max_len=50) == ""


class TestA11yClickElement:
    """Tests for click_element function."""

    async def test_successful_click_with_name(self, mock_ctx: MagicMock):
        """Test clicking an element with role and name."""
        mock_locator = MagicMock()
        mock_locator.click = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await click_element(mock_ctx, "button", "Submit")

        mock_ctx.deps.page.get_by_role.assert_called_once_with("button", name="Submit")
        mock_locator.click.assert_called_once_with(timeout=5000)
        assert "Clicked button with name 'Submit'" in result

    async def test_successful_click_without_name(self, mock_ctx: MagicMock):
        """Test clicking an element by role only."""
        mock_locator = MagicMock()
        mock_locator.click = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await click_element(mock_ctx, "button")

        mock_ctx.deps.page.get_by_role.assert_called_once_with("button")
        assert "Clicked button" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_locator = MagicMock()
        mock_locator.click = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        with pytest.raises(PlaywrightTimeout):
            await click_element(mock_ctx, "button", "Submit")


class TestA11yFillElement:
    """Tests for fill_element function."""

    async def test_successful_fill(self, mock_ctx: MagicMock):
        """Test filling a form element."""
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await fill_element(mock_ctx, "textbox", "test value", "Username")

        mock_locator.fill.assert_called_once_with("test value", timeout=5000)
        assert "Filled textbox 'Username'" in result
        assert "test value" in result

    async def test_fill_truncates_long_value_in_response(self, mock_ctx: MagicMock):
        """Test that long values are truncated in the response message."""
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        long_value = "x" * 100
        result = await fill_element(mock_ctx, "textbox", long_value)

        assert "..." in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        with pytest.raises(PlaywrightTimeout):
            await fill_element(mock_ctx, "textbox", "value", "Field")


class TestA11ySelectOption:
    """Tests for select_option function."""

    async def test_successful_select(self, mock_ctx: MagicMock):
        """Test selecting an option from combobox."""
        mock_locator = MagicMock()
        mock_locator.select_option = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await select_option(mock_ctx, "Country", "United States")

        mock_ctx.deps.page.get_by_role.assert_called_once_with("combobox", name="Country")
        mock_locator.select_option.assert_called_once_with("United States", timeout=5000)
        assert "Selected 'United States' from 'Country'" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_locator = MagicMock()
        mock_locator.select_option = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        with pytest.raises(PlaywrightTimeout):
            await select_option(mock_ctx, "Country", "United States")


class TestA11yCheckElement:
    """Tests for check_element function."""

    async def test_successful_check(self, mock_ctx: MagicMock):
        """Test checking a checkbox."""
        mock_locator = MagicMock()
        mock_locator.check = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await check_element(mock_ctx, "Terms", checked=True)

        mock_locator.check.assert_called_once_with(timeout=5000)
        assert "Checked checkbox 'Terms'" in result

    async def test_successful_uncheck(self, mock_ctx: MagicMock):
        """Test unchecking a checkbox."""
        mock_locator = MagicMock()
        mock_locator.uncheck = AsyncMock()
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        result = await check_element(mock_ctx, "Terms", checked=False)

        mock_locator.uncheck.assert_called_once_with(timeout=5000)
        assert "Unchecked checkbox 'Terms'" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_locator = MagicMock()
        mock_locator.check = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.get_by_role = MagicMock(return_value=mock_locator)

        with pytest.raises(PlaywrightTimeout):
            await check_element(mock_ctx, "Terms")


class TestA11yScrollPage:
    """Tests for scroll_page function."""

    async def test_scroll_down(self, mock_ctx: MagicMock):
        """Test scrolling down."""
        mock_ctx.deps.page.evaluate = AsyncMock()

        result = await scroll_page(mock_ctx, direction="down", amount=500)

        mock_ctx.deps.page.evaluate.assert_called_once_with("window.scrollBy(0, 500)")
        assert "Scrolled down by 500 pixels" in result

    async def test_scroll_up(self, mock_ctx: MagicMock):
        """Test scrolling up."""
        mock_ctx.deps.page.evaluate = AsyncMock()

        result = await scroll_page(mock_ctx, direction="up", amount=300)

        mock_ctx.deps.page.evaluate.assert_called_once_with("window.scrollBy(0, -300)")
        assert "Scrolled up by 300 pixels" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.evaluate = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await scroll_page(mock_ctx, direction="down", amount=500)


class TestHtmlClickSelector:
    """Tests for click_selector function."""

    async def test_successful_click(self, mock_ctx: MagicMock):
        """Test clicking by CSS selector."""
        mock_ctx.deps.page.click = AsyncMock()

        result = await click_selector(mock_ctx, "#submit-btn")

        mock_ctx.deps.page.click.assert_called_once_with("#submit-btn", timeout=5000)
        assert "Clicked element matching selector: #submit-btn" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.click = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await click_selector(mock_ctx, "#submit-btn")


class TestHtmlFillInput:
    """Tests for fill_input function."""

    async def test_successful_fill(self, mock_ctx: MagicMock):
        """Test filling input by CSS selector."""
        mock_ctx.deps.page.fill = AsyncMock()

        result = await fill_input(mock_ctx, "#username", "testuser")

        mock_ctx.deps.page.fill.assert_called_once_with("#username", "testuser", timeout=5000)
        assert "Filled input '#username'" in result
        assert "testuser" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.fill = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await fill_input(mock_ctx, "#username", "testuser")


class TestHtmlGetPageText:
    """Tests for get_page_text function."""

    async def test_returns_page_text(self, mock_ctx: MagicMock):
        """Test getting page text content."""
        mock_ctx.deps.page.inner_text = AsyncMock(return_value="Page content here")

        result = await get_page_text(mock_ctx)

        mock_ctx.deps.page.inner_text.assert_called_once_with("body")
        assert result == "Page content here"

    async def test_truncates_long_text(self, mock_ctx: MagicMock):
        """Test that long text is truncated."""
        long_text = "x" * (MAX_PAGE_TEXT_LENGTH + 1000)
        mock_ctx.deps.page.inner_text = AsyncMock(return_value=long_text)

        result = await get_page_text(mock_ctx)

        assert len(result) < len(long_text)
        assert result.endswith("...[truncated]")


class TestHtmlScrollToElement:
    """Tests for scroll_to_element function."""

    async def test_successful_scroll(self, mock_ctx: MagicMock):
        """Test scrolling to element."""
        mock_locator = MagicMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_ctx.deps.page.locator = MagicMock(return_value=mock_locator)

        result = await scroll_to_element(mock_ctx, "#target")

        mock_ctx.deps.page.locator.assert_called_once_with("#target")
        mock_locator.scroll_into_view_if_needed.assert_called_once_with(timeout=5000)
        assert "Scrolled to '#target'" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_locator = MagicMock()
        mock_locator.scroll_into_view_if_needed = AsyncMock(
            side_effect=PlaywrightTimeout("timeout")
        )
        mock_ctx.deps.page.locator = MagicMock(return_value=mock_locator)

        with pytest.raises(PlaywrightTimeout):
            await scroll_to_element(mock_ctx, "#target")


class TestCommonPressKey:
    """Tests for press_key function."""

    async def test_successful_key_press(self, mock_ctx: MagicMock):
        """Test pressing a key."""
        mock_keyboard = MagicMock()
        mock_keyboard.press = AsyncMock()
        mock_ctx.deps.page.keyboard = mock_keyboard

        result = await press_key(mock_ctx, "Enter")

        mock_keyboard.press.assert_called_once_with("Enter")
        assert "Pressed key: Enter" in result

    async def test_press_special_key(self, mock_ctx: MagicMock):
        """Test pressing a special key like ArrowDown."""
        mock_keyboard = MagicMock()
        mock_keyboard.press = AsyncMock()
        mock_ctx.deps.page.keyboard = mock_keyboard

        result = await press_key(mock_ctx, "ArrowDown")

        mock_keyboard.press.assert_called_once_with("ArrowDown")
        assert "Pressed key: ArrowDown" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_keyboard = MagicMock()
        mock_keyboard.press = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.keyboard = mock_keyboard

        with pytest.raises(PlaywrightTimeout):
            await press_key(mock_ctx, "Enter")


class TestCommonGotoUrl:
    """Tests for goto_url function."""

    async def test_successful_navigation(self, mock_ctx: MagicMock):
        """Test navigating to a URL."""
        mock_ctx.deps.page.goto = AsyncMock()
        mock_ctx.deps.page.url = "https://example.com/page"

        result = await goto_url(mock_ctx, "https://example.com/page")

        mock_ctx.deps.page.goto.assert_called_once_with("https://example.com/page", timeout=30000)
        assert "Navigated to: https://example.com/page" in result

    async def test_navigation_returns_final_url(self, mock_ctx: MagicMock):
        """Test that navigation returns the final URL (after redirects)."""
        mock_ctx.deps.page.goto = AsyncMock()
        # Simulate redirect - page.url differs from requested URL
        mock_ctx.deps.page.url = "https://example.com/redirected"

        result = await goto_url(mock_ctx, "https://example.com/original")

        assert "https://example.com/redirected" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.goto = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await goto_url(mock_ctx, "https://slow.example.com")


class TestCommonGoBack:
    """Tests for go_back function."""

    async def test_successful_go_back(self, mock_ctx: MagicMock):
        """Test going back in history."""
        mock_ctx.deps.page.go_back = AsyncMock()
        mock_ctx.deps.page.url = "https://example.com/previous"

        result = await go_back(mock_ctx)

        mock_ctx.deps.page.go_back.assert_called_once()
        assert "Went back" in result
        assert "https://example.com/previous" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.go_back = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await go_back(mock_ctx)


class TestCommonGoForward:
    """Tests for go_forward function."""

    async def test_successful_go_forward(self, mock_ctx: MagicMock):
        """Test going forward in history."""
        mock_ctx.deps.page.go_forward = AsyncMock()
        mock_ctx.deps.page.url = "https://example.com/next"

        result = await go_forward(mock_ctx)

        mock_ctx.deps.page.go_forward.assert_called_once()
        assert "Went forward" in result
        assert "https://example.com/next" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_ctx.deps.page.go_forward = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        with pytest.raises(PlaywrightTimeout):
            await go_forward(mock_ctx)


class TestScreenshotClick:
    """Tests for screenshot click function."""

    async def test_successful_click(self, mock_ctx: MagicMock):
        """Test clicking at coordinates."""
        mock_mouse = MagicMock()
        mock_mouse.click = AsyncMock()
        mock_ctx.deps.page.mouse = mock_mouse

        result = await click(mock_ctx, 100, 200)

        mock_mouse.click.assert_called_once_with(100, 200, button="left")
        assert "Clicked at (100, 200)" in result

    async def test_click_with_right_button(self, mock_ctx: MagicMock):
        """Test right-clicking at coordinates."""
        mock_mouse = MagicMock()
        mock_mouse.click = AsyncMock()
        mock_ctx.deps.page.mouse = mock_mouse

        result = await click(mock_ctx, 100, 200, button="right")

        mock_mouse.click.assert_called_once_with(100, 200, button="right")
        assert "right button" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_mouse = MagicMock()
        mock_mouse.click = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.mouse = mock_mouse

        with pytest.raises(PlaywrightTimeout):
            await click(mock_ctx, 100, 200)


class TestScreenshotScroll:
    """Tests for screenshot scroll function."""

    async def test_successful_scroll(self, mock_ctx: MagicMock):
        """Test scrolling from coordinates."""
        mock_mouse = MagicMock()
        mock_mouse.move = AsyncMock()
        mock_ctx.deps.page.mouse = mock_mouse
        mock_ctx.deps.page.evaluate = AsyncMock()

        result = await scroll(mock_ctx, 100, 200, 0, 500)

        mock_mouse.move.assert_called_once_with(100, 200)
        mock_ctx.deps.page.evaluate.assert_called_once_with("window.scrollBy(0, 500)")
        assert "Scrolled by (0, 500)" in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_mouse = MagicMock()
        mock_mouse.move = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.mouse = mock_mouse

        with pytest.raises(PlaywrightTimeout):
            await scroll(mock_ctx, 100, 200, 0, 500)


class TestScreenshotTypeText:
    """Tests for screenshot type_text function."""

    async def test_successful_type(self, mock_ctx: MagicMock):
        """Test typing text."""
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock()
        mock_ctx.deps.page.keyboard = mock_keyboard

        result = await type_text(mock_ctx, "hello world")

        mock_keyboard.type.assert_called_once_with("hello world")
        assert "Typed: hello world" in result

    async def test_type_truncates_long_text_in_response(self, mock_ctx: MagicMock):
        """Test that long text is truncated in the response message."""
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock()
        mock_ctx.deps.page.keyboard = mock_keyboard

        long_text = "x" * 100
        result = await type_text(mock_ctx, long_text)

        assert "..." in result

    async def test_timeout_raises_exception(self, mock_ctx: MagicMock):
        """Test that timeout raises PlaywrightTimeout."""
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock(side_effect=PlaywrightTimeout("timeout"))
        mock_ctx.deps.page.keyboard = mock_keyboard

        with pytest.raises(PlaywrightTimeout):
            await type_text(mock_ctx, "hello")

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for BrowserEnvironment with real Docker containers.

These tests verify the browser environment works correctly with real containers.
They require Docker to be running and will pull/start containers.

Run with: pytest -vx -m docker_integration tests/datasets/browser_dataset/integration/
Skip with: pytest -vx -m "not docker_integration"
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_siren.datasets.browser_dataset import (
    BrowserDatasetConfig,
    create_screenshot_browser_dataset,
    ScreenshotBrowserDataset,
)
from prompt_siren.sandbox_managers.docker.manager import DockerSandboxManager
from prompt_siren.sandbox_managers.sandbox_task_setup import SandboxTaskSetup

pytestmark = [pytest.mark.anyio, pytest.mark.docker_integration]


class TestBrowserContainerIntegration:
    """Integration tests for browser containers with real Docker."""

    async def test_browser_container_starts(
        self,
        docker_sandbox_manager: DockerSandboxManager,
        browser_task_setup: SandboxTaskSetup,
    ) -> None:
        """Test that browser container starts and CDP port is accessible."""
        async with docker_sandbox_manager.setup_batch([browser_task_setup]):
            async with docker_sandbox_manager.setup_task(browser_task_setup) as sandbox_state:
                # Container should be running
                assert sandbox_state.agent_container_id is not None

                # Give browser a moment to start
                await asyncio.sleep(2)

                # Execute a command in the container to verify it's running
                result = await docker_sandbox_manager.exec(
                    sandbox_state.agent_container_id,
                    ["echo", "Browser container is running"],
                )
                assert result.exit_code == 0
                assert result.stdout is not None
                assert "Browser container is running" in result.stdout

    async def test_browser_cdp_connection(
        self,
        docker_sandbox_manager: DockerSandboxManager,
        browser_task_setup: SandboxTaskSetup,
    ) -> None:
        """Test that we can connect to browser via CDP using Playwright."""
        pytest.importorskip("playwright")
        from playwright.async_api import async_playwright

        async with docker_sandbox_manager.setup_batch([browser_task_setup]):
            async with docker_sandbox_manager.setup_task(browser_task_setup):
                # Give browser time to start
                await asyncio.sleep(3)

                # Connect via CDP (use 127.0.0.1 to force IPv4, container doesn't bind IPv6)
                cdp_endpoint = "http://127.0.0.1:9222"

                async with async_playwright() as pw:
                    # Connect to the browser
                    browser = await pw.chromium.connect_over_cdp(cdp_endpoint)
                    try:
                        assert browser.is_connected()

                        # Create a page and navigate
                        page = await browser.new_page()
                        await page.goto("about:blank")

                        # Verify page works
                        title = await page.title()
                        assert title == ""  # about:blank has empty title

                        await page.close()
                    finally:
                        await browser.close()

    async def test_browser_page_screenshot(
        self,
        docker_sandbox_manager: DockerSandboxManager,
        browser_task_setup: SandboxTaskSetup,
    ) -> None:
        """Test that we can take screenshots from browser pages."""
        pytest.importorskip("playwright")
        from playwright.async_api import async_playwright

        async with docker_sandbox_manager.setup_batch([browser_task_setup]):
            async with docker_sandbox_manager.setup_task(browser_task_setup):
                await asyncio.sleep(3)

                async with async_playwright() as pw:
                    browser = await pw.chromium.connect_over_cdp("http://127.0.0.1:9222")
                    try:
                        page = await browser.new_page()

                        # Set some content
                        await page.set_content("<html><body><h1>Test Page</h1></body></html>")

                        # Take screenshot
                        screenshot = await page.screenshot()
                        assert screenshot is not None
                        assert len(screenshot) > 0
                        # PNG magic bytes
                        assert screenshot[:8] == b"\x89PNG\r\n\x1a\n"

                        await page.close()
                    finally:
                        await browser.close()


class TestBrowserDatasetIntegration:
    """Integration tests for browser dataset with real Docker."""

    @pytest.fixture
    def dataset(self, docker_sandbox_manager: DockerSandboxManager) -> ScreenshotBrowserDataset:
        """Create a browser dataset with real Docker sandbox manager."""
        config = BrowserDatasetConfig()
        return create_screenshot_browser_dataset(config, sandbox_manager=docker_sandbox_manager)

    async def test_dataset_batch_context(
        self,
        dataset: ScreenshotBrowserDataset,
        docker_sandbox_manager: DockerSandboxManager,
    ) -> None:
        """Test that dataset batch context works with real Docker."""
        # Get a single task to test with
        tasks = dataset.benign_tasks[:1]

        # This should prepare images via setup_batch
        async with dataset.environment.create_batch_context(tasks):
            # Batch context should be active
            pass  # Success if no exception

    def test_dataset_has_tasks(self, dataset: ScreenshotBrowserDataset) -> None:
        """Test that dataset has tasks configured."""
        assert len(dataset.benign_tasks) > 0
        assert len(dataset.malicious_tasks) > 0
        assert len(dataset.task_couples) > 0

    def test_dataset_environment_configured(self, dataset: ScreenshotBrowserDataset) -> None:
        """Test that dataset environment is properly configured."""
        env = dataset.environment
        assert env.name == "browser-screenshot"
        assert len(env.all_injection_ids) > 0


# Note: Tests that don't require Docker are in test_browser_dataset.py
# The following classes are kept here for Docker-specific integration testing

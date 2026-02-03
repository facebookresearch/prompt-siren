# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for Docker task sandbox contexts."""

import asyncio
from unittest.mock import patch

import pytest
from prompt_siren.sandbox_managers.docker.contexts import (
    _background_tasks,
    _fire_and_forget,
)

pytestmark = pytest.mark.anyio


class TestFireAndForget:
    """Tests for _fire_and_forget function."""

    async def test_completed_tasks_are_discarded(self):
        """Test that completed tasks are removed from tracking set."""

        async def quick_task():
            return "done"

        initial_count = len(_background_tasks)
        _fire_and_forget(quick_task())

        # Wait for task to complete
        await asyncio.sleep(0.1)

        # Task should be removed from tracking
        assert len(_background_tasks) == initial_count

    async def test_failed_tasks_are_logged_and_discarded(self):
        """Test that failed tasks are logged and removed from tracking."""

        async def failing_task():
            raise ValueError("intentional failure")

        initial_count = len(_background_tasks)

        with patch("prompt_siren.sandbox_managers.docker.contexts.logger") as mock_logger:
            _fire_and_forget(failing_task())
            await asyncio.sleep(0.1)

            # Task should be removed from tracking
            assert len(_background_tasks) == initial_count
            # Error should be logged at warning level
            mock_logger.warning.assert_called_once()

    async def test_cancelled_tasks_are_handled_gracefully(self):
        """Test that cancelled tasks don't cause warnings."""

        async def slow_task():
            await asyncio.sleep(10)

        initial_count = len(_background_tasks)

        with patch("prompt_siren.sandbox_managers.docker.contexts.logger") as mock_logger:
            _fire_and_forget(slow_task())
            await asyncio.sleep(0.01)

            # Find and cancel the task
            for task in list(_background_tasks):
                task.cancel()

            await asyncio.sleep(0.1)

            # Task should be removed, no warning logged for cancellation
            assert len(_background_tasks) == initial_count
            mock_logger.warning.assert_not_called()

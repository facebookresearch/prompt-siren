# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Task couples for browser-based dataset.

Couples are defined manually based on semantic compatibility between
benign and malicious tasks. Each couple pairs a benign task with a
malicious task that could realistically be encountered during that
benign task's execution.

NOTE: PR1 ships infrastructure only. PR2 will add all task couples.
"""

from ...environments.browser_env import BrowserEnvState
from ...tasks import TaskCouple

# Empty for PR1 - couples will be added in PR2 along with malicious tasks
TASK_COUPLES: list[TaskCouple[BrowserEnvState]] = []

__all__ = [
    "TASK_COUPLES",
]

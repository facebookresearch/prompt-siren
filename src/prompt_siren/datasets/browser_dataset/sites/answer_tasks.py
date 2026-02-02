# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Benign task definitions for Apache Answer site.

NOTE: PR1 includes only Gitea site. PR2 will add Answer site and all its tasks.
"""

from ....environments.browser_env import BrowserEnvState
from ....tasks import BenignTask

# Empty for PR1 - Answer tasks will be added in PR2
ANSWER_BENIGN_TASKS: list[BenignTask[BrowserEnvState]] = []

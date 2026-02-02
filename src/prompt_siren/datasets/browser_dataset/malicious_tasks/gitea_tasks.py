# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Gitea-specific malicious tasks.

These tasks test if agents can be tricked via injections in Gitea content
(issues, PRs, code comments, etc.) to perform malicious actions.

NOTE: PR1 ships infrastructure only. PR2 will add all malicious tasks.
"""

from ....environments.browser_env import BrowserEnvState
from ....tasks import MaliciousTask

GITEA_MALICIOUS_TASKS: list[MaliciousTask[BrowserEnvState]] = []

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Cross-site malicious tasks.

These tasks test trust boundary violations where agents are tricked to
perform actions across multiple sites (e.g., copying sensitive data from
one site to another).

NOTE: PR1 ships infrastructure only. PR2 will add all cross-site malicious tasks.
"""

from ....environments.browser_env import BrowserEnvState
from ....tasks import MaliciousTask

CROSS_SITE_MALICIOUS_TASKS: list[MaliciousTask[BrowserEnvState]] = []

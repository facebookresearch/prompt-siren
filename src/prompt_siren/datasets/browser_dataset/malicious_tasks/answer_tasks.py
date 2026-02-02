# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Apache Answer-specific malicious tasks.

These tasks test if agents can be tricked via injections in Q&A content
(questions, answers, comments) to perform malicious actions.

NOTE: PR1 ships infrastructure only. PR2 will add all malicious tasks.
"""

from ....environments.browser_env import BrowserEnvState
from ....tasks import MaliciousTask

ANSWER_MALICIOUS_TASKS: list[MaliciousTask[BrowserEnvState]] = []

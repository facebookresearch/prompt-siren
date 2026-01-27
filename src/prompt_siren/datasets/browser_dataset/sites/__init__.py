# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Site implementations for browser-based dataset."""

from .answer_tasks import ANSWER_BENIGN_TASKS
from .gitea_tasks import GITEA_BENIGN_TASKS

__all__ = [
    "ANSWER_BENIGN_TASKS",
    "GITEA_BENIGN_TASKS",
]

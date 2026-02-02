# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Site implementations for browser-based dataset.

NOTE: PR1 includes only Gitea. PR2 will add Answer and WikiJS.
"""

from .gitea_tasks import GITEA_BENIGN_TASKS

__all__ = [
    "GITEA_BENIGN_TASKS",
]

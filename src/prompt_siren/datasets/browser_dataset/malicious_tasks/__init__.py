# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Malicious tasks for browser-based dataset.

NOTE: PR1 includes only Gitea. PR2 will add Answer and cross-site malicious tasks.
"""

from .gitea_tasks import GITEA_MALICIOUS_TASKS

MALICIOUS_TASKS = GITEA_MALICIOUS_TASKS

__all__ = [
    "GITEA_MALICIOUS_TASKS",
    "MALICIOUS_TASKS",
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Malicious tasks for browser-based dataset."""

from .answer_tasks import ANSWER_MALICIOUS_TASKS
from .cross_site_tasks import CROSS_SITE_MALICIOUS_TASKS
from .gitea_tasks import GITEA_MALICIOUS_TASKS

MALICIOUS_TASKS = GITEA_MALICIOUS_TASKS + ANSWER_MALICIOUS_TASKS + CROSS_SITE_MALICIOUS_TASKS

__all__ = [
    "ANSWER_MALICIOUS_TASKS",
    "CROSS_SITE_MALICIOUS_TASKS",
    "GITEA_MALICIOUS_TASKS",
    "MALICIOUS_TASKS",
]

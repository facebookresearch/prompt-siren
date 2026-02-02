# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Injection system for browser-based dataset.

This module provides injection vector IDs and utilities for injecting attack
content into web pages.
"""

import logging
from collections.abc import Sequence

from ...types import InjectionVectorID

logger = logging.getLogger(__name__)

# Pre-defined injection vector IDs for Gitea (Git Forge)
GITEA_VECTORS: list[InjectionVectorID] = [
    "gitea_issue_body",
    "gitea_issue_comment",
    "gitea_db_issue",
    "gitea_perf_issue",
    "gitea_infra_issue",
    "gitea_pr_description",
    "gitea_pr_comment",
    "gitea_code_comment",
    "gitea_readme_content",
    "gitea_frontend_readme",
    "gitea_infra_readme",
    "gitea_wiki_content",
    "gitea_commit_message",
]

# Pre-defined injection vector IDs for Apache Answer (Q&A Platform)
ANSWER_VECTORS: list[InjectionVectorID] = [
    "answer_question_body",
    "answer_answer_content",
    "answer_comment",
    "answer_user_bio",
    "answer_docker_network",
    "answer_api_design",
    "answer_react_cleanup",
    "answer_git_rebase",
    "answer_postgres_perf",
    "answer_k8s_crash",
    "answer_typescript_generic",
]

# Pre-defined injection vector IDs for Wiki.js
WIKIJS_VECTORS: list[InjectionVectorID] = [
    "wikijs_home_content",
    "wikijs_getting_started",
    "wikijs_architecture",
    "wikijs_api_reference",
    "wikijs_deployment",
    "wikijs_runbooks",
    "wikijs_security",
    "wikijs_onboarding",
]

# Pre-defined injection vector IDs for VWA Classifieds
CLASSIFIEDS_VECTORS: list[InjectionVectorID] = [
    "classifieds_listing_description",
    "classifieds_listing_comment",
    "classifieds_user_profile",
    "classifieds_search_result",
]

# All vectors by site
VECTORS_BY_SITE: dict[str, list[InjectionVectorID]] = {
    "gitea": GITEA_VECTORS,
    "answer": ANSWER_VECTORS,
    "wikijs": WIKIJS_VECTORS,
    "classifieds": CLASSIFIEDS_VECTORS,
}

# Combined list of all vectors
ALL_BROWSER_VECTORS: list[InjectionVectorID] = (
    GITEA_VECTORS + ANSWER_VECTORS + WIKIJS_VECTORS + CLASSIFIEDS_VECTORS
)


def get_vectors_for_sites(sites: Sequence[str]) -> list[InjectionVectorID]:
    """Get all injection vector IDs for the specified sites.

    Args:
        sites: List of site names (gitea, answer, wikijs, classifieds)

    Returns:
        Combined list of injection vector IDs for all specified sites

    Raises:
        ValueError: If any site name is not recognized.
    """
    vectors: list[InjectionVectorID] = []
    unknown_sites: list[str] = []

    for site in sites:
        if site in VECTORS_BY_SITE:
            vectors.extend(VECTORS_BY_SITE[site])
        else:
            unknown_sites.append(site)

    if unknown_sites:
        raise ValueError(
            f"Unknown site(s) requested for injection vectors: {unknown_sites}. "
            f"Valid sites: {sorted(VECTORS_BY_SITE.keys())}."
        )

    return vectors

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for browser dataset site models validators."""

import pytest
from prompt_siren.datasets.browser_dataset.sites.models import (
    GiteaIssue,
    GiteaRepository,
    GiteaSeedData,
    GiteaUser,
    WikiPage,
    WikiSeedData,
)
from pydantic import ValidationError


class TestGiteaSeedDataValidator:
    """Tests for GiteaSeedData.check_issue_repos_exist validator."""

    def test_rejects_issue_with_unknown_repo(self) -> None:
        """Issues referencing non-existent repos should raise ValidationError."""
        with pytest.raises(ValidationError, match="unknown repository"):
            GiteaSeedData(
                repositories=[GiteaRepository(name="existing-repo", description="A repo")],
                issues=[GiteaIssue(repo="nonexistent-repo", title="Bug", body="desc")],
            )

    def test_accepts_valid_issue_repo_reference(self) -> None:
        """Issues referencing existing repos should be valid."""
        data = GiteaSeedData(
            repositories=[GiteaRepository(name="my-repo", description="A repo")],
            issues=[GiteaIssue(repo="my-repo", title="Bug", body="desc")],
        )
        assert len(data.issues) == 1
        assert data.issues[0].repo == "my-repo"

    def test_accepts_empty_issues_list(self) -> None:
        """Empty issues list should be valid."""
        data = GiteaSeedData(
            repositories=[GiteaRepository(name="my-repo", description="A repo")],
            issues=[],
        )
        assert len(data.issues) == 0

    def test_accepts_empty_repositories_and_issues(self) -> None:
        """Empty repositories and issues should be valid."""
        data = GiteaSeedData(repositories=[], issues=[])
        assert len(data.repositories) == 0
        assert len(data.issues) == 0

    def test_multiple_issues_referencing_different_repos(self) -> None:
        """Multiple issues can reference different valid repos."""
        data = GiteaSeedData(
            repositories=[
                GiteaRepository(name="repo-a", description="Repo A"),
                GiteaRepository(name="repo-b", description="Repo B"),
            ],
            issues=[
                GiteaIssue(repo="repo-a", title="Issue 1", body="body"),
                GiteaIssue(repo="repo-b", title="Issue 2", body="body"),
                GiteaIssue(repo="repo-a", title="Issue 3", body="body"),
            ],
        )
        assert len(data.issues) == 3

    def test_error_message_includes_available_repos(self) -> None:
        """Error message should list available repositories."""
        with pytest.raises(ValidationError) as exc_info:
            GiteaSeedData(
                repositories=[
                    GiteaRepository(name="alpha", description=""),
                    GiteaRepository(name="beta", description=""),
                ],
                issues=[GiteaIssue(repo="gamma", title="Bug", body="")],
            )
        error_msg = str(exc_info.value)
        assert "alpha" in error_msg
        assert "beta" in error_msg


class TestWikiSeedDataValidator:
    """Tests for WikiSeedData.check_unique_paths validator."""

    def test_rejects_duplicate_paths(self) -> None:
        """Duplicate page paths should raise ValidationError."""
        with pytest.raises(ValidationError, match="Duplicate page paths"):
            WikiSeedData(
                pages=[
                    WikiPage(path="home", title="Home", content="Welcome"),
                    WikiPage(path="home", title="Home 2", content="Another"),
                ]
            )

    def test_accepts_unique_paths(self) -> None:
        """Unique page paths should be valid."""
        data = WikiSeedData(
            pages=[
                WikiPage(path="home", title="Home", content="Welcome"),
                WikiPage(path="about", title="About", content="About us"),
            ]
        )
        assert len(data.pages) == 2

    def test_accepts_empty_pages_list(self) -> None:
        """Empty pages list should be valid."""
        data = WikiSeedData(pages=[])
        assert len(data.pages) == 0

    def test_accepts_single_page(self) -> None:
        """Single page should be valid."""
        data = WikiSeedData(pages=[WikiPage(path="index", title="Index", content="Main page")])
        assert len(data.pages) == 1

    def test_multiple_duplicates_reported(self) -> None:
        """Error message should include all duplicate paths."""
        with pytest.raises(ValidationError) as exc_info:
            WikiSeedData(
                pages=[
                    WikiPage(path="home", title="Home", content=""),
                    WikiPage(path="about", title="About", content=""),
                    WikiPage(path="home", title="Home 2", content=""),
                    WikiPage(path="about", title="About 2", content=""),
                ]
            )
        error_msg = str(exc_info.value)
        assert "home" in error_msg
        assert "about" in error_msg


class TestGiteaUserValidation:
    """Tests for GiteaUser field validation."""

    def test_requires_valid_email(self) -> None:
        """GiteaUser should require a valid email address."""
        with pytest.raises(ValidationError, match="email"):
            GiteaUser(username="user1", email="not-an-email")

    def test_accepts_valid_email(self) -> None:
        """GiteaUser should accept valid email addresses."""
        user = GiteaUser(username="user1", email="user@example.com")
        assert user.email == "user@example.com"

    def test_requires_non_empty_username(self) -> None:
        """GiteaUser should require non-empty username."""
        with pytest.raises(ValidationError, match="string_too_short"):
            GiteaUser(username="", email="user@example.com")

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pydantic models for seed data structures used by Gitea, Answer, and Wiki.js sites."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, model_validator


# Gitea models
class GiteaUser(BaseModel):
    """A Gitea user to create."""

    username: str = Field(..., min_length=1)
    email: EmailStr
    full_name: str = ""


class GiteaRepository(BaseModel):
    """A Gitea repository with files."""

    name: str = Field(..., min_length=1)
    description: str
    files: dict[str, str] = {}


class GiteaIssue(BaseModel):
    """A Gitea issue with comments."""

    repo: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    body: str
    comments: list[str] = []


class GiteaSeedData(BaseModel):
    """Complete Gitea seed data."""

    users: list[GiteaUser] = []
    repositories: list[GiteaRepository] = []
    issues: list[GiteaIssue] = []

    @model_validator(mode="after")
    def check_issue_repos_exist(self) -> GiteaSeedData:
        """Ensure all issues reference existing repositories."""
        repo_names = {r.name for r in self.repositories}
        for issue in self.issues:
            if issue.repo not in repo_names:
                raise ValueError(
                    f"Issue '{issue.title}' references unknown repository: {issue.repo}. "
                    f"Available repositories: {sorted(repo_names)}"
                )
        return self


# Answer models
class AnswerAnswer(BaseModel):
    """An answer to a question."""

    content: str = Field(..., min_length=1)
    comments: list[str] = []


class AnswerQuestion(BaseModel):
    """A question with answers."""

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tags: list[str] = []
    answers: list[AnswerAnswer] = []


class AnswerSeedData(BaseModel):
    """Complete Answer seed data."""

    user_bio: str = ""
    questions: list[AnswerQuestion] = []


# Wiki.js models
class WikiPage(BaseModel):
    """A Wiki.js page."""

    path: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str


class WikiSeedData(BaseModel):
    """Complete Wiki.js seed data."""

    pages: list[WikiPage] = []

    @model_validator(mode="after")
    def check_unique_paths(self) -> WikiSeedData:
        """Ensure all page paths are unique."""
        paths = [p.path for p in self.pages]
        if len(paths) != len(set(paths)):
            duplicates = [p for p in paths if paths.count(p) > 1]
            raise ValueError(f"Duplicate page paths found: {sorted(set(duplicates))}")
        return self

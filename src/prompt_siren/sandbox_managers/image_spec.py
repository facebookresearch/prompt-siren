# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import TypeAlias

from pydantic import BaseModel, Field

ImageTag: TypeAlias = str


class PullImageSpec(BaseModel):
    """Specification to pull a pre-built image from a registry.

    Examples:
        PullImageSpec(tag="python:3.12")
        PullImageSpec(tag="alpine:latest")
    """

    tag: ImageTag = Field(description="Image identifier (e.g., 'python:3.12', 'alpine:latest')")


class BuildImageSpec(BaseModel):
    """Specification to build an image from a Dockerfile.

    Examples:
        BuildImageSpec(
            context_path="./docker/my-env",
            tag="my-env:latest"
        )

        BuildImageSpec(
            context_path="./docker/python-app",
            dockerfile_path="Dockerfile.dev",
            tag="python-app:dev",
            build_args={"PYTHON_VERSION": "3.12", "ENV": "development"}
        )
    """

    context_path: str = Field(description="Path to the build context directory")
    dockerfile_path: str | None = Field(
        default=None,
        description="Path to Dockerfile relative to context_path. Defaults to 'Dockerfile'",
    )
    tag: ImageTag = Field(description="Tag for the built image (e.g., 'my-env:latest')")
    build_args: dict[str, str] | None = Field(
        default=None, description="Build-time variables for Docker build"
    )


class BuildStage(BaseModel):
    """Represents a single stage in a multi-stage Docker build.

    Examples:
        BuildStage(
            tag="base:latest",
            context_path="./docker/base",
            parent_tag=None,
            cache_key="base_abc123"
        )

        BuildStage(
            tag="env:latest",
            context_path="./docker/env",
            parent_tag="base:latest",
            cache_key="env_def456"
        )
    """

    tag: ImageTag = Field(description="Tag for this stage's image")
    context_path: str = Field(description="Build context for this stage")
    dockerfile_path: str | None = Field(
        default=None,
        description="Path to Dockerfile relative to context_path. Defaults to 'Dockerfile'",
    )
    build_args: dict[str, str] | None = Field(
        default=None, description="Build-time variables for this stage"
    )
    parent_tag: ImageTag | None = Field(
        default=None, description="Tag of parent image (FROM clause). None for base images."
    )
    cache_key: str | None = Field(
        default=None,
        description="Cache key for reusing images. Stages with same cache_key can be reused.",
    )


class MultiStageBuildImageSpec(BaseModel):
    """Specification for multi-stage Docker builds with intermediate caching.

    Enables efficient builds where intermediate stages (e.g., base, environment)
    can be cached and reused across multiple final images.

    Examples:
        MultiStageBuildImageSpec(
            stages=[
                BuildStage(tag="base:latest", context_path="./base", cache_key="base_hash"),
                BuildStage(tag="env:latest", context_path="./env", parent_tag="base:latest", cache_key="env_hash"),
                BuildStage(tag="app:latest", context_path="./app", parent_tag="env:latest")
            ],
            final_tag="app:latest"
        )
    """

    stages: list[BuildStage] = Field(
        description="Ordered list of build stages. Each stage builds on the previous one."
    )
    final_tag: ImageTag = Field(
        description="Tag of the final image to use for containers (typically the last stage's tag)"
    )

    @property
    def tag(self) -> ImageTag:
        """Final image tag (for compatibility with setup_task)."""
        return self.final_tag


ImageSpec = PullImageSpec | BuildImageSpec | MultiStageBuildImageSpec

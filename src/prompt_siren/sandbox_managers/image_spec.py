# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections.abc import Awaitable, Callable
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator, StringConstraints
from typing_extensions import Self

ImageTag: TypeAlias = str

# Type alias for seeder functions (async functions with no parameters)
SeederFn: TypeAlias = Callable[[], Awaitable[None]]


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

        BuildImageSpec(
            context_path="./sites/gitea",
            tag="prompt-siren/gitea:latest",
            seeder=gitea_seeding.generate_seed  # Pre-build seeding function
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context_path: str = Field(description="Path to the build context directory")
    dockerfile_path: str | None = Field(
        default=None,
        description="Path to Dockerfile relative to context_path. Defaults to 'Dockerfile'",
    )
    tag: ImageTag = Field(description="Tag for the built image (e.g., 'my-env:latest')")
    build_args: dict[str, str] | None = Field(
        default=None, description="Build-time variables for Docker build"
    )
    seeder: SeederFn | None = Field(
        default=None,
        description="Async function to run before building (e.g., seed generation)",
        exclude=True,  # Don't serialize
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

    @model_validator(mode="after")
    def _validate_final_tag_matches_last_stage(self) -> Self:
        if not self.stages:
            raise ValueError("stages must not be empty")
        if self.final_tag != self.stages[-1].tag:
            raise ValueError(
                f"final_tag must match the last stage's tag: "
                f"final_tag={self.final_tag!r}, last stage tag={self.stages[-1].tag!r}"
            )
        return self

    @property
    def tag(self) -> ImageTag:
        """Alias for final_tag, providing a uniform .tag interface across all ImageBuildSpec types."""
        return self.final_tag


class DerivedImageSpec(BaseModel):
    """Specification for an image derived from a base image with modifications.

    Used for creating images where additional Dockerfile instructions
    are applied on top of a pre-built base image.

    Examples:
        DerivedImageSpec(
            base_image_tag="myproject-base:latest",
            dockerfile_extra="RUN pip install extra-package",
            tag="myproject-derived:latest"
        )
    """

    base_image_tag: ImageTag = Field(
        description="Tag of the base image to derive from (must be built first)"
    )
    dockerfile_extra: Annotated[str, StringConstraints(min_length=1)] = Field(
        description="Additional Dockerfile instructions to append"
    )
    tag: ImageTag = Field(description="Tag for the derived image")

    @model_validator(mode="after")
    def _validate_tag_differs_from_base(self) -> Self:
        if self.tag == self.base_image_tag:
            raise ValueError(f"'tag' must differ from 'base_image_tag', got '{self.tag}' for both")
        return self


# Type alias for specs that require building (excludes PullImageSpec)
ImageBuildSpec = BuildImageSpec | MultiStageBuildImageSpec | DerivedImageSpec

# Type alias for all image specs (includes PullImageSpec)
ImageSpec = PullImageSpec | ImageBuildSpec

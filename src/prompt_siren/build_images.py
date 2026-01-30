# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Script for building Docker images.

Pre-builds all Docker images needed for running datasets that use Docker
containers. Uses the ImageBuildableDataset protocol to discover which images
each dataset needs, then builds them with proper dependency ordering.

Usage:
    # Build images for a specific dataset
    prompt-siren-build-images --dataset swebench [OPTIONS]

    # Build images for all datasets
    prompt-siren-build-images --all-datasets [OPTIONS]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import click
from pydantic import BaseModel

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from .datasets import (
    dataset_registry,
    get_dataset_config_class,
    get_datasets_with_image_specs,
    get_image_build_specs,
)
from .datasets.registry import ImageBuildableDataset
from .sandbox_managers.docker.manager import create_docker_client_from_config
from .sandbox_managers.docker.plugins import AbstractDockerClient
from .sandbox_managers.docker.plugins.errors import DockerClientError
from .sandbox_managers.image_spec import (
    BuildImageSpec,
    DerivedImageSpec,
    ImageBuildSpec,
    MultiStageBuildImageSpec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildError:
    """Failed build result."""

    image_tag: str
    error: Exception
    phase: Literal["build", "push"] = "build"


def _handle_build_failures(
    failed_builds: list[BuildError],
) -> None:
    """Handle logging and raising errors for failed builds.

    Args:
        failed_builds: List of BuildError instances

    Raises:
        ExceptionGroup: If failed_builds is not empty
    """
    if not failed_builds:
        return

    for build_error in failed_builds:
        logger.error(
            f"Build failed for image {build_error.image_tag}",
            exc_info=build_error.error,
        )

    image_tags = ", ".join(e.image_tag for e in failed_builds)
    raise ExceptionGroup(
        f"{len(failed_builds)} image build(s) failed: {image_tags}",
        [error.error for error in failed_builds],
    )


class ImageBuilder:
    """Handles building Docker images for datasets."""

    def __init__(
        self,
        docker_client: AbstractDockerClient,
        cache_dir: Path,
        rebuild_existing: bool = False,
        registry: str | None = None,
    ):
        """Initialize the image builder.

        Args:
            docker_client: Docker client for building images
            cache_dir: Directory for caching build contexts
            rebuild_existing: If True, delete and rebuild existing images.
                If False (default), skip building images that already exist.
            registry: Optional registry to tag and push images to
        """
        self._docker = docker_client
        self._cache_dir = cache_dir
        self._rebuild_existing = rebuild_existing
        self._registry = registry
        # Track built images to avoid duplicates
        self._built_images: set[str] = set()
        # Disable registry checks after auth failures to avoid log spam
        self._registry_check_disabled = False

    async def image_exists(self, tag: str) -> bool:
        """Check if an image already exists locally.

        Args:
            tag: Image tag to check

        Returns:
            True if image exists, False otherwise
        """
        try:
            await self._docker.inspect_image(tag)
            return True
        except DockerClientError as e:
            if e.status == 404:
                return False
            logger.error(f"Docker error while checking if image '{tag}' exists: {e}")
            raise

    async def delete_image(self, tag: str) -> None:
        """Delete an image by tag.

        Args:
            tag: Image tag to delete
        """
        await self._docker.delete_image(tag, force=True)

    async def _prepare_for_build(self, tag: str) -> bool:
        """Decide whether to build an image, with side effects.

        Checks whether the image was already built in this session or already
        exists locally. When ``rebuild_existing`` is True and the image exists,
        it is **deleted** so that the caller can rebuild it. The tag is added
        to ``_built_images`` when the image already exists and is being kept
        (i.e. skipped).

        Args:
            tag: Image tag to check

        Returns:
            True if the caller should proceed with the build, False to skip
        """
        if tag in self._built_images:
            return False

        if await self.image_exists(tag):
            if self._rebuild_existing:
                await self.delete_image(tag)
                return True
            self._built_images.add(tag)
            return False

        return True

    async def push_to_registry(self, tag: str) -> None:
        """Tag and push an image to the configured registry.

        Args:
            tag: Local image tag to push
        """
        if not self._registry:
            return

        registry_tag = f"{self._registry}/{tag}"
        if self._registry_check_disabled and not self._rebuild_existing:
            return

        if not self._rebuild_existing:
            try:
                await self._docker.pull_image(registry_tag)
                logger.info(f"Registry image already exists, skipping push: {registry_tag}")
                return
            except DockerClientError as e:
                error_text = str(e).lower()
                if e.status == 404:
                    logger.info(f"Registry image not found, pushing: {registry_tag}")
                elif e.status in (401, 403) or "unauthorized" in error_text:
                    if not self._registry_check_disabled:
                        logger.warning(
                            "Registry auth failed for %s; skipping registry pushes. "
                            'Use --rebuild-existing to force pushes or --registry "" to disable.',
                            self._registry,
                        )
                    self._registry_check_disabled = True
                    return
                else:
                    logger.info(
                        f"Could not verify registry image {registry_tag} ({e}); proceeding to push"
                    )
        logger.info(f"Tagging and pushing image {registry_tag}")
        await self._docker.tag_image(tag, registry_tag)
        await self._docker.push_image(registry_tag)
        logger.info(f"Successfully pushed image {registry_tag}")

    async def build_from_context(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        build_args: dict[str, str] | None = None,
    ) -> None:
        """Build a Docker image from a build context.

        Args:
            context_path: Path to the build context directory
            tag: Tag for the built image
            dockerfile_path: Path to Dockerfile relative to context
            build_args: Build arguments

        Raises:
            RuntimeError: If build fails
        """
        if not await self._prepare_for_build(tag):
            return

        await self._do_build(
            context_path=context_path,
            tag=tag,
            dockerfile_path=dockerfile_path,
            build_args=build_args,
        )

    async def build_modified_image(
        self,
        base_tag: str,
        dockerfile_extra: str,
        output_tag: str,
    ) -> None:
        """Build a modified image with additional Dockerfile instructions.

        Args:
            base_tag: Base image tag to build from
            dockerfile_extra: Additional Dockerfile instructions
            output_tag: Tag for the output image
        """
        if not await self._prepare_for_build(output_tag):
            return

        logger.info(f"Building modified image {output_tag} from {base_tag}")

        # Create temporary build context with Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dockerfile_path = temp_path / "Dockerfile"

            # Write Dockerfile with base image + extra instructions
            dockerfile_content = f"FROM {base_tag}\n{dockerfile_extra}"
            dockerfile_path.write_text(dockerfile_content)

            await self._do_build(
                context_path=str(temp_path),
                tag=output_tag,
            )

    async def tag_image(self, source_tag: str, target_tag: str) -> None:
        """Tag an existing image with a new tag.

        Args:
            source_tag: Source image tag
            target_tag: Target image tag
        """
        if not await self._prepare_for_build(target_tag):
            return
        await self._docker.tag_image(source_tag, target_tag)
        self._built_images.add(target_tag)

    async def _build_single_spec(self, spec: BuildImageSpec | MultiStageBuildImageSpec) -> None:
        """Build a single non-derived image spec.

        Args:
            spec: Image specification to build
        """
        if isinstance(spec, MultiStageBuildImageSpec):
            for stage in spec.stages:
                build_args = dict(stage.build_args) if stage.build_args else {}
                if stage.parent_tag:
                    build_args["BASE_IMAGE"] = stage.parent_tag

                await self.build_from_context(
                    context_path=stage.context_path,
                    tag=stage.tag,
                    dockerfile_path=stage.dockerfile_path,
                    build_args=build_args or None,
                )
            return

        if not await self._prepare_for_build(spec.tag):
            return

        # Run seeder before building if defined
        if spec.seeder is not None:
            logger.info(f"Running pre-build seeder for {spec.tag}")
            await spec.seeder()
            logger.info(f"Pre-build seeder completed for {spec.tag}")

        await self._do_build(
            context_path=spec.context_path,
            tag=spec.tag,
            dockerfile_path=spec.dockerfile_path,
            build_args=spec.build_args,
        )

    async def _do_build(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        build_args: dict[str, str] | None = None,
    ) -> None:
        """Execute the actual Docker build (assumes _prepare_for_build check already done).

        Args:
            context_path: Path to the build context directory
            tag: Tag for the built image
            dockerfile_path: Path to Dockerfile relative to context
            build_args: Build arguments

        Raises:
            RuntimeError: If build fails
        """
        logger.info(f"Building image {tag} from {context_path}")
        errors = []

        async for log_line in self._docker.build_image(
            context_path=context_path,
            tag=tag,
            dockerfile_path=dockerfile_path,
            buildargs=build_args,
        ):
            if "stream" in log_line:
                stream = log_line["stream"].strip()
                if stream:
                    logger.debug(stream)
            if "error" in log_line:
                error = log_line["error"]
                logger.error(f"Build error: {error}")
                errors.append(error)

        if errors:
            raise RuntimeError(f"Failed to build image {tag}: {errors}")

        self._built_images.add(tag)
        logger.info(f"Successfully built image {tag}")

    async def build_all_specs(self, specs: list[ImageBuildSpec]) -> list[BuildError]:
        """Build all image specs with proper dependency ordering.

        First builds all non-derived specs (BuildImageSpec, MultiStageBuildImageSpec),
        then builds derived specs (DerivedImageSpec) which depend on the base images.
        Derived specs are skipped if their base image failed to build.

        Args:
            specs: List of image specifications to build

        Returns:
            List of build errors (empty if all succeeded)
        """
        errors: list[BuildError] = []
        failed_base_tags: set[str] = set()

        _recoverable = (RuntimeError, DockerClientError, OSError)

        # Phase 1: Build non-derived specs (base images)
        base_specs = [s for s in specs if not isinstance(s, DerivedImageSpec)]
        for spec in base_specs:
            try:
                await self._build_single_spec(spec)
            except _recoverable as e:
                logger.error(
                    f"Failed to build image {spec.tag}: {type(e).__name__}: {e}",
                    exc_info=e,
                )
                errors.append(BuildError(image_tag=spec.tag, error=e))
                failed_base_tags.add(spec.tag)
                continue

            try:
                await self.push_to_registry(spec.tag)
            except _recoverable as e:
                logger.error(
                    f"Failed to push image {spec.tag}: {type(e).__name__}: {e}",
                    exc_info=e,
                )
                errors.append(BuildError(image_tag=spec.tag, error=e, phase="push"))

        # Phase 2: Build derived specs (depend on base images)
        derived_specs = [s for s in specs if isinstance(s, DerivedImageSpec)]

        for spec in derived_specs:
            # Skip derived images whose base image failed to build.
            # Note: failed_base_tags tracks the final tag of MultiStageBuildImageSpec
            # (not intermediate stage tags). This works because current consumers only
            # reference final tags in DerivedImageSpec.base_image_tag.
            if spec.base_image_tag in failed_base_tags:
                errors.append(
                    BuildError(
                        image_tag=spec.tag,
                        error=RuntimeError(f"Base image {spec.base_image_tag} failed to build"),
                    )
                )
                continue
            try:
                await self.build_modified_image(
                    base_tag=spec.base_image_tag,
                    dockerfile_extra=spec.dockerfile_extra,
                    output_tag=spec.tag,
                )
            except _recoverable as e:
                logger.error(
                    f"Failed to build derived image {spec.tag}: {type(e).__name__}: {e}",
                    exc_info=e,
                )
                errors.append(BuildError(image_tag=spec.tag, error=e))
                continue

            try:
                await self.push_to_registry(spec.tag)
            except _recoverable as e:
                logger.error(
                    f"Failed to push derived image {spec.tag}: {type(e).__name__}: {e}",
                    exc_info=e,
                )
                errors.append(BuildError(image_tag=spec.tag, error=e, phase="push"))

        return errors


T = TypeVar("T", bound=BaseModel)


def _maybe_override_cache_dir(config: T, cache_dir: str | None) -> T:
    """Override cache_dir on configs that define it."""
    if cache_dir is None:
        return config
    if not hasattr(config, "cache_dir"):
        raise ValueError(
            f"Dataset config {type(config).__name__} does not define cache_dir, "
            "but build_images was called with a cache_dir override."
        )
    return config.model_copy(update={"cache_dir": str(cache_dir)})


async def build_dataset_images(
    dataset_name: str,
    builder: ImageBuilder,
    cache_dir: str | None = None,
) -> list[BuildError]:
    """Build all images for a specific dataset using default configuration.

    Args:
        dataset_name: Name of the dataset to build images for
        builder: Image builder instance
        cache_dir: Optional cache directory override for datasets that support it

    Returns:
        List of build errors (empty if all succeeded)

    Raises:
        ValueError: If dataset doesn't support image building
    """
    # Get the dataset's config class and create default config
    config_class = get_dataset_config_class(dataset_name)
    config = config_class()
    config = _maybe_override_cache_dir(config, cache_dir)

    # Get image build specs directly from the dataset class (no instance needed)
    specs = get_image_build_specs(dataset_name, config)
    logger.info(f"Found {len(specs)} image spec(s) for dataset {dataset_name}")

    # Build all specs
    return await builder.build_all_specs(specs)


def _validate_datasets(datasets: list[str]) -> None:
    """Validate that all dataset names exist and support image building.

    Performs upfront validation so input errors fail fast before any
    Docker operations begin.

    Args:
        datasets: List of dataset names to validate

    Raises:
        UnknownComponentError: If a dataset name is not registered
        ImportError: If a dataset's dependencies are not installed
        ValueError: If a dataset doesn't support image building
    """
    for name in datasets:
        # Triggers entry point loading + checks for failed entry points
        get_dataset_config_class(name)
        classes = dataset_registry.get_component_classes()
        dataset_class = classes.get(name)
        if dataset_class is None or not issubclass(dataset_class, ImageBuildableDataset):
            raise ValueError(
                f"Dataset '{name}' does not support image building. "
                "Only datasets implementing ImageBuildableDataset can be used with this command."
            )


async def run_build(
    datasets: list[str],
    cache_dir: str = ".build_cache",
    rebuild_existing: bool = False,
    registry: str | None = None,
) -> None:
    """Run the image building process for specified datasets.

    Args:
        datasets: List of dataset names to build images for
        cache_dir: Directory for caching build contexts
        rebuild_existing: If True, delete and rebuild existing images.
        registry: Optional registry to tag and push images to

    Raises:
        ExceptionGroup: If any image builds failed
        ValueError: If a dataset name is invalid or doesn't support image building
    """
    _validate_datasets(datasets)

    # Create Docker client - always use local client for image building
    logger.warning(
        "Image building always uses the local Docker client. Ensure Docker is running locally."
    )
    docker_client = create_docker_client_from_config("local", {})

    # Collect all build errors
    all_build_errors: list[BuildError] = []

    try:
        # Create image builder
        builder = ImageBuilder(
            docker_client=docker_client,
            cache_dir=Path(cache_dir),
            rebuild_existing=rebuild_existing,
            registry=registry,
        )

        # Build images for each dataset
        for dataset_name in datasets:
            errors = await build_dataset_images(
                dataset_name,
                builder,
                cache_dir=cache_dir,
            )
            all_build_errors.extend(errors)

        if all_build_errors:
            logger.warning(f"Image building completed with {len(all_build_errors)} error(s)")
        else:
            logger.info("Image building complete!")

    finally:
        await docker_client.close()

    # Raise ExceptionGroup if there were any errors
    _handle_build_failures(all_build_errors)


@click.command()
@click.option(
    "--dataset",
    multiple=True,
    help="Dataset(s) to build images for (can be specified multiple times)",
)
@click.option(
    "--all-datasets",
    is_flag=True,
    default=False,
    help="Build images for all registered datasets that support Docker",
)
@click.option(
    "--cache-dir",
    default=".build_cache",
    help="Directory for caching build contexts",
)
@click.option(
    "--rebuild-existing",
    is_flag=True,
    default=False,
    help="Delete and rebuild existing images instead of skipping them",
)
@click.option(
    "--registry",
    type=str,
    default="ghcr.io/ethz-spylab/prompt-siren-images",
    help=(
        "Registry to tag and push images to "
        "(default: 'ghcr.io/ethz-spylab/prompt-siren-images'). "
        "Use an empty string to disable."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def main(
    dataset: tuple[str, ...],
    all_datasets: bool,
    cache_dir: str,
    rebuild_existing: bool,
    registry: str | None,
    verbose: bool,
) -> None:
    """Build Docker images for datasets.

    This command pre-builds all Docker images needed for running dataset
    evaluations. Images are tagged with a consistent scheme so they can be
    pulled during experiment runs.

    Examples:
        # Build images for SWE-bench dataset
        prompt-siren-build-images --dataset swebench

        # Build images for all supported datasets
        prompt-siren-build-images --all-datasets
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if registry is not None and registry.strip() == "":
        registry = None

    if all_datasets:
        datasets_to_build = get_datasets_with_image_specs()
        if not datasets_to_build:
            logger.error("No datasets with Docker support found")
            sys.exit(1)
        logger.info(f"Building images for all Docker datasets: {datasets_to_build}")
    elif dataset:
        datasets_to_build = list(dataset)
    else:
        logger.error("Must specify --dataset or --all-datasets")
        sys.exit(1)

    try:
        asyncio.run(
            run_build(
                datasets=datasets_to_build,
                cache_dir=cache_dir,
                rebuild_existing=rebuild_existing,
                registry=registry,
            )
        )
    except ExceptionGroup as eg:
        # ExceptionGroup from _handle_build_failures - details already logged
        logger.error(f"Build completed with failures: {eg}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build failed unexpectedly: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

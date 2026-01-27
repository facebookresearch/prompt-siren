# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Script for building Docker images.

This script pre-builds all Docker images needed for running datasets that
use Docker containers, including:
- SWE-bench: benign task images, malicious service containers, pair images

Usage:
    # Build images for a specific dataset
    prompt-siren-build-images --dataset swebench [OPTIONS]

    # Build images for all datasets
    prompt-siren-build-images --all-datasets [OPTIONS]

The script uses the ImageBuildableDataset protocol to discover which images each
dataset needs, then builds them with proper dependency ordering.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from pydantic import ValidationError

from .datasets import (
    get_dataset_config_class,
    get_datasets_with_image_specs,
    get_image_build_specs,
)
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
                logger.debug(f"Image '{tag}' does not exist (404)")
                return False
            logger.error(f"Docker error while checking if image '{tag}' exists: {e}")
            raise

    async def delete_image(self, tag: str) -> None:
        """Delete an image by tag.

        Args:
            tag: Image tag to delete
        """
        logger.info(f"Deleting existing image {tag}")
        await self._docker.delete_image(tag, force=True)

    async def _should_build(self, tag: str) -> bool:
        """Check if an image should be built, handling already-built and existing images.

        Returns True if the image should be built, False if it should be skipped.
        If rebuild_existing is True and image exists, deletes it first.

        Args:
            tag: Image tag to check

        Returns:
            True if the image should be built, False to skip
        """
        if tag in self._built_images:
            logger.debug(f"Image {tag} already built in this session, skipping")
            return False

        if await self.image_exists(tag):
            if self._rebuild_existing:
                await self.delete_image(tag)
                return True
            logger.info(f"Image {tag} already exists, skipping")
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
        logger.info(f"Tagging image {tag} as {registry_tag}")
        await self._docker.tag_image(tag, registry_tag)

        logger.info(f"Pushing image {registry_tag}")
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
        if not await self._should_build(tag):
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
        if not await self._should_build(output_tag):
            return

        logger.info(f"Building modified image {output_tag} from {base_tag}")

        # Create temporary build context with Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dockerfile_path = temp_path / "Dockerfile"

            # Write Dockerfile with base image + extra instructions
            dockerfile_content = f"FROM {base_tag}\n{dockerfile_extra}"
            dockerfile_path.write_text(dockerfile_content)

            await self.build_from_context(
                context_path=str(temp_path),
                tag=output_tag,
            )

    async def tag_image(self, source_tag: str, target_tag: str) -> None:
        """Tag an existing image with a new tag.

        Args:
            source_tag: Source image tag
            target_tag: Target image tag
        """
        if not await self._should_build(target_tag):
            return

        logger.info(f"Tagging image {source_tag} as {target_tag}")
        await self._docker.tag_image(source_tag, target_tag)
        self._built_images.add(target_tag)
        logger.info(f"Successfully tagged image {target_tag}")

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

        # BuildImageSpec handling
        if not await self._should_build(spec.tag):
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
        """Execute the actual Docker build (assumes _should_build check already done).

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

        # Phase 1: Build non-derived specs (base images)
        base_specs = [s for s in specs if not isinstance(s, DerivedImageSpec)]
        logger.info(f"Phase 1: Building {len(base_specs)} base image(s)")

        for spec in base_specs:
            try:
                await self._build_single_spec(spec)
                await self.push_to_registry(spec.tag)
            except (  # noqa: PERF203
                KeyboardInterrupt,
                SystemExit,
                asyncio.CancelledError,
            ):
                # Don't swallow cancellation - let it propagate
                raise
            except Exception as e:
                logger.error(f"Failed to build image {spec.tag}: {type(e).__name__}: {e}")
                errors.append(BuildError(image_tag=spec.tag, error=e))
                failed_base_tags.add(spec.tag)

        # Phase 2: Build derived specs (depend on base images)
        derived_specs = [s for s in specs if isinstance(s, DerivedImageSpec)]
        logger.info(f"Phase 2: Building {len(derived_specs)} derived image(s)")

        for spec in derived_specs:
            # Skip derived images whose base image failed to build
            if spec.base_image_tag in failed_base_tags:
                logger.warning(
                    f"Skipping derived image {spec.tag}: "
                    f"base image {spec.base_image_tag} failed to build"
                )
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
                await self.push_to_registry(spec.tag)
            except (
                KeyboardInterrupt,
                SystemExit,
                asyncio.CancelledError,
            ):
                # Don't swallow cancellation - let it propagate
                raise
            except Exception as e:
                logger.error(f"Failed to build derived image {spec.tag}: {type(e).__name__}: {e}")
                errors.append(BuildError(image_tag=spec.tag, error=e))

        return errors


async def build_dataset_images(
    dataset_name: str,
    builder: ImageBuilder,
    **config_overrides: Any,
) -> list[BuildError]:
    """Build all images for a specific dataset.

    Args:
        dataset_name: Name of the dataset to build images for
        builder: Image builder instance
        **config_overrides: Optional configuration overrides for the dataset

    Returns:
        List of build errors (empty if all succeeded)

    Raises:
        ValueError: If dataset doesn't support image building or config is invalid
    """
    logger.info(f"Building images for dataset: {dataset_name}")

    # Get the dataset's config class and create config
    config_class = get_dataset_config_class(dataset_name)
    try:
        config = config_class(**config_overrides)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for dataset '{dataset_name}': {e}") from e

    # Get image build specs directly from the dataset class (no instance needed)
    specs = get_image_build_specs(dataset_name, config)
    logger.info(f"Found {len(specs)} image spec(s) for dataset {dataset_name}")

    # Build all specs
    return await builder.build_all_specs(specs)


async def run_build(
    datasets: list[str],
    cache_dir: str = ".build_cache",
    rebuild_existing: bool = False,
    registry: str | None = None,
    config_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Run the image building process for specified datasets.

    Args:
        datasets: List of dataset names to build images for
        cache_dir: Directory for caching build contexts
        rebuild_existing: If True, delete and rebuild existing images.
        registry: Optional registry to tag and push images to
        config_overrides: Optional configuration overrides per dataset

    Raises:
        ExceptionGroup: If any image builds failed
    """
    config_overrides = config_overrides or {}

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
            try:
                overrides: dict[str, Any] = config_overrides.get(dataset_name, {})
                errors = await build_dataset_images(
                    dataset_name,
                    builder,
                    **overrides,
                )
                all_build_errors.extend(errors)
            except (  # noqa: PERF203
                KeyboardInterrupt,
                SystemExit,
                asyncio.CancelledError,
            ):
                # Don't swallow cancellation - let it propagate
                raise
            except Exception as e:
                logger.error(
                    f"Failed to build images for dataset {dataset_name}: {type(e).__name__}: {e}"
                )
                all_build_errors.append(BuildError(image_tag=f"dataset:{dataset_name}", error=e))

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
    default=None,
    help="Registry to tag and push images to (e.g., 'my-registry.com/myrepo')",
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
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine which datasets to build
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
    except (KeyboardInterrupt, SystemExit):
        # User cancellation - exit cleanly
        raise
    except Exception as e:
        logger.error(f"Build failed unexpectedly: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

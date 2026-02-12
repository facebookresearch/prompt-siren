# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Standalone script for building Docker images for SWEBench tasks.

This script pre-builds all Docker images needed for running the SWEBench dataset,
including:
- Benign task images (multi-stage builds with base/env/instance layers)
- Malicious task service container images
- Combined images for benign x malicious pairs (with dockerfile_extra applied)

Usage:
    prompt-siren-build-images [OPTIONS]

The script reads the SWEBench dataset configuration and builds Docker images
with a consistent tagging scheme that can then be referenced via PullImageSpec
in the core CLI.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click

# ExceptionGroup is built-in in Python 3.11+, needs backport for 3.10
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

try:
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.utils import load_swebench_dataset
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

from .datasets.swebench_dataset.config import SwebenchDatasetConfig
from .datasets.swebench_dataset.constants import INSTANCE_INJECTION_MAPPING
from .datasets.swebench_dataset.docker_builder import prepare_build_context
from .datasets.swebench_dataset.image_tags import (
    get_benign_image_tag,
    get_pair_image_tag,
)
from .datasets.swebench_dataset.malicious_tasks import MALICIOUS_TASKS
from .datasets.swebench_dataset.malicious_tasks.build_registry import (
    get_all_service_container_build_specs,
)
from .datasets.swebench_dataset.task_metadata import SWEBenchMaliciousTaskMetadata
from .sandbox_managers.docker.manager import create_docker_client_from_config
from .sandbox_managers.docker.plugins import AbstractDockerClient
from .sandbox_managers.docker.plugins.errors import DockerClientError
from .sandbox_managers.image_spec import BuildImageSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildSuccess:
    """Successful build result."""

    image_tag: str


@dataclass(frozen=True)
class BuildError:
    """Failed build result."""

    image_tag: str
    error: Exception


BuildResult = BuildSuccess | BuildError


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
    """Handles building Docker images for SWEBench tasks."""

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
            # Check if this is a 404 "image not found" error
            if e.status == 404:
                logger.debug(f"Image '{tag}' does not exist (404)")
                return False
            # For other errors (daemon issues, permission errors, etc.), re-raise
            # so the caller can handle them appropriately
            logger.error(f"Docker error while checking if image '{tag}' exists: {e}")
            raise

    async def delete_image(self, tag: str) -> None:
        """Delete an image by tag.

        Args:
            tag: Image tag to delete
        """
        logger.info(f"Deleting existing image {tag}")
        await self._docker.delete_image(tag, force=True)

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
        if tag in self._built_images:
            logger.debug(f"Image {tag} already built in this session, skipping")
            return

        image_exists = await self.image_exists(tag)

        if image_exists:
            if self._rebuild_existing:
                # Delete the existing image before rebuilding
                await self.delete_image(tag)
            else:
                # Default: skip existing images
                logger.info(f"Image {tag} already exists, skipping")
                self._built_images.add(tag)
                return

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
        if output_tag in self._built_images:
            logger.debug(f"Image {output_tag} already built in this session, skipping")
            return

        image_exists = await self.image_exists(output_tag)

        if image_exists:
            if self._rebuild_existing:
                # Delete the existing image before rebuilding
                await self.delete_image(output_tag)
            else:
                # Default: skip existing images
                logger.info(f"Image {output_tag} already exists, skipping")
                self._built_images.add(output_tag)
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
        if target_tag in self._built_images:
            logger.debug(f"Image {target_tag} already exists in this session, skipping")
            return

        image_exists = await self.image_exists(target_tag)

        if image_exists:
            if self._rebuild_existing:
                # Delete the existing image before re-tagging
                await self.delete_image(target_tag)
            else:
                # Default: skip existing images
                logger.info(f"Image {target_tag} already exists, skipping tagging")
                self._built_images.add(target_tag)
                return

        logger.info(f"Tagging image {source_tag} as {target_tag}")
        await self._docker.tag_image(source_tag, target_tag)
        self._built_images.add(target_tag)
        logger.info(f"Successfully tagged image {target_tag}")


async def build_benign_task_images(
    builder: ImageBuilder,
    instances: list[SWEbenchInstance],
    config: SwebenchDatasetConfig,
) -> tuple[dict[str, str], list[BuildError]]:
    """Build images for all benign tasks.

    Args:
        builder: Image builder instance
        instances: List of SWEBench instances
        config: SWEBench dataset configuration

    Returns:
        Tuple of (mapping from instance_id to built image tag, list of build errors)
    """

    benign_images: dict[str, str] = {}
    build_errors: list[BuildError] = []

    async def build_single_benign(instance: SWEbenchInstance) -> tuple[str, BuildResult]:
        """Build a single benign image and return result."""
        instance_id = instance["instance_id"]
        output_tag = get_benign_image_tag(instance_id)

        try:
            # Generate the multi-stage build spec
            spec, _test_spec = prepare_build_context(instance, config)

            # Build each stage
            for stage in spec.stages:
                build_args = dict(stage.build_args) if stage.build_args else {}
                if stage.parent_tag:
                    build_args["BASE_IMAGE"] = stage.parent_tag

                await builder.build_from_context(
                    context_path=stage.context_path,
                    tag=stage.tag,
                    dockerfile_path=stage.dockerfile_path,
                    build_args=build_args if build_args else None,
                )

            # Tag the final image with our standardized tag
            # The final_tag from the spec is the instance_image_key (SWEBench's native tag)
            # We re-tag it to our standardized naming convention for consistency
            if spec.final_tag != output_tag:
                await builder.tag_image(spec.final_tag, output_tag)

            # Push to registry if configured
            await builder.push_to_registry(output_tag)

            return instance_id, BuildSuccess(image_tag=output_tag)
        except Exception as e:
            logger.warning(f"Failed to build benign image for {instance_id}: {e}")
            return instance_id, BuildError(image_tag=output_tag, error=e)

    for instance in instances:
        instance_id, result = await build_single_benign(instance)
        match result:
            case BuildSuccess(image_tag=tag):
                benign_images[instance_id] = tag
            case BuildError():
                build_errors.append(result)

    return benign_images, build_errors


async def build_malicious_task_images(
    builder: ImageBuilder,
) -> tuple[dict[str, str], list[BuildError]]:
    """Build images for all malicious tasks' service containers.

    Uses the build registry to get BuildImageSpecs for all service containers.

    Args:
        builder: Image builder instance

    Returns:
        Tuple of (mapping from image_tag -> image_tag, list of build errors)
    """
    built_images: dict[str, str] = {}
    build_errors: list[BuildError] = []

    async def build_single_malicious(build_spec: BuildImageSpec) -> BuildResult:
        """Build a single malicious image and return result."""
        try:
            await builder.build_from_context(
                context_path=build_spec.context_path,
                tag=build_spec.tag,
                dockerfile_path=build_spec.dockerfile_path,
                build_args=build_spec.build_args,
            )
            # Push to registry if configured
            await builder.push_to_registry(build_spec.tag)
            return BuildSuccess(image_tag=build_spec.tag)
        except Exception as e:
            logger.warning(f"Failed to build malicious image {build_spec.tag}: {e}")
            return BuildError(image_tag=build_spec.tag, error=e)

    for build_spec in get_all_service_container_build_specs():
        result = await build_single_malicious(build_spec)
        match result:
            case BuildSuccess(image_tag=tag):
                built_images[tag] = tag
            case BuildError():
                build_errors.append(result)

    return built_images, build_errors


async def build_pair_images(
    builder: ImageBuilder,
    benign_images: dict[str, str],
) -> tuple[dict[tuple[str, str], str], list[BuildError]]:
    """Build images for all benign x malicious pairs.

    For each pair, this applies the malicious task's benign_dockerfile_extra
    to the benign task's image.

    Args:
        builder: Image builder instance
        benign_images: Mapping from benign instance_id to image tag

    Returns:
        Tuple of (mapping from (benign_id, malicious_id) to built image tag, list of build errors)
    """
    pair_images: dict[tuple[str, str], str] = {}
    build_errors: list[BuildError] = []

    async def build_single_pair(
        benign_id: str,
        benign_tag: str,
        malicious_id: str,
        dockerfile_extra: str,
        pair_tag: str,
    ) -> BuildResult:
        """Build a single pair image and return result."""
        try:
            await builder.build_modified_image(
                base_tag=benign_tag,
                dockerfile_extra=dockerfile_extra,
                output_tag=pair_tag,
            )
            # Push to registry if configured
            await builder.push_to_registry(pair_tag)
            return BuildSuccess(image_tag=pair_tag)
        except Exception as e:
            logger.warning(f"Failed to build pair image {pair_tag}: {e}")
            return BuildError(image_tag=pair_tag, error=e)

    for benign_id, benign_tag in benign_images.items():
        for task in MALICIOUS_TASKS:
            malicious_id = task.id

            # Check if this malicious task has dockerfile_extra to apply
            if (
                isinstance(task.metadata, SWEBenchMaliciousTaskMetadata)
                and task.metadata.benign_dockerfile_extra
            ):
                dockerfile_extra = task.metadata.benign_dockerfile_extra
                pair_tag = get_pair_image_tag(benign_id, malicious_id)

                result = await build_single_pair(
                    benign_id, benign_tag, malicious_id, dockerfile_extra, pair_tag
                )
                match result:
                    case BuildSuccess(image_tag=tag):
                        pair_images[(benign_id, malicious_id)] = tag
                    case BuildError():
                        build_errors.append(result)
            else:
                # No modification needed, use the benign image directly
                pair_images[(benign_id, malicious_id)] = benign_tag

    return pair_images, build_errors


async def run_build(
    cache_dir: str = ".swebench_cache",
    rebuild_existing: bool = False,
    registry: str | None = None,
    max_instances: int | None = None,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    skip_benign: bool = False,
    skip_malicious: bool = False,
    skip_pairs: bool = False,
) -> None:
    """Run the image building process.

    Args:
        cache_dir: Directory for caching build contexts
        rebuild_existing: If True, delete and rebuild existing images.
            If False (default), skip building images that already exist.
        registry: Optional registry to tag and push images to
        max_instances: Maximum number of instances to build
        instance_ids: Specific instance IDs to build
        dataset_name: SWEBench dataset name
        skip_benign: Skip building benign task images
        skip_malicious: Skip building malicious task service images
        skip_pairs: Skip building pair images

    Raises:
        ExceptionGroup: If any image builds failed
    """

    # Create dataset config
    config = SwebenchDatasetConfig(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        max_instances=max_instances,
    )

    # Load and filter instances
    logger.info(f"Loading SWEBench dataset: {dataset_name}")
    all_instances = load_swebench_dataset(dataset_name)

    # Filter to supported instances
    supported_instances = [
        i for i in all_instances if i["instance_id"] in INSTANCE_INJECTION_MAPPING
    ]
    logger.info(f"Found {len(supported_instances)} supported instances")

    if max_instances:
        logger.warning(
            f"The parameter max_instances was specified, ignoring some instance_ids, and building up to {max_instances}. This is probably only desired for testing!"
        )
        instances = supported_instances[:max_instances]
    else:
        instances = supported_instances

    logger.info(f"Building images for {len(instances)} instances")

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

        # Build benign task images
        benign_images: dict[str, str] = {}
        if not skip_benign:
            logger.info("Building benign task images...")
            benign_images, benign_errors = await build_benign_task_images(
                builder, instances, config
            )
            all_build_errors.extend(benign_errors)
            logger.info(f"Built {len(benign_images)} benign images")

        # Build malicious task service images
        if not skip_malicious:
            logger.info("Building malicious task service images...")
            malicious_images, malicious_errors = await build_malicious_task_images(builder)
            all_build_errors.extend(malicious_errors)
            logger.info(f"Built {len(malicious_images)} malicious service images")

        # Build pair images
        if skip_pairs:
            logger.info("Skipping pair image builds (--skip-pairs flag set)")
        elif not benign_images:
            logger.warning(
                "Skipping pair image builds: no benign images available. "
                "Either --skip-benign was set or all benign builds failed."
            )
        else:
            logger.info("Building pair images...")
            pair_images, pair_errors = await build_pair_images(builder, benign_images)
            all_build_errors.extend(pair_errors)
            logger.info(f"Built {len(pair_images)} pair images")

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
    "--cache-dir",
    default=".swebench_cache",
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
    "--max-instances",
    type=int,
    default=None,
    help="Maximum number of instances to build",
)
@click.option(
    "--instance-ids",
    multiple=True,
    default=None,
    help="Specific instance IDs to build (can be specified multiple times)",
)
@click.option(
    "--dataset",
    default="SWE-bench/SWE-bench_Lite",
    help="SWEBench dataset name",
)
@click.option(
    "--skip-benign",
    is_flag=True,
    default=False,
    help="Skip building benign task images",
)
@click.option(
    "--skip-malicious",
    is_flag=True,
    default=False,
    help="Skip building malicious task service images",
)
@click.option(
    "--skip-pairs",
    is_flag=True,
    default=False,
    help="Skip building pair images",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def main(
    cache_dir: str,
    rebuild_existing: bool,
    registry: str | None,
    max_instances: int | None,
    dataset: str,
    skip_benign: bool,
    skip_malicious: bool,
    skip_pairs: bool,
    verbose: bool,
) -> None:
    """Build Docker images for SWEBench tasks.

    This command pre-builds all Docker images needed for running SWEBench
    evaluations. Images are tagged with a consistent scheme so they can be
    pulled during experiment runs.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(
            run_build(
                cache_dir=cache_dir,
                rebuild_existing=rebuild_existing,
                registry=registry,
                max_instances=max_instances,
                dataset_name=dataset,
                skip_benign=skip_benign,
                skip_malicious=skip_malicious,
                skip_pairs=skip_pairs,
            )
        )
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

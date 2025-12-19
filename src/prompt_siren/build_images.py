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
from pathlib import Path

import click

try:
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
from .datasets.swebench_dataset.malicious_tasks.constants import (
    _BASIC_AGENT_BUILD_SPEC,
)
from .datasets.swebench_dataset.task_metadata import SWEBenchMaliciousTaskMetadata
from .sandbox_managers.docker.manager import create_docker_client_from_config
from .sandbox_managers.docker.plugins import AbstractDockerClient
from .sandbox_managers.docker.plugins.errors import DockerClientError

logger = logging.getLogger(__name__)


class ImageBuilder:
    """Handles building Docker images for SWEBench tasks."""

    def __init__(
        self,
        docker_client: AbstractDockerClient,
        cache_dir: Path,
        use_cache: bool = True,
    ):
        """Initialize the image builder.

        Args:
            docker_client: Docker client for building images
            cache_dir: Directory for caching build contexts
            use_cache: Whether to skip rebuilding existing images
        """
        self._docker = docker_client
        self._cache_dir = cache_dir
        self._use_cache = use_cache
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
        except DockerClientError:
            return False

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

        if self._use_cache and await self.image_exists(tag):
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

        if self._use_cache and await self.image_exists(output_tag):
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
                dockerfile_path="Dockerfile",
            )


async def build_benign_task_images(
    builder: ImageBuilder,
    instances: list,
    config,
) -> dict[str, str]:
    """Build images for all benign tasks.

    Args:
        builder: Image builder instance
        instances: List of SWEBench instances
        config: SWEBench dataset configuration

    Returns:
        Mapping from instance_id to built image tag
    """

    benign_images: dict[str, str] = {}

    for instance in instances:
        instance_id = instance["instance_id"]
        output_tag = get_benign_image_tag(instance_id)

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
        # The final_tag from the spec is the instance_image_key
        if spec.final_tag != output_tag:
            # We need to re-tag or just use the original tag
            # For simplicity, we'll use the original final_tag as the benign tag
            benign_images[instance_id] = spec.final_tag
        else:
            benign_images[instance_id] = output_tag

    return benign_images


async def build_malicious_task_images(
    builder: ImageBuilder,
) -> dict[str, str]:
    """Build images for all malicious tasks' service containers.

    Uses the build registry to get BuildImageSpecs for all service containers.

    Args:
        builder: Image builder instance

    Returns:
        Mapping from image_tag -> image_tag (for tracking what was built)
    """
    built_images: dict[str, str] = {}

    for build_spec in get_all_service_container_build_specs():
        await builder.build_from_context(
            context_path=build_spec.context_path,
            tag=build_spec.tag,
            dockerfile_path=build_spec.dockerfile_path,
            build_args=build_spec.build_args,
        )
        built_images[build_spec.tag] = build_spec.tag

    return built_images


async def build_pair_images(
    builder: ImageBuilder,
    benign_images: dict[str, str],
) -> dict[tuple[str, str], str]:
    """Build images for all benign x malicious pairs.

    For each pair, this applies the malicious task's benign_dockerfile_extra
    to the benign task's image.

    Args:
        builder: Image builder instance
        benign_images: Mapping from benign instance_id to image tag

    Returns:
        Mapping from (benign_id, malicious_id) to built image tag
    """
    pair_images: dict[tuple[str, str], str] = {}

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

                await builder.build_modified_image(
                    base_tag=benign_tag,
                    dockerfile_extra=dockerfile_extra,
                    output_tag=pair_tag,
                )
                pair_images[(benign_id, malicious_id)] = pair_tag
            else:
                # No modification needed, use the benign image directly
                pair_images[(benign_id, malicious_id)] = benign_tag

    return pair_images


async def build_basic_agent_image(builder: ImageBuilder) -> str:
    """Build the basic agent container image used by malicious tasks.

    Uses the build registry to get the BuildImageSpec for the basic agent.

    Args:
        builder: Image builder instance

    Returns:
        The image tag for the basic agent
    """
    await builder.build_from_context(
        context_path=_BASIC_AGENT_BUILD_SPEC.context_path,
        tag=_BASIC_AGENT_BUILD_SPEC.tag,
        dockerfile_path=_BASIC_AGENT_BUILD_SPEC.dockerfile_path,
        build_args=_BASIC_AGENT_BUILD_SPEC.build_args,
    )
    return _BASIC_AGENT_BUILD_SPEC.tag


async def run_build(
    cache_dir: str = ".swebench_cache",
    use_cache: bool = True,
    max_instances: int | None = None,
    instance_ids: list[str] | None = None,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    skip_benign: bool = False,
    skip_malicious: bool = False,
    skip_pairs: bool = False,
) -> None:
    """Run the image building process.

    Args:
        cache_dir: Directory for caching build contexts
        use_cache: Whether to skip rebuilding existing images
        max_instances: Maximum number of instances to build
        instance_ids: Specific instance IDs to build
        dataset_name: SWEBench dataset name
        skip_benign: Skip building benign task images
        skip_malicious: Skip building malicious task service images
        skip_pairs: Skip building pair images
    """

    # Create dataset config
    config = SwebenchDatasetConfig(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        use_cache=use_cache,
        max_instances=max_instances,
        instance_ids=instance_ids,
    )

    # Load and filter instances
    logger.info(f"Loading SWEBench dataset: {dataset_name}")
    all_instances = load_swebench_dataset(dataset_name)

    # Filter to supported instances
    supported_instances = [
        i for i in all_instances if i["instance_id"] in INSTANCE_INJECTION_MAPPING
    ]
    logger.info(f"Found {len(supported_instances)} supported instances")

    # Apply filters
    if instance_ids:
        instances = [i for i in supported_instances if i["instance_id"] in instance_ids]
    elif max_instances:
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

    try:
        # Create image builder
        builder = ImageBuilder(
            docker_client=docker_client,
            cache_dir=Path(cache_dir),
            use_cache=use_cache,
        )

        # Build basic agent image (used by malicious tasks)
        if not skip_malicious:
            logger.info("Building basic agent image...")
            await build_basic_agent_image(builder)

        # Build benign task images
        benign_images: dict[str, str] = {}
        if not skip_benign:
            logger.info("Building benign task images...")
            benign_images = await build_benign_task_images(builder, instances, config)
            logger.info(f"Built {len(benign_images)} benign images")

        # Build malicious task service images
        if not skip_malicious:
            logger.info("Building malicious task service images...")
            malicious_images = await build_malicious_task_images(builder)
            logger.info(f"Built {len(malicious_images)} malicious service images")

        # Build pair images
        if not skip_pairs and benign_images:
            logger.info("Building pair images...")
            pair_images = await build_pair_images(builder, benign_images)
            logger.info(f"Built {len(pair_images)} pair images")

        logger.info("Image building complete!")

    finally:
        await docker_client.close()


@click.command()
@click.option(
    "--cache-dir",
    default=".swebench_cache",
    help="Directory for caching build contexts",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Force rebuild all images",
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
    no_cache: bool,
    max_instances: int | None,
    instance_ids: tuple[str, ...],
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

    # Convert tuple to list or None
    instance_id_list = list(instance_ids) if instance_ids else None

    try:
        asyncio.run(
            run_build(
                cache_dir=cache_dir,
                use_cache=not no_cache,
                max_instances=max_instances,
                instance_ids=instance_id_list,
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

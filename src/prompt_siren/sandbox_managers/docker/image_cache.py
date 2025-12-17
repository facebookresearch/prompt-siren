# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image caching and building for Docker sandbox management."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from typing_extensions import assert_never

from ..image_spec import (
    BuildImageSpec,
    ImageSpec,
    ImageTag,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from ..sandbox_task_setup import ContainerSetup, SandboxTaskSetup
from .plugins import AbstractDockerClient, DockerClientError

logger = logging.getLogger(__name__)


class ImageBuildError(Exception):
    """Raised when image building fails."""

    def __init__(self, tag: str, errors: list[str]):
        self.tag = tag
        self.errors = errors
        super().__init__(f"Failed to build image {tag}: {errors}")


class ImageCache:
    """Manages image building, pulling, and caching for a batch.

    Images are prepared sequentially during setup_batch to avoid
    concurrent Docker build race conditions.
    """

    def __init__(
        self,
        docker: AbstractDockerClient,
        batch_id: str,
        built_images_file: Path | None = None,
        registry_prefix: str | None = None,
        execution_mode: str = "build_and_run",
    ):
        """Initialize image cache.

        Args:
            docker: Abstract Docker client for all operations
            batch_id: Unique identifier for this batch
            built_images_file: Optional path to file where built image tags will be written
            registry_prefix: Optional registry prefix for modified images (e.g., 'ghcr.io/org/repo')
            execution_mode: Execution mode ('build_and_run', 'build_only', or 'run_from_prebuilt')
        """
        self._docker = docker
        self._batch_id = batch_id
        self._built_images_file = built_images_file
        self._registry_prefix = registry_prefix
        self._execution_mode = execution_mode
        self._built_images: set[ImageTag] = set()
        # Map from cache key to prepared image tag
        self._base_image_cache: dict[str, ImageTag] = {}
        # Map from (base_tag, dockerfile_extra_hash) to modified image tag
        self._modified_image_cache: dict[tuple[str, str], ImageTag] = {}

    async def ensure_all_base_images(
        self, task_setups: list[SandboxTaskSetup] | list[ContainerSetup]
    ) -> None:
        """Ensure all base images and modified images are available before task execution.

        Builds/pulls all unique base images sequentially to avoid race conditions,
        then builds any modified images for containers with dockerfile_extra.

        Args:
            task_setups: All task setups from the batch (preferred), or legacy container setups
        """
        # Handle both new task_setups format and legacy container_setups format
        if task_setups and isinstance(task_setups[0], ContainerSetup):
            # Legacy format: list of ContainerSetup (no task_id available for modified images)
            container_setups = task_setups
            task_setups_list: list[SandboxTaskSetup] = []
        else:
            # New format: list of SandboxTaskSetup
            task_setups_list = task_setups  # type: ignore[assignment]
            container_setups = []
            for task_setup in task_setups_list:
                container_setups.append(task_setup.agent_container)
                container_setups.extend(task_setup.service_containers.values())

        logger.debug(
            f"[ImageCache] ensure_all_base_images called with {len(container_setups)} container setups"
        )

        # Phase 1: Build/pull all unique base images
        seen_specs: set[str] = set()
        for setup in container_setups:
            cache_key = self._get_cache_key(setup.spec.image_spec)
            logger.debug(
                f"[ImageCache] Processing setup '{setup.name}' with cache_key: {cache_key}"
            )
            if cache_key not in seen_specs:
                seen_specs.add(cache_key)
                logger.debug(f"[ImageCache] Preparing base image for cache_key: {cache_key}")
                await self._prepare_base_image(setup.spec.image_spec)
            else:
                logger.debug(f"[ImageCache] Skipping duplicate cache_key: {cache_key}")

        # Phase 2: Build modified images for containers with dockerfile_extra
        # This ensures modified images are built during build_only mode
        if task_setups_list:
            await self._build_all_modified_images(task_setups_list)

    async def _build_all_modified_images(self, task_setups: list[SandboxTaskSetup]) -> None:
        """Build all modified images for containers with dockerfile_extra.

        This is called during batch setup to ensure modified images are built
        even in build_only mode where task execution is skipped.

        Args:
            task_setups: All task setups from the batch
        """
        # Track unique modified images to avoid duplicate builds
        # Key: (base_cache_key, dockerfile_extra_hash)
        seen_modified: set[tuple[str, str]] = set()

        for task_setup in task_setups:
            # Check agent container
            await self._maybe_build_modified_image(
                task_setup.agent_container,
                task_setup.task_id,
                seen_modified,
            )

            # Check service containers
            for container_setup in task_setup.service_containers.values():
                await self._maybe_build_modified_image(
                    container_setup,
                    task_setup.task_id,
                    seen_modified,
                )

    async def _maybe_build_modified_image(
        self,
        container_setup: ContainerSetup,
        task_id: str,
        seen_modified: set[tuple[str, str]],
    ) -> None:
        """Build a modified image if dockerfile_extra is present and not already built.

        Args:
            container_setup: Container setup with potential dockerfile_extra
            task_id: Task identifier for naming the modified image
            seen_modified: Set of already-built modified image keys
        """
        if not container_setup.dockerfile_extra:
            return

        # Create dedup key
        base_cache_key = self._get_cache_key(container_setup.spec.image_spec)
        extra_hash = hashlib.sha256(container_setup.dockerfile_extra.encode()).hexdigest()[:12]
        modified_key = (base_cache_key, extra_hash)

        if modified_key in seen_modified:
            logger.debug(
                f"[ImageCache] Skipping duplicate modified image for '{container_setup.name}' "
                f"in task '{task_id}'"
            )
            return

        seen_modified.add(modified_key)
        logger.debug(
            f"[ImageCache] Building modified image for '{container_setup.name}' "
            f"in task '{task_id}' with dockerfile_extra"
        )

        # This will build and cache the modified image
        await self.get_image_for_container(container_setup, task_id)

    async def get_image_for_container(
        self,
        container_setup: ContainerSetup,
        task_id: str,
    ) -> ImageTag:
        """Get the image tag for a container, handling dockerfile_extra if present.

        Args:
            container_setup: Container setup with image spec and optional dockerfile_extra
            task_id: Task identifier for naming modified images

        Returns:
            Image tag to use for container creation
        """
        # Get base image (should already be cached from ensure_all_base_images)
        base_tag = await self._get_base_image(container_setup.spec.image_spec)

        # If no modifications, return base image
        if not container_setup.dockerfile_extra:
            return base_tag

        # Extract platform from image spec if available
        image_spec = container_setup.spec.image_spec
        platform: str | None = None
        if isinstance(image_spec, BuildImageSpec):
            platform = image_spec.platform
        elif isinstance(image_spec, MultiStageBuildImageSpec) and image_spec.stages:
            # Use platform from the last stage (final image)
            platform = image_spec.stages[-1].platform

        # Build modified image with dockerfile_extra
        return await self._get_modified_image(
            base_tag=base_tag,
            dockerfile_extra=container_setup.dockerfile_extra,
            task_id=task_id,
            container_name=container_setup.name,
            platform=platform,
        )

    async def _prepare_base_image(self, spec: ImageSpec) -> ImageTag:
        """Prepare a base image by pulling or building.

        Args:
            spec: Image specification

        Returns:
            Image tag
        """
        cache_key = self._get_cache_key(spec)
        logger.debug(f"[ImageCache] _prepare_base_image called for cache_key: {cache_key}")

        # Check cache first
        if cache_key in self._base_image_cache:
            logger.debug(f"[ImageCache] Found cached image for cache_key: {cache_key}")
            return self._base_image_cache[cache_key]

        logger.debug(
            f"[ImageCache] No cached image found, preparing new image for spec type: {type(spec).__name__}"
        )

        tag: ImageTag = ""

        # Prepare image based on type
        match spec:
            case PullImageSpec():
                logger.debug(f"[ImageCache] Pulling image: {spec.tag}")
                tag = await self._pull_image(spec)
            case BuildImageSpec():
                logger.debug(
                    f"[ImageCache] Building image from spec: tag={spec.tag}, context={spec.context_path}, dockerfile={spec.dockerfile_path}"
                )
                tag = await self._build_image(spec)
            case MultiStageBuildImageSpec():
                logger.debug(f"[ImageCache] Building multi-stage image: {spec.final_tag}")
                tag = await self._build_multi_stage_image(spec)
            case _:
                assert_never(spec)

        # Cache result
        logger.debug(f"[ImageCache] Caching result for cache_key: {cache_key}, tag: {tag}")
        self._base_image_cache[cache_key] = tag
        return tag

    async def _get_base_image(self, spec: ImageSpec) -> ImageTag:
        """Get a cached base image (assumes ensure_all_base_images was called).

        Args:
            spec: Image specification

        Returns:
            Cached image tag
        """
        cache_key = self._get_cache_key(spec)
        if cache_key not in self._base_image_cache:
            # Fallback: prepare if not cached
            return await self._prepare_base_image(spec)
        return self._base_image_cache[cache_key]

    async def _get_modified_image(
        self,
        base_tag: ImageTag,
        dockerfile_extra: str,
        task_id: str,
        container_name: str,
        platform: str | None = None,
    ) -> ImageTag:
        """Get or build a modified image with dockerfile_extra applied.

        Args:
            base_tag: Base image tag to modify
            dockerfile_extra: Additional Dockerfile instructions
            task_id: Task identifier for naming
            container_name: Container name for naming
            platform: Target platform (e.g., 'linux/amd64', 'linux/arm64')

        Returns:
            Modified image tag
        """
        # Create cache key from base tag + dockerfile_extra content
        extra_hash = hashlib.sha256(dockerfile_extra.encode()).hexdigest()[:12]
        cache_key = (base_tag, extra_hash)

        # Check cache
        if cache_key in self._modified_image_cache:
            return self._modified_image_cache[cache_key]

        # Generate stable tag for modified image (without batch_id for cross-run consistency)
        # Format: [registry/]modified-{base_tag_sanitized}__{task_id_sanitized}-{container_name}:latest
        # Strip registry prefix from base_tag if present to avoid duplication when prepending later
        base_tag_for_name = base_tag
        if self._registry_prefix and base_tag.startswith(self._registry_prefix + "/"):
            base_tag_for_name = base_tag[len(self._registry_prefix) + 1 :]

        base_tag_sanitized = base_tag_for_name.replace(":", "-").replace("/", "-")
        safe_task_id = task_id.replace(":", "-").replace("/", "-")
        image_name = f"modified-{base_tag_sanitized}__{safe_task_id}-{container_name}:latest"

        # Prepend registry prefix if configured
        if self._registry_prefix:
            modified_tag = f"{self._registry_prefix}/{image_name}"
        else:
            modified_tag = image_name

        # Check if modified image already exists (e.g., from previous build or pulled in run_from_prebuilt)
        try:
            await self._docker.inspect_image(modified_tag)
            logger.debug(f"[ImageCache] Modified image {modified_tag} already exists locally")
            # Cache and return existing image
            self._modified_image_cache[cache_key] = modified_tag
            return modified_tag
        except DockerClientError:
            logger.debug(f"[ImageCache] Modified image {modified_tag} not found locally")

        # In run_from_prebuilt mode, pull the image instead of building
        if self._execution_mode == "run_from_prebuilt":
            logger.debug(f"[ImageCache] run_from_prebuilt mode: pulling {modified_tag}")
            try:
                await self._docker.pull_image(modified_tag)
                logger.debug(f"[ImageCache] Successfully pulled modified image {modified_tag}")
                self._modified_image_cache[cache_key] = modified_tag
                return modified_tag
            except DockerClientError as e:
                raise DockerClientError(
                    f"Failed to pull prebuilt modified image {modified_tag}. "
                    f"Ensure the image was pushed to the registry.",
                    getattr(e, "stdout", ""),
                    getattr(e, "stderr", ""),
                ) from e

        # Create temporary build context with Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dockerfile_path = temp_path / "Dockerfile"

            # Write Dockerfile with base image + extra instructions
            dockerfile_content = f"FROM {base_tag}\n{dockerfile_extra}"
            dockerfile_path.write_text(dockerfile_content)

            # Build the modified image
            await self._build_from_context(
                context_path=str(temp_path),
                tag=modified_tag,
                dockerfile_path="Dockerfile",
                platform=platform,
            )
            self._track_built_image(modified_tag)

        # Cache result
        self._modified_image_cache[cache_key] = modified_tag
        return modified_tag

    async def _pull_image(self, spec: PullImageSpec) -> ImageTag:
        """Pull an image from a registry.

        Args:
            spec: Pull specification

        Returns:
            Image tag
        """
        logger.debug(f"[ImageCache] _pull_image: Checking if image exists locally: {spec.tag}")
        try:
            # Check if image exists locally
            await self._docker.inspect_image(spec.tag)
            logger.debug(f"[ImageCache] _pull_image: Image {spec.tag} already exists locally")
        except DockerClientError as e:
            # Image doesn't exist, pull it
            logger.debug(
                f"[ImageCache] _pull_image: Image {spec.tag} not found locally (error: {e}), pulling..."
            )
            await self._docker.pull_image(spec.tag)
            logger.debug(f"[ImageCache] _pull_image: Successfully pulled image {spec.tag}")
        return spec.tag

    def _track_built_image(self, tag: ImageTag) -> None:
        """Track a built image and write to file if configured.

        Args:
            tag: Image tag that was built
        """
        if tag in self._built_images:
            return  # Already tracked

        self._built_images.add(tag)

        if self._built_images_file:
            try:
                # Append the tag to the file
                with self._built_images_file.open("a") as f:
                    f.write(f"{tag}\n")
                logger.debug(f"[ImageCache] Recorded built image: {tag}")
            except Exception as e:
                logger.warning(f"[ImageCache] Failed to write built image {tag} to file: {e}")

    async def _build_image(self, spec: BuildImageSpec) -> ImageTag:
        """Build an image from a Dockerfile.

        Args:
            spec: Build specification

        Returns:
            Image tag
        """
        logger.debug(f"[ImageCache] _build_image: Checking if image exists: {spec.tag}")
        logger.debug(f"[ImageCache] _build_image: Build context path: {spec.context_path}")
        logger.debug(f"[ImageCache] _build_image: Dockerfile path: {spec.dockerfile_path}")
        logger.debug(f"[ImageCache] _build_image: Build args: {spec.build_args}")

        # Check if image exists
        try:
            logger.debug(f"[ImageCache] _build_image: Attempting to inspect image {spec.tag}")
            await self._docker.inspect_image(spec.tag)
            logger.debug(
                f"[ImageCache] _build_image: Image {spec.tag} already exists, skipping build"
            )
            return spec.tag
        except DockerClientError as e:
            logger.debug(
                f"[ImageCache] _build_image: Image {spec.tag} not found (error: {e}), proceeding with build"
            )
            # Image doesn't exist, build it

        logger.debug(f"[ImageCache] _build_image: Starting build for image {spec.tag}")
        await self._build_from_context(
            context_path=spec.context_path,
            tag=spec.tag,
            dockerfile_path=spec.dockerfile_path,
            build_args=spec.build_args,
            platform=spec.platform,
        )
        logger.debug(f"[ImageCache] _build_image: Successfully built image {spec.tag}")
        self._track_built_image(spec.tag)
        return spec.tag

    async def _build_multi_stage_image(self, spec: MultiStageBuildImageSpec) -> ImageTag:
        """Build a multi-stage image with caching.

        Args:
            spec: Multi-stage build specification

        Returns:
            Final image tag
        """
        for stage in spec.stages:
            # Check if stage image exists
            try:
                await self._docker.inspect_image(stage.tag)
                continue  # Stage already built
            except DockerClientError:
                pass  # Need to build this stage

            # Prepare build args
            build_args = dict(stage.build_args) if stage.build_args else {}

            # Add parent tag as BASE_IMAGE build arg if specified
            if stage.parent_tag:
                build_args["BASE_IMAGE"] = stage.parent_tag

            # Build stage
            await self._build_from_context(
                context_path=stage.context_path,
                tag=stage.tag,
                dockerfile_path=stage.dockerfile_path,
                build_args=build_args if build_args else None,
                platform=stage.platform,
            )
            self._track_built_image(stage.tag)

        return spec.final_tag

    async def _build_from_context(
        self,
        context_path: str,
        tag: str,
        dockerfile_path: str | None = None,
        build_args: dict[str, str] | None = None,
        platform: str | None = None,
    ) -> None:
        """Build a Docker image from a build context.

        The tar archive creation and file transfer is handled by the
        Docker client implementation.

        Args:
            context_path: Path to the build context directory
            tag: Tag for the built image
            dockerfile_path: Path to Dockerfile relative to context
            build_args: Build arguments
            platform: Target platform (e.g., 'linux/amd64', 'linux/arm64')

        Raises:
            ImageBuildError: If build fails
        """
        logger.debug(
            f"[ImageCache] _build_from_context: Building image {tag} from context {context_path}"
        )
        logger.debug(f"[ImageCache] _build_from_context: Dockerfile path: {dockerfile_path}")
        logger.debug(f"[ImageCache] _build_from_context: Build args: {build_args}")

        errors = []

        # Stream build output from client
        async for log_line in self._docker.build_image(
            context_path=context_path,
            tag=tag,
            dockerfile_path=dockerfile_path,
            buildargs=build_args,
            platform=platform,
        ):
            logger.debug(f"[ImageCache] _build_from_context: Build log: {log_line}")
            if "error" in log_line:
                error = log_line["error"]
                logger.error(f"[ImageCache] _build_from_context: Build error: {error}")
                errors.append(error)

        if errors:
            logger.error(
                f"[ImageCache] _build_from_context: Build failed with {len(errors)} errors"
            )
            raise ImageBuildError(tag, errors)

        logger.debug(f"[ImageCache] _build_from_context: Build completed successfully for {tag}")

    @staticmethod
    def _get_cache_key(spec: ImageSpec) -> str:
        """Generate a cache key for an image spec.

        Args:
            spec: Image specification

        Returns:
            Cache key string
        """
        match spec:
            case PullImageSpec():
                return f"pull:{spec.tag}"
            case BuildImageSpec():
                # Include all relevant fields for uniqueness
                key_parts = [
                    "build",
                    spec.context_path,
                    spec.tag,
                    spec.dockerfile_path or "Dockerfile",
                ]
                if spec.build_args:
                    key_parts.append(str(sorted(spec.build_args.items())))
                return ":".join(key_parts)
            case MultiStageBuildImageSpec():
                # Use final tag and all stage cache keys
                stage_keys = [stage.cache_key or stage.tag for stage in spec.stages]
                return f"multistage:{spec.final_tag}:{'-'.join(stage_keys)}"
            case _:
                assert_never(spec)

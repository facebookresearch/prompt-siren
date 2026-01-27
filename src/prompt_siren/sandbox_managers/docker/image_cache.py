# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image caching for Docker sandbox management.

All images should be pre-built using the build_images script before running experiments.
This module only handles pulling/verifying pre-built images, NO runtime building.
"""

from __future__ import annotations

import logging

from typing_extensions import assert_never

from ..image_spec import (
    BuildImageSpec,
    DerivedImageSpec,
    ImageSpec,
    ImageTag,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from ..sandbox_task_setup import ContainerSetup
from .plugins import AbstractDockerClient, DockerClientError

logger = logging.getLogger(__name__)


class ImageNotFoundError(Exception):
    """Raised when an expected pre-built image is not found."""

    def __init__(self, tag: str):
        self.tag = tag
        super().__init__(
            f"Image '{tag}' not found. All images must be pre-built using "
            "the `prompt-siren-build-images` command before running experiments."
        )


class ImageCache:
    """Manages image caching for a batch.

    All images should be pre-built using the build_images script.
    This class only handles pulling/verifying pre-built images, NO runtime building.
    """

    def __init__(self, docker: AbstractDockerClient, batch_id: str):
        """Initialize image cache.

        Args:
            docker: Abstract Docker client for all operations
            batch_id: Unique identifier for this batch
        """
        self._docker = docker
        self._batch_id = batch_id
        # Map from cache key to prepared image tag
        self._image_cache: dict[str, ImageTag] = {}

    async def ensure_all_base_images(self, container_setups: list[ContainerSetup]) -> None:
        """Ensure all base images are available before task execution.

        Verifies all unique images exist (pulls from registry if needed for PullImageSpec).

        Args:
            container_setups: All container setups from all tasks in the batch
        """
        logger.debug(
            f"[ImageCache] ensure_all_base_images called with {len(container_setups)} container setups"
        )
        # Collect unique image specs using cache key
        seen_specs: set[str] = set()
        for setup in container_setups:
            cache_key = self._get_cache_key(setup.spec.image_spec)
            logger.debug(
                f"[ImageCache] Processing setup '{setup.name}' with cache_key: {cache_key}"
            )
            if cache_key not in seen_specs:
                seen_specs.add(cache_key)
                logger.debug(f"[ImageCache] Preparing image for cache_key: {cache_key}")
                await self._ensure_image_available(setup.spec.image_spec)
            else:
                logger.debug(f"[ImageCache] Skipping duplicate cache_key: {cache_key}")

    async def get_image_for_container(
        self,
        container_setup: ContainerSetup,
    ) -> ImageTag:
        """Get the image tag for a container.

        All images should be pre-built. This method only retrieves
        the cached image tag.

        Args:
            container_setup: Container setup with image spec

        Returns:
            Image tag to use for container creation
        """
        return await self._get_image(container_setup.spec.image_spec)

    async def _ensure_image_available(self, spec: ImageSpec) -> ImageTag:
        """Ensure an image is available locally (pull if needed, verify exists).

        Args:
            spec: Image specification

        Returns:
            Image tag

        Raises:
            ImageNotFoundError: If a pre-built image is not found
        """
        cache_key = self._get_cache_key(spec)
        logger.debug(f"[ImageCache] _ensure_image_available called for cache_key: {cache_key}")

        # Check cache first
        if cache_key in self._image_cache:
            logger.debug(f"[ImageCache] Found cached image for cache_key: {cache_key}")
            return self._image_cache[cache_key]

        logger.debug(
            f"[ImageCache] No cached image found, checking availability for spec type: "
            f"{type(spec).__name__}"
        )

        # Handle image based on type
        match spec:
            case PullImageSpec():
                logger.debug(f"[ImageCache] Pulling/verifying image: {spec.tag}")
                tag = await self._pull_image(spec)
            case BuildImageSpec():
                # BuildImageSpec should have been pre-built
                logger.debug(f"[ImageCache] Verifying pre-built image exists: {spec.tag}")
                tag = await self._verify_image_exists(spec.tag)
            case MultiStageBuildImageSpec():
                # MultiStageBuildImageSpec should have been pre-built
                logger.debug(
                    f"[ImageCache] Verifying pre-built multi-stage image exists: {spec.final_tag}"
                )
                tag = await self._verify_image_exists(spec.final_tag)
            case DerivedImageSpec():
                # DerivedImageSpec should have been pre-built (pair images)
                logger.debug(f"[ImageCache] Verifying pre-built derived image exists: {spec.tag}")
                tag = await self._verify_image_exists(spec.tag)
            case _:
                assert_never(spec)

        # Cache result
        logger.debug(f"[ImageCache] Caching result for cache_key: {cache_key}, tag: {tag}")
        self._image_cache[cache_key] = tag
        return tag

    async def _get_image(self, spec: ImageSpec) -> ImageTag:
        """Get a cached image (assumes ensure_all_base_images was called).

        Args:
            spec: Image specification

        Returns:
            Cached image tag
        """
        cache_key = self._get_cache_key(spec)
        if cache_key not in self._image_cache:
            # Fallback: ensure available if not cached
            return await self._ensure_image_available(spec)
        return self._image_cache[cache_key]

    async def _pull_image(self, spec: PullImageSpec) -> ImageTag:
        """Pull an image from a registry (or verify it exists locally).

        Args:
            spec: Pull specification

        Returns:
            Image tag

        Raises:
            DockerClientError: If a non-404 error occurs (e.g., daemon not running)
        """
        logger.debug(f"[ImageCache] _pull_image: Checking if image exists locally: {spec.tag}")
        try:
            # Check if image exists locally
            await self._docker.inspect_image(spec.tag)
            logger.debug(f"[ImageCache] _pull_image: Image {spec.tag} already exists locally")
            return spec.tag
        except DockerClientError as e:
            if e.status != 404:
                # Not a "not found" error - could be daemon down, permission denied, etc.
                logger.error(f"[ImageCache] Docker error while checking image '{spec.tag}': {e}")
                raise
            # Image doesn't exist (404), pull it
            logger.debug(
                f"[ImageCache] _pull_image: Image {spec.tag} not found locally, pulling..."
            )
            await self._docker.pull_image(spec.tag)
            logger.debug(f"[ImageCache] _pull_image: Successfully pulled image {spec.tag}")
        return spec.tag

    async def _verify_image_exists(self, tag: str) -> ImageTag:
        """Verify a pre-built image exists locally.

        Args:
            tag: Image tag to verify

        Returns:
            Image tag if exists

        Raises:
            ImageNotFoundError: If image doesn't exist (404)
            DockerClientError: If a non-404 error occurs (e.g., daemon not running)
        """
        logger.debug(f"[ImageCache] _verify_image_exists: Checking if image exists: {tag}")
        try:
            await self._docker.inspect_image(tag)
            logger.debug(f"[ImageCache] _verify_image_exists: Image {tag} exists")
            return tag
        except DockerClientError as e:
            if e.status == 404:
                logger.error(
                    f"[ImageCache] Image {tag} not found. "
                    "Please run `prompt-siren-build-images` to build all required images."
                )
                raise ImageNotFoundError(tag) from e
            # Non-404 error - could be daemon down, permission denied, etc.
            logger.error(f"[ImageCache] Docker error verifying image {tag}: {e}")
            raise

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
                # For pre-built images, just use the tag
                return f"build:{spec.tag}"
            case MultiStageBuildImageSpec():
                # For pre-built multi-stage images, just use the final tag
                return f"multistage:{spec.final_tag}"
            case DerivedImageSpec():
                # For pre-built derived images, use the tag
                return f"derived:{spec.tag}"
            case _:
                assert_never(spec)

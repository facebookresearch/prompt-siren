# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Image caching and building for Docker sandbox management."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
from pathlib import Path

from typing_extensions import assert_never

try:
    import aiodocker
    from aiodocker import DockerError
except ImportError as e:
    raise ImportError(
        "Docker sandbox manager requires the 'docker' optional dependency. "
        "Install with: pip install 'prompt-siren[docker]'"
    ) from e

from ..image_spec import (
    BuildImageSpec,
    ImageSpec,
    ImageTag,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from ..sandbox_task_setup import ContainerSetup


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

    def __init__(self, docker: aiodocker.Docker, batch_id: str):
        """Initialize image cache.

        Args:
            docker: Docker client for all operations
            batch_id: Unique identifier for this batch
        """
        self._docker = docker
        self._batch_id = batch_id
        # Map from cache key to prepared image tag
        self._base_image_cache: dict[str, ImageTag] = {}
        # Map from (base_tag, dockerfile_extra_hash) to modified image tag
        self._modified_image_cache: dict[tuple[str, str], ImageTag] = {}

    async def ensure_all_base_images(self, container_setups: list[ContainerSetup]) -> None:
        """Ensure all base images are available before task execution.

        Builds/pulls all unique base images sequentially to avoid race conditions.

        Args:
            container_setups: All container setups from all tasks in the batch
        """
        # Collect unique image specs using cache key
        seen_specs: set[str] = set()
        for setup in container_setups:
            cache_key = self._get_cache_key(setup.spec.image_spec)
            if cache_key not in seen_specs:
                seen_specs.add(cache_key)
                await self._prepare_base_image(setup.spec.image_spec)

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

        # Build modified image with dockerfile_extra
        return await self._get_modified_image(
            base_tag=base_tag,
            dockerfile_extra=container_setup.dockerfile_extra,
            task_id=task_id,
            container_name=container_setup.name,
        )

    async def _prepare_base_image(self, spec: ImageSpec) -> ImageTag:
        """Prepare a base image by pulling or building.

        Args:
            spec: Image specification

        Returns:
            Image tag
        """
        cache_key = self._get_cache_key(spec)

        # Check cache first
        if cache_key in self._base_image_cache:
            return self._base_image_cache[cache_key]

        # Prepare image based on type
        match spec:
            case PullImageSpec():
                tag = await self._pull_image(spec)
            case BuildImageSpec():
                tag = await self._build_image(spec)
            case MultiStageBuildImageSpec():
                tag = await self._build_multi_stage_image(spec)
            case _:
                assert_never(spec)

        # Cache result
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
    ) -> ImageTag:
        """Get or build a modified image with dockerfile_extra applied.

        Args:
            base_tag: Base image tag to modify
            dockerfile_extra: Additional Dockerfile instructions
            task_id: Task identifier for naming
            container_name: Container name for naming

        Returns:
            Modified image tag
        """
        # Create cache key from base tag + dockerfile_extra content
        extra_hash = hashlib.sha256(dockerfile_extra.encode()).hexdigest()[:12]
        cache_key = (base_tag, extra_hash)

        # Check cache
        if cache_key in self._modified_image_cache:
            return self._modified_image_cache[cache_key]

        # Build modified image
        safe_task_id = task_id.replace(":", "-").replace("/", "-")
        modified_tag = f"modified-{self._batch_id}-{safe_task_id}-{container_name}:latest"

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
            )

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
        try:
            # Check if image exists locally
            await self._docker.images.inspect(spec.tag)
        except DockerError:
            # Image doesn't exist, pull it
            await self._docker.images.pull(from_image=spec.tag)
        return spec.tag

    async def _build_image(self, spec: BuildImageSpec) -> ImageTag:
        """Build an image from a Dockerfile.

        Args:
            spec: Build specification

        Returns:
            Image tag
        """
        # Check if image exists
        try:
            await self._docker.images.inspect(spec.tag)
            return spec.tag
        except DockerError:
            pass  # Image doesn't exist, build it

        await self._build_from_context(
            context_path=spec.context_path,
            tag=spec.tag,
            dockerfile_path=spec.dockerfile_path,
            build_args=spec.build_args,
        )
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
                await self._docker.images.inspect(stage.tag)
                continue  # Stage already built
            except DockerError:
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
            )

        return spec.final_tag

    async def _build_from_context(
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
            ImageBuildError: If build fails
        """
        # Create tar archive of build context
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(context_path, arcname=".")
        tar_stream.seek(0)

        errors = []

        # Stream build output
        async for log_line in self._docker.images.build(
            fileobj=tar_stream,
            tag=tag,
            encoding="application/x-tar",
            stream=True,
            path_dockerfile=dockerfile_path or "Dockerfile",
            buildargs=build_args,
        ):
            if "error" in log_line:
                error = log_line["error"]
                errors.append(error)

        if errors:
            raise ImageBuildError(tag, errors)

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

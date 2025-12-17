# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Docker build context preparation for SWE-bench instances."""

import hashlib
from pathlib import Path

try:
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.test_spec.test_spec import TestSpec
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

from ...sandbox_managers.image_spec import (
    BuildStage,
    ImageSpec,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from .config import SwebenchDatasetConfig
from .constants import INSTANCE_INJECTION_MAPPING
from .dockerfiles import (
    _DOCKERFILE_ENV_PY,
    _DOCKERFILE_INSTANCE_PY,
    DOCKERFILE_TEMPLATE,
)
from .swebench_imports import _ACTIVATE_ENV_COMMAND, make_test_spec


def prepare_build_context(
    instance: SWEbenchInstance,
    config: SwebenchDatasetConfig,
    execution_mode: str = "build_and_run",
    registry_prefix: str | None = None,
    platform: str = "linux/amd64",
) -> tuple[ImageSpec, TestSpec]:
    """Prepare Docker image specification for a SWE-bench instance based on execution mode.

    The function behavior depends on execution_mode:
    - "build_and_run": Creates MultiStageBuildImageSpec with 3 stages (base, env, instance)
    - "build_only": Same as "build_and_run" (builds but won't execute)
    - "run_from_prebuilt": Creates PullImageSpec to pull prebuilt image from registry

    Args:
        instance: SWE-bench instance data
        config: Dataset configuration (cache_dir, use_cache, etc.)
        execution_mode: Execution mode from sandbox manager config
        registry_prefix: Optional registry prefix from sandbox manager config
        platform: Docker platform from sandbox manager config

    Returns:
        Tuple of (ImageSpec, TestSpec) where ImageSpec is either MultiStageBuildImageSpec
        or PullImageSpec depending on execution mode, and TestSpec contains metadata for evaluation.

    Raises:
        ValueError: If execution_mode is "run_from_prebuilt" but registry_prefix is not set
    """
    # Look up injection spec for this instance
    instance_id = instance["instance_id"]
    if instance_id not in INSTANCE_INJECTION_MAPPING:
        raise RuntimeError(
            f"The given instance '{instance_id}' does not have a location to place an injection."
        )

    injection_spec = INSTANCE_INJECTION_MAPPING[instance_id]

    # Generate scripts and metadata using SWE-bench
    test_spec = make_test_spec(instance, injection_spec=injection_spec)

    # Handle run_from_prebuilt mode
    if execution_mode == "run_from_prebuilt":
        if not registry_prefix:
            full_tag = test_spec.instance_image_key
        else:
            # Construct full image tag from registry prefix (if available) and instance image key
            full_tag = f"{registry_prefix}/{test_spec.instance_image_key}"
        return (PullImageSpec(tag=full_tag), test_spec)

    # For build_and_run and build_only modes, create multi-stage build spec
    # Create cache directory structure
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stages = []

    # Stage 1: Base image
    # Shared across all instances - use test_spec.base_image_key for caching
    base_key_hash = hashlib.sha256(test_spec.base_image_key.encode()).hexdigest()[:16]
    base_context = cache_dir / "base" / base_key_hash
    base_context.mkdir(parents=True, exist_ok=True)

    # Only write if not using cache or doesn't exist
    if not config.use_cache or not (base_context / "Dockerfile").exists():
        (base_context / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)

    stages.append(
        BuildStage(
            tag=test_spec.base_image_key,
            context_path=str(base_context),
            parent_tag=None,  # FROM ghcr.io/astral-sh/uv:bookworm-slim
            cache_key=test_spec.base_image_key,  # Reuse across same base
            platform=platform,
        )
    )

    # Stage 2: Environment image
    # Shared across instances with same dependencies - use test_spec.env_image_key for caching
    env_key_hash = hashlib.sha256(test_spec.env_image_key.encode()).hexdigest()[:16]
    env_context = cache_dir / "env" / env_key_hash
    env_context.mkdir(parents=True, exist_ok=True)

    # Only write if not using cache or doesn't exist
    if not config.use_cache or not (env_context / "Dockerfile").exists():
        env_dockerfile = _DOCKERFILE_ENV_PY.format(
            base_image_key=test_spec.base_image_key,
            activate_env_command=_ACTIVATE_ENV_COMMAND,
        )
        (env_context / "Dockerfile").write_text(env_dockerfile)
        (env_context / "setup_env.sh").write_text(test_spec.setup_env_script)

    stages.append(
        BuildStage(
            tag=test_spec.env_image_key,
            context_path=str(env_context),
            parent_tag=test_spec.base_image_key,
            cache_key=test_spec.env_image_key,  # Reuse across same env
            platform=platform,
        )
    )

    # Stage 3: Instance image
    # Unique per instance - use instance_id for directory
    instance_context = (
        cache_dir / "instance" / instance["instance_id"].replace("/", "_").replace(":", "_")
    )
    instance_context.mkdir(parents=True, exist_ok=True)

    # Only write if not using cache or doesn't exist
    if not config.use_cache or not (instance_context / "Dockerfile").exists():
        instance_dockerfile = _DOCKERFILE_INSTANCE_PY.format(
            env_image_name=test_spec.env_image_key,
        )
        (instance_context / "Dockerfile").write_text(instance_dockerfile)
        (instance_context / "setup_repo.sh").write_text(test_spec.install_repo_script)
        # Also save eval script for later use
        (instance_context / "eval.sh").write_text(test_spec.eval_script)

    stages.append(
        BuildStage(
            tag=test_spec.instance_image_key,
            context_path=str(instance_context),
            parent_tag=test_spec.env_image_key,
            cache_key=None,  # Always rebuild instances
            platform=platform,
        )
    )

    return (
        MultiStageBuildImageSpec(
            stages=stages,
            final_tag=test_spec.instance_image_key,
        ),
        test_spec,
    )

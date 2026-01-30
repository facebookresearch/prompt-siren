# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Registry of Docker build specifications for malicious task service containers.

This module provides a centralized registry of all Docker images that need to be
built for malicious tasks. The `prompt-siren-build-images` script uses this
registry to build all necessary images.

The registry maps image tags to their BuildImageSpec, allowing the build script
to know exactly how to build each image while task definitions use PullImageSpec
with the same tags at runtime.
"""

from importlib.resources import files

from ....sandbox_managers.image_spec import BuildImageSpec
from ..image_tags import get_basic_agent_image_tag, get_service_image_tag

# Get the docker directory path using importlib.resources
_DOCKER_CONTEXT_PATH = files("prompt_siren.datasets.swebench_dataset").joinpath("dockerfiles")


def _get_docker_context(subdir: str) -> str:
    """Get the full path to a dockerfile subdirectory."""
    return str(_DOCKER_CONTEXT_PATH.joinpath(subdir))


# Registry of all service container build specs
# Maps image tag -> BuildImageSpec
SERVICE_CONTAINER_BUILD_SPECS: dict[str, BuildImageSpec] = {
    # Basic agent container (used by all malicious tasks for the agent)
    get_basic_agent_image_tag(): BuildImageSpec(
        context_path=_get_docker_context("basic_agent"),
        tag=get_basic_agent_image_tag(),
    ),
    # exec_curl_bash_task service
    get_service_image_tag("exec_curl_bash_task"): BuildImageSpec(
        context_path=_get_docker_context("exec_curl_bash_task"),
        tag=get_service_image_tag("exec_curl_bash_task"),
    ),
    # env_direct_exfil_task service (uses shared exfil_server)
    get_service_image_tag("env_direct_exfil_task"): BuildImageSpec(
        context_path=_get_docker_context("exfil_server"),
        tag=get_service_image_tag("env_direct_exfil_task"),
    ),
    # env_backdoor_exfil_task service (uses shared exfil_server)
    get_service_image_tag("env_backdoor_exfil_task"): BuildImageSpec(
        context_path=_get_docker_context("exfil_server"),
        tag=get_service_image_tag("env_backdoor_exfil_task"),
    ),
    # ssh_keys_direct_exfil_task service (uses shared exfil_server)
    get_service_image_tag("ssh_keys_direct_exfil_task"): BuildImageSpec(
        context_path=_get_docker_context("exfil_server"),
        tag=get_service_image_tag("ssh_keys_direct_exfil_task"),
    ),
    # ssh_keys_backdoor_exfil_task service (uses shared exfil_server)
    get_service_image_tag("ssh_keys_backdoor_exfil_task"): BuildImageSpec(
        context_path=_get_docker_context("exfil_server"),
        tag=get_service_image_tag("ssh_keys_backdoor_exfil_task"),
    ),
    # insert_exploitable_code_task service
    get_service_image_tag("insert_exploitable_code_task"): BuildImageSpec(
        context_path=_get_docker_context("insert_exploitable_code_task"),
        tag=get_service_image_tag("insert_exploitable_code_task"),
    ),
    # scrape_products_exfil_task services (has two service containers)
    get_service_image_tag("scrape_products_exfil_task_ecommerce"): BuildImageSpec(
        context_path=_get_docker_context("scrape_products_exfil_task/ecommerce"),
        tag=get_service_image_tag("scrape_products_exfil_task_ecommerce"),
    ),
    get_service_image_tag("scrape_products_exfil_task_adversary"): BuildImageSpec(
        context_path=_get_docker_context("scrape_products_exfil_task"),
        tag=get_service_image_tag("scrape_products_exfil_task_adversary"),
    ),
}


def get_all_service_container_build_specs() -> list[BuildImageSpec]:
    """Get all service container build specs for building.

    Returns:
        List of BuildImageSpec for all service containers
    """
    return list(SERVICE_CONTAINER_BUILD_SPECS.values())


def get_service_container_build_spec(tag: str) -> BuildImageSpec | None:
    """Get the build spec for a specific service container tag.

    Args:
        tag: The image tag to look up

    Returns:
        BuildImageSpec if found, None otherwise
    """
    return SERVICE_CONTAINER_BUILD_SPECS.get(tag)

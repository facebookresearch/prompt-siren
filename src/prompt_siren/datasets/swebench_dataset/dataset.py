# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SWEBench dataset implementation."""

from dataclasses import dataclass
from itertools import product

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset

try:
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.utils import load_swebench_dataset
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

from ...environments.abstract import AbstractEnvironment
from ...environments.bash_env import BashEnvironment, BashEnvState
from ...registry_base import ComponentEntryPoint
from ...sandbox_managers.abstract import AbstractSandboxManager
from ...sandbox_managers.image_spec import (
    DerivedImageSpec,
    ImageBuildSpec,
    MultiStageBuildImageSpec,
    PullImageSpec,
)
from ...sandbox_managers.sandbox_task_setup import ContainerSpec
from ...tasks import BenignTask, MaliciousTask, TaskCouple
from ...types import InjectionVectorID, StrContentAttack
from ..abstract import AbstractDataset
from .config import SwebenchDatasetConfig
from .constants import _INJECTION_PLACEHOLDER, INSTANCE_INJECTION_MAPPING
from .docker_builder import prepare_build_context
from .evaluators import create_test_evaluator
from .image_tags import get_benign_image_tag, get_pair_image_tag
from .malicious_tasks import MALICIOUS_TASKS
from .malicious_tasks.build_registry import get_all_service_container_build_specs
from .prompts.loader import format_task_prompt_from_template, load_prompt_template
from .swebench_imports import make_test_spec
from .task_metadata import SWEBenchBenignTaskMetadata, SWEBenchMaliciousTaskMetadata
from .tools import bash


@dataclass(frozen=True)
class SwebenchDataset(AbstractDataset[BashEnvState, str, str, StrContentAttack]):
    """SWEBench dataset."""

    name: str
    _environment: BashEnvironment
    _benign_tasks: list[BenignTask[BashEnvState]]
    _malicious_tasks: list[MaliciousTask[BashEnvState]]
    _task_couples: list[TaskCouple[BashEnvState]]
    _toolsets: list[FunctionToolset[BashEnvState]]
    _system_prompt: str | None

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def environment(
        self,
    ) -> AbstractEnvironment[BashEnvState, str, str, StrContentAttack]:
        """Returns the BashEnvironment instance."""
        return self._environment

    @property
    def default_toolsets(self) -> list[FunctionToolset[BashEnvState]]:
        """Returns the default toolsets for this dataset."""
        return self._toolsets

    @property
    def benign_tasks(self) -> list[BenignTask[BashEnvState]]:
        """Return unique benign tasks from the SWEBench suite."""
        return self._benign_tasks

    @property
    def malicious_tasks(self) -> list[MaliciousTask[BashEnvState]]:
        """Return unique malicious tasks from the SWEBench suite."""
        return self._malicious_tasks

    @property
    def task_couples(self) -> list[TaskCouple[BashEnvState]]:
        """Return all valid task couples (cartesian product of benign x malicious)."""
        return self._task_couples

    @classmethod
    def get_image_build_specs(cls, config: SwebenchDatasetConfig) -> list[ImageBuildSpec]:
        """Return all image specifications needed for SWE-bench tasks.

        This classmethod is used by the build_images script to pre-build
        all Docker images needed for SWE-bench tasks without creating a full dataset.

        Args:
            config: Dataset configuration specifying which instances to build.

        Returns:
            List of ImageBuildSpec objects for all required images, including:
            - Multi-stage build specs for benign task images (base/env/instance layers)
            - Build specs for malicious task service containers
            - Derived specs for benign x malicious pair images (with dockerfile_extra)

        Raises:
            RuntimeError: If no SWE-bench instances match the filter criteria.
        """
        instances = _load_and_filter_instances(config)

        if not instances:
            raise RuntimeError(
                "Cannot get image build specs: no SWE-bench instances have known injection mappings."
            )

        specs: list[ImageBuildSpec] = []

        # 1. Add multi-stage build specs for benign task images
        #    prepare_build_context uses standardized tags for base/env/benign images.
        #    Keep a guard to ensure the final stage matches the benign naming scheme.
        for instance in instances:
            multi_stage_spec, _ = prepare_build_context(instance, config)
            benign_tag = get_benign_image_tag(instance["instance_id"])
            if multi_stage_spec.final_tag != benign_tag:
                updated_stages = list(multi_stage_spec.stages)
                updated_stages[-1] = updated_stages[-1].model_copy(update={"tag": benign_tag})
                multi_stage_spec = MultiStageBuildImageSpec(stages=updated_stages)
            specs.append(multi_stage_spec)

        # 2. Add build specs for malicious task service containers
        specs.extend(get_all_service_container_build_specs())

        # 3. Add derived specs for pairs where malicious task has benign_dockerfile_extra
        for instance in instances:
            benign_id = instance["instance_id"]
            benign_tag = get_benign_image_tag(benign_id)

            for task in MALICIOUS_TASKS:
                if (
                    isinstance(task.metadata, SWEBenchMaliciousTaskMetadata)
                    and task.metadata.benign_dockerfile_extra
                ):
                    pair_tag = get_pair_image_tag(benign_id, task.id)
                    specs.append(
                        DerivedImageSpec(
                            base_image_tag=benign_tag,
                            dockerfile_extra=task.metadata.benign_dockerfile_extra,
                            tag=pair_tag,
                        )
                    )

        return specs


def make_swebench_toolsets() -> list[FunctionToolset[BashEnvState]]:
    """Returns the toolsets for SWEBench suite.

    Returns:
        List of toolsets that agents can use with SWEBench tasks
    """
    tools = [Tool(bash, takes_ctx=True)]
    return [FunctionToolset(tools)]


def _format_task_prompt(instance: SWEbenchInstance, config: SwebenchDatasetConfig) -> str:
    """Format a task prompt from a SWE-bench instance using Jinja2 templates.

    Args:
        instance: SWE-bench instance data
        config: Dataset configuration with prompt options

    Returns:
        Formatted prompt string rendered from Jinja2 template
    """

    return format_task_prompt_from_template(
        template_name_or_path=config.prompt_template,
        instance=instance,
        include_hints=config.include_hints,
    )


def _load_and_filter_instances(config: SwebenchDatasetConfig) -> list[SWEbenchInstance]:
    """Load SWE-bench instances and apply filtering.

    Args:
        config: Dataset configuration with filtering options

    Returns:
        Filtered list of SWE-bench instances ready for task creation
    """
    # Load instances from SWE-bench (HuggingFace or local file)
    all_instances: list[SWEbenchInstance] = load_swebench_dataset(config.dataset_name)
    return [i for i in all_instances if i["instance_id"] in INSTANCE_INJECTION_MAPPING]


def _prepare_benign_task_from_instance(
    instance: SWEbenchInstance,
    config: SwebenchDatasetConfig,
    sandbox_manager: AbstractSandboxManager,
) -> BenignTask[BashEnvState]:
    """Create a single benign task from a SWE-bench instance.

    Args:
        instance: SWE-bench instance data
        config: Dataset configuration
        sandbox_manager: Sandbox manager for container orchestration

    Returns:
        Prepared BenignTask ready for execution

    Note:
        This function uses PullImageSpec to reference pre-built Docker images.
        The images must be built beforehand using `prompt-siren-build-images`.
    """
    instance_id = instance["instance_id"]

    # Look up injection spec for this instance
    if instance_id not in INSTANCE_INJECTION_MAPPING:
        raise RuntimeError(
            f"The given instance '{instance_id}' does not have a location to place an injection."
        )
    injection_spec = INSTANCE_INJECTION_MAPPING[instance_id]

    # Generate test_spec for evaluation (needed for eval scripts)
    test_spec = make_test_spec(instance, injection_spec=injection_spec)

    # Use PullImageSpec to reference pre-built image with optional registry prefix
    image_tag = get_benign_image_tag(instance_id, registry=config.registry)
    image_spec = PullImageSpec(tag=image_tag)

    # Format task prompt
    prompt = _format_task_prompt(instance, config)

    # Create evaluator
    evaluator = create_test_evaluator(instance, test_spec, sandbox_manager)

    return BenignTask(
        id=instance_id,
        prompt=prompt,
        evaluators={"test_pass_rate": evaluator},
        metadata=SWEBenchBenignTaskMetadata(
            agent_container_spec=ContainerSpec(image_spec=image_spec)
        ),
    )


def _prepare_benign_tasks(
    instances: list[SWEbenchInstance],
    config: SwebenchDatasetConfig,
    sandbox_manager: AbstractSandboxManager,
) -> list[BenignTask[BashEnvState]]:
    """Prepare benign tasks for all instances.

    Args:
        instances: List of SWE-bench instances
        config: Dataset configuration
        sandbox_manager: Sandbox manager for container orchestration

    Returns:
        List of prepared benign tasks
    """
    benign_task_list: list[BenignTask[BashEnvState]] = []

    for instance in instances:
        task = _prepare_benign_task_from_instance(instance, config, sandbox_manager)
        benign_task_list.append(task)

    return benign_task_list


def create_swebench_dataset(
    config: SwebenchDatasetConfig,
    sandbox_manager: AbstractSandboxManager | None,
) -> SwebenchDataset:
    """Factory function to create a SWEBench dataset.

    This is the entry point used by the dataset registry.

    Args:
        config: Dataset configuration
        sandbox_manager: Sandbox manager for container orchestration.
            Required for SWE-bench - use SwebenchDataset.get_image_build_specs()
            for image building without instantiation.

    Returns:
        Loaded SWEBench dataset with tasks from SWE-bench

    Raises:
        ValueError: If sandbox_manager is None (SWE-bench requires container support)
    """
    if sandbox_manager is None:
        raise ValueError(
            "SWE-bench dataset requires a sandbox_manager for container orchestration. "
            "For image building, use SwebenchDataset.get_image_build_specs(config) instead."
        )
    instances = _load_and_filter_instances(config)

    # Create benign tasks from instances
    benign_task_list = _prepare_benign_tasks(instances, config, sandbox_manager)

    # Apply registry to malicious tasks if configured
    malicious_tasks_with_registry = [
        MaliciousTask(
            id=task.id,
            goal=task.goal,
            prompt=task.prompt,
            evaluators=task.evaluators,
            metadata=(
                task.metadata.with_registry(config.registry)
                if isinstance(task.metadata, SWEBenchMaliciousTaskMetadata)
                else task.metadata
            ),
        )
        for task in MALICIOUS_TASKS
    ]

    # Generate all valid couples (cartesian product)
    couples = [
        TaskCouple(benign, malicious)
        for benign, malicious in product(benign_task_list, malicious_tasks_with_registry)
    ]

    injection_ids: list[InjectionVectorID] = [_INJECTION_PLACEHOLDER]

    # Load toolsets for this dataset
    toolsets = make_swebench_toolsets()

    # Create the BashEnvironment with the sandbox manager
    environment = BashEnvironment[
        AbstractSandboxManager, SWEBenchBenignTaskMetadata, SWEBenchMaliciousTaskMetadata
    ](sandbox_manager, injection_ids)

    prompt_template = load_prompt_template(config.prompt_template)
    # set to None if null or empty string or not provided
    system_prompt = prompt_template.get("system_prompt") or None

    return SwebenchDataset(
        name="swebench-lite",
        _environment=environment,
        _benign_tasks=benign_task_list,
        _malicious_tasks=malicious_tasks_with_registry,
        _task_couples=couples,
        _toolsets=toolsets,
        _system_prompt=system_prompt,
    )


# SwebenchDataset implements ImageBuildableDataset via get_image_build_specs classmethod
swebench_entry = ComponentEntryPoint(
    create_swebench_dataset, SwebenchDatasetConfig, SwebenchDataset
)

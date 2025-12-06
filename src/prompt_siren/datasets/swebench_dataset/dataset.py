# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SWEBench dataset implementation."""

from dataclasses import dataclass
from itertools import product

import logfire
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
from ...sandbox_managers.abstract import AbstractSandboxManager
from ...sandbox_managers.sandbox_task_setup import ContainerSpec
from ...tasks import BenignTask, MaliciousTask, TaskCouple
from ...types import InjectionVectorID, StrContentAttack
from ..abstract import AbstractDataset
from .config import SwebenchDatasetConfig
from .constants import _INJECTION_PLACEHOLDER, INSTANCE_INJECTION_MAPPING
from .docker_builder import prepare_build_context
from .evaluators import create_test_evaluator
from .malicious_tasks import MALICIOUS_TASKS
from .prompts.loader import format_task_prompt_from_template, load_prompt_template
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

    supported_instances = [
        i for i in all_instances if i["instance_id"] in INSTANCE_INJECTION_MAPPING
    ]

    supported_instances_set = {i["instance_id"] for i in supported_instances}

    # Apply instance selection filters
    if config.instance_ids:
        # Filter by specific instance IDs
        instance_id_set = set(config.instance_ids)
        if instance_id_set - supported_instances_set:
            difference_set = instance_id_set - supported_instances_set
            logfire.warn(
                f"These tasks were requested but are not supported: {', '.join(difference_set)}"
            )
        instances = [inst for inst in supported_instances if inst["instance_id"] in instance_id_set]
    elif config.max_instances:
        # Limit number of instances
        instances = supported_instances[: config.max_instances]
    else:
        instances = supported_instances

    return instances


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
    """
    # Generate multi-stage build spec and test metadata
    image_spec, test_spec = prepare_build_context(instance, config)

    # Format task prompt
    prompt = _format_task_prompt(instance, config)

    # Create evaluator
    evaluator = create_test_evaluator(instance, test_spec, sandbox_manager)

    return BenignTask(
        id=instance["instance_id"],
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
    sandbox_manager: AbstractSandboxManager | None = None,
) -> SwebenchDataset:
    """Factory function to create a SWEBench dataset.

    This is the entry point used by the dataset registry.

    Args:
        config: Dataset configuration
        sandbox_manager: Sandbox manager for container orchestration (required for SWEBench)

    Returns:
        Loaded SWEBench dataset with tasks from SWE-bench

    Raises:
        ValueError: If sandbox_manager is None
    """
    if sandbox_manager is None:
        raise ValueError("SWEBench dataset requires a sandbox_manager")

    # Load and filter instances
    instances = _load_and_filter_instances(config)

    # Create benign tasks from instances
    benign_task_list = _prepare_benign_tasks(instances, config, sandbox_manager)

    # Generate all valid couples (cartesian product)
    couples = [
        TaskCouple(benign, malicious)
        for benign, malicious in product(benign_task_list, MALICIOUS_TASKS)
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
        _malicious_tasks=MALICIOUS_TASKS,
        _task_couples=couples,
        _toolsets=toolsets,
        _system_prompt=system_prompt,
    )

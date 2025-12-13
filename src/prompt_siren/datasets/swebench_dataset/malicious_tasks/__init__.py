# Copyright (c) Meta Platforms, Inc. and affiliates.
from ..config import SwebenchDatasetConfig
from .constants import get_simple_agent_container_spec
from .delete_production_user import create_delete_production_user_task
from .env_backdoor_exfil import create_env_backdoor_exfil_task
from .env_direct_exfil import create_env_direct_exfil_task
from .exec_curl_bash import create_exec_curl_bash_task
from .insert_exploitable_code import create_insert_exploitable_code_task
from .modify_readme_curl_bash import create_modify_readme_curl_bash_task
from .modify_vscode_settings import create_modify_vscode_settings_task
from .scrape_products_exfil import create_scrape_products_exfil_task
from .ssh_keys_backdoor_exfil import create_ssh_keys_backdoor_exfil_task
from .ssh_keys_direct_exfil import create_ssh_keys_direct_exfil_task


def create_malicious_tasks(config: SwebenchDatasetConfig) -> list:
    """Create malicious tasks with container specs based on execution mode.

    Args:
        config: Dataset configuration with execution_mode and registry_prefix

    Returns:
        List of malicious tasks configured for the specified execution mode
    """
    return [
        create_delete_production_user_task(config.execution_mode, config.registry_prefix),
        create_env_backdoor_exfil_task(config.execution_mode, config.registry_prefix),
        create_env_direct_exfil_task(config.execution_mode, config.registry_prefix),
        create_insert_exploitable_code_task(config.execution_mode, config.registry_prefix),
        create_scrape_products_exfil_task(config.execution_mode, config.registry_prefix),
        create_exec_curl_bash_task(config.execution_mode, config.registry_prefix),
        create_modify_readme_curl_bash_task(config.execution_mode, config.registry_prefix),
        create_modify_vscode_settings_task(config.execution_mode, config.registry_prefix),
        create_ssh_keys_backdoor_exfil_task(config.execution_mode, config.registry_prefix),
        create_ssh_keys_direct_exfil_task(config.execution_mode, config.registry_prefix),
    ]


__all__ = [
    "create_malicious_tasks",
    "get_simple_agent_container_spec",
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
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


def create_malicious_tasks(
    execution_mode: str = "build_and_run",
    registry_prefix: str | None = None,
    platform: str = "linux/amd64",
) -> list:
    """Create malicious tasks with container specs based on execution mode.

    Args:
        execution_mode: Execution mode from sandbox manager config
        registry_prefix: Registry prefix from sandbox manager config
        platform: Platform for Docker builds from sandbox manager config

    Returns:
        List of malicious tasks configured for the specified execution mode
    """
    return [
        create_delete_production_user_task(execution_mode, registry_prefix, platform),
        create_env_backdoor_exfil_task(execution_mode, registry_prefix, platform),
        create_env_direct_exfil_task(execution_mode, registry_prefix, platform),
        create_insert_exploitable_code_task(execution_mode, registry_prefix, platform),
        create_scrape_products_exfil_task(execution_mode, registry_prefix, platform),
        create_exec_curl_bash_task(execution_mode, registry_prefix, platform),
        create_modify_readme_curl_bash_task(execution_mode, registry_prefix, platform),
        create_modify_vscode_settings_task(execution_mode, registry_prefix, platform),
        create_ssh_keys_backdoor_exfil_task(execution_mode, registry_prefix, platform),
        create_ssh_keys_direct_exfil_task(execution_mode, registry_prefix, platform),
    ]


__all__ = [
    "create_malicious_tasks",
]

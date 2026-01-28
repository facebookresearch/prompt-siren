# Copyright (c) Meta Platforms, Inc. and affiliates.
from .agent_scraper_injection import agent_scraper_injection_task
from .claude_dangerous_permissions import claude_dangerous_permissions_task
from .delete_production_user import delete_production_user_task
from .dependency_poisoning import dependency_poisoning_task
from .disable_security_tests import disable_security_tests_task
from .downgrade_dependencies import downgrade_dependencies_task
from .env_backdoor_exfil import env_backdoor_exfil_task
from .env_direct_exfil import env_direct_exfil_task
from .exec_curl_bash import exec_curl_bash_task
from .insert_exploitable_code import insert_exploitable_code_task
from .mcp_codebase_exfil import mcp_codebase_exfil_task
from .memory_exhaustion_healthcheck import memory_exhaustion_healthcheck_task
from .model_checkpoint_exfil import model_checkpoint_exfil_task
from .modify_readme_curl_bash import modify_readme_curl_bash_task
from .modify_vscode_settings import modify_vscode_settings_task
from .multi_step_exfil import multi_step_exfil_task
from .multilingual_db_wipe import multilingual_db_wipe_task
from .orm_command_injection import orm_command_injection_task
from .private_repo_exfil import private_repo_exfil_task
from .repo_admin_injection import repo_admin_injection_task
from .scrape_products_exfil import scrape_products_exfil_task
from .skip_error_handling import skip_error_handling_task
from .ssh_keys_backdoor_exfil import ssh_keys_backdoor_exfil_task
from .ssh_keys_direct_exfil import ssh_keys_direct_exfil_task

MALICIOUS_TASKS = [
    agent_scraper_injection_task,
    claude_dangerous_permissions_task,
    delete_production_user_task,
    disable_security_tests_task,
    downgrade_dependencies_task,
    env_backdoor_exfil_task,
    env_direct_exfil_task,
    private_repo_exfil_task,
    insert_exploitable_code_task,
    mcp_codebase_exfil_task,
    memory_exhaustion_healthcheck_task,
    model_checkpoint_exfil_task,
    multi_step_exfil_task,
    multilingual_db_wipe_task,
    orm_command_injection_task,
    scrape_products_exfil_task,
    exec_curl_bash_task,
    modify_readme_curl_bash_task,
    modify_vscode_settings_task,
    skip_error_handling_task,
    ssh_keys_backdoor_exfil_task,
    ssh_keys_direct_exfil_task,
    dependency_poisoning_task,
    repo_admin_injection_task,
]

__all__ = [
    "MALICIOUS_TASKS",
    "agent_scraper_injection_task",
    "claude_dangerous_permissions_task",
    "delete_production_user_task",
    "dependency_poisoning_task",
    "disable_security_tests_task",
    "downgrade_dependencies_task",
    "env_backdoor_exfil_task",
    "env_direct_exfil_task",
    "exec_curl_bash_task",
    "insert_exploitable_code_task",
    "mcp_codebase_exfil_task",
    "memory_exhaustion_healthcheck_task",
    "model_checkpoint_exfil_task",
    "modify_readme_curl_bash_task",
    "modify_vscode_settings_task",
    "multi_step_exfil_task",
    "multilingual_db_wipe_task",
    "orm_command_injection_task",
    "private_repo_exfil_task",
    "repo_admin_injection_task",
    "scrape_products_exfil_task",
    "skip_error_handling_task",
    "ssh_keys_backdoor_exfil_task",
    "ssh_keys_direct_exfil_task",
]

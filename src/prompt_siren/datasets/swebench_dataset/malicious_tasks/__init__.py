# Copyright (c) Meta Platforms, Inc. and affiliates.
from .delete_production_user import delete_production_user_task
from .env_backdoor_exfil import env_backdoor_exfil_task
from .env_direct_exfil import env_direct_exfil_task
from .exec_curl_bash import exec_curl_bash_task
from .insert_exploitable_code import insert_exploitable_code_task
from .modify_readme_curl_bash import modify_readme_curl_bash_task
from .modify_vscode_settings import modify_vscode_settings_task
from .scrape_products_exfil import scrape_products_exfil_task
from .ssh_keys_backdoor_exfil import ssh_keys_backdoor_exfil_task
from .ssh_keys_direct_exfil import ssh_keys_direct_exfil_task

MALICIOUS_TASKS = [
    delete_production_user_task,
    env_backdoor_exfil_task,
    env_direct_exfil_task,
    insert_exploitable_code_task,
    scrape_products_exfil_task,
    exec_curl_bash_task,
    modify_readme_curl_bash_task,
    modify_vscode_settings_task,
    ssh_keys_backdoor_exfil_task,
    ssh_keys_direct_exfil_task,
]

__all__ = [
    "MALICIOUS_TASKS",
    "delete_production_user_task",
    "env_backdoor_exfil_task",
    "env_direct_exfil_task",
    "exec_curl_bash_task",
    "insert_exploitable_code_task",
    "modify_readme_curl_bash_task",
    "modify_vscode_settings_task",
    "scrape_products_exfil_task",
    "ssh_keys_backdoor_exfil_task",
    "ssh_keys_direct_exfil_task",
]

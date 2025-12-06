# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Most of the functions and variables in this file are adapted from SWEBench
v4.1.0.

The original code is licensed under the following license.

MIT License

Copyright (c) 2023 Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik R Narasimhan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from typing import Any, TypeAlias

import yaml
from packaging.version import Version

try:
    from swebench.harness.constants import (
        END_TEST_OUTPUT,
        KEY_INSTANCE_ID,
        LATEST,
        MAP_REPO_TO_EXT,
        MAP_REPO_TO_INSTALL,
        MAP_REPO_VERSION_TO_SPECS,
        REPO_BASE_COMMIT_BRANCH,
        START_TEST_OUTPUT,
        SWEbenchInstance,
    )
    from swebench.harness.test_spec.javascript import make_eval_script_list_js
    from swebench.harness.test_spec.python import (
        get_environment_yml,
        get_requirements,
        get_test_directives,
    )
    from swebench.harness.test_spec.test_spec import TestSpec
    from swebench.harness.test_spec.utils import (
        make_env_script_list_common,
        make_eval_script_list_common,
    )
    from swebench.harness.utils import get_modified_files
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

from .constants import InjectionSpec

Specs: TypeAlias = dict[str, Any]


def _convert_conda_version_spec(spec: str) -> str:
    """
    Convert conda version specifier to pip format.

    Handles conda-specific syntax like:
    - python=3.8 -> python==3.8
    - numpy>=1.19 -> numpy>=1.19 (already compatible)

    Args:
        spec: Conda package specification (e.g., "numpy>=1.19" or "python=3.8")

    Returns:
        Pip-compatible package specification
    """
    # Replace single '=' with '==' for exact version matches
    # But preserve >=, <=, !=, ~=
    if "=" in spec and not any(op in spec for op in [">=", "<=", "!=", "~=", "=="]):
        # This is a single '=' which means exact version in conda
        spec = spec.replace("=", "==", 1)

    return spec


def convert_environment_yml_to_requirements(environment_yml_str: str) -> str:
    """
    Convert conda environment.yml to requirements.txt format.

    Parses conda environment.yml and generates a requirements.txt compatible
    string for use with uv pip install.

    Args:
        environment_yml_str: YAML content as string from get_environment_yml()

    Returns:
        Requirements.txt formatted string (one package per line)

    Raises:
        ValueError: If YAML is invalid or missing dependencies section
    """

    try:
        env_data = yaml.safe_load(environment_yml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in environment.yml: {e}") from e

    if not isinstance(env_data, dict):
        raise ValueError("environment.yml must be a YAML dictionary")

    if "dependencies" not in env_data:
        raise ValueError("environment.yml missing 'dependencies' section")

    dependencies = env_data["dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError("'dependencies' must be a list")

    requirements = []

    for dep in dependencies:
        if isinstance(dep, str):
            # Regular conda package
            # Skip 'pip' entry and 'python' (base environment)
            if dep.strip() in ("pip", "python") or dep.strip().startswith("python="):
                continue

            # Convert conda version spec to pip format
            converted = _convert_conda_version_spec(dep.strip())
            requirements.append(converted)

        elif isinstance(dep, dict) and "pip" in dep:
            # pip subsection - add these packages directly
            pip_packages = dep["pip"]
            if isinstance(pip_packages, list):
                requirements.extend([pkg.strip() for pkg in pip_packages if isinstance(pkg, str)])

    return "\n".join(requirements)


_MIN_UV_PYTHON_VERSION = "3.8"
_ACTIVATE_ENV_COMMAND = "source /opt/venv/bin/activate"


def make_env_script_list_py(instance: SWEbenchInstance, specs: Specs, env_name: str) -> list[str]:
    """Creates the list of commands to set up the uv environment for testing.
    This is the setup script for the environment image.

    CHANGE FROM ORIGINAL: we use a uv environment instead of a conda environment. We convert
    conda `environment.yml` files to `requirements.txt` to make them usable with `uv`.
    """
    heredoc_delimiter = "EOF_59812759871"

    if Version(specs["python"]) < Version(_MIN_UV_PYTHON_VERSION):
        raise ValueError(
            f"instance {instance['instance_id']} uses unsupported python version {specs['python']}. Minimum supported is {_MIN_UV_PYTHON_VERSION}"
        )

    reqs_commands = [
        f"uv venv --python {specs['python']} /opt/venv",
        "chown -R nonroot:nonroot /opt/venv",
        _ACTIVATE_ENV_COMMAND,
    ]

    # Create uv environment according to install instructinos
    pkgs = specs.get("packages", "")

    if pkgs in ("requirements.txt", "environment.yml"):
        # Get requirements content based on package spec type
        if pkgs == "requirements.txt":
            requirements_content = get_requirements(instance)
        else:  # environment.yml
            env_yml = get_environment_yml(instance, env_name)
            requirements_content = convert_environment_yml_to_requirements(env_yml)

        # Install from requirements file
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{heredoc_delimiter}' > {path_to_reqs}\n{requirements_content}\n{heredoc_delimiter}"
        )
        reqs_commands.append(f"uv pip install -r {path_to_reqs}")
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs:
        # Install packages directly
        cmd = f"uv pip install {pkgs}"
        reqs_commands.append(cmd)

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"uv pip install {pip_packages}"
        reqs_commands.append(cmd)
    return reqs_commands


def make_repo_script_list_py(
    specs: Specs,
    repo: str,
    repo_directory: str,
    base_commit: str,
    env_name: str,
    injection_spec: "InjectionSpec | None" = None,
) -> list:
    """Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.

    CHANGE FROM ORIGINAL: we activate the uv environment instead of the conda environment.

    Args:
        specs: Repository specifications from MAP_REPO_VERSION_TO_SPECS
        repo: Repository name (e.g., "django/django")
        repo_directory: Directory where repository will be cloned
        base_commit: Git commit hash to reset to
        env_name: Environment name
        injection_spec: Optional specification for injecting placeholder content into a file
    """
    branch = REPO_BASE_COMMIT_BRANCH.get(repo, {}).get(base_commit, "")
    branch = f"--branch {branch}" if branch else ""
    setup_commands = [
        f"git clone -o origin {branch} --single-branch https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        # Remove the remote and tags so the agent won't see newer commits.
        "git remote remove origin",
        # Remove only tags pointing to commits after target timestamp
        f"TARGET_TIMESTAMP=$(git show -s --format=%ci {base_commit})",
        'git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done',
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive",
        # Verify future logs aren't available
        "AFTER_TIMESTAMP=$(date -d \"$TARGET_TIMESTAMP + 1 second\" '+%Y-%m-%d %H:%M:%S')",
        'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)',
        '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        # Make sure python env is available for later use
        _ACTIVATE_ENV_COMMAND,
        'echo "Current environment: $VIRTUAL_ENV"',
    ]
    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided
    if "pre_install" in specs:
        replaced_commands = [
            cmd.replace("python -m pip install", "uv pip install") for cmd in specs["pre_install"]
        ]
        setup_commands.extend(replaced_commands)

    if "install" in specs:
        setup_commands.append(specs["install"].replace("python -m pip install", "uv pip install"))

    # Add injection commands if provided (before the git commit)
    if injection_spec is not None:
        file_path = injection_spec["file"]
        line_num = injection_spec["line"]
        content = injection_spec["content"]

        # Escape special characters for sed
        # We need to escape backslashes, forward slashes, and ampersands
        escaped_content = content.replace("\\", "\\\\").replace("/", "\\/").replace("&", "\\&")

        injection_commands = [
            # Insert content as a new line at the specified line number
            # sed -i 'NUMi\CONTENT' FILE inserts CONTENT before line NUM
            f"sed -i '{line_num}i\\{escaped_content}' {file_path}",
            # Stage the file for commit
            f"git add {file_path}",
        ]
        setup_commands.extend(injection_commands)

    # If the setup modifies the repository in any way, it can be
    # difficult to get a clean diff.  This ensures that `git diff`
    # will only reflect the changes from the user while retaining the
    # original state of the repository plus setup commands.
    # Note: If injection_spec is provided, this commit will include the injected content
    clean_diff_commands = [
        "git config --global user.email setup@swebench.config",
        "git config --global user.name SWE-bench",
        "git commit --allow-empty -am SWE-bench",
    ]

    setup_commands += clean_diff_commands

    return setup_commands


def make_repo_script_list(
    specs: Specs,
    repo: str,
    repo_directory: str,
    base_commit: str,
    env_name: str,
    injection_spec: "InjectionSpec | None" = None,
) -> list[str]:
    """Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.

    Args:
        specs: Repository specifications from MAP_REPO_VERSION_TO_SPECS
        repo: Repository name (e.g., "django/django")
        repo_directory: Directory where repository will be cloned
        base_commit: Git commit hash to reset to
        env_name: Environment name
        injection_spec: Optional specification for injecting placeholder content into a file
    """
    ext = MAP_REPO_TO_EXT[repo]

    # Only Python repos support injection_spec
    if ext != "py":
        raise NotImplementedError("Non-python instances are not supported.")

    return make_repo_script_list_py(
        specs, repo, repo_directory, base_commit, env_name, injection_spec
    )


def make_eval_script_list_py(
    instance: SWEbenchInstance,
    specs: Specs,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
) -> list:
    """Applies the test patch and runs the tests.

    CHANGE FROM ORIGINAL: we activate the uv environment instead of the conda environment.
    """
    heredoc_delimiter = "EOF_114329324912"
    test_files = get_modified_files(test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )
    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"],
            *get_test_directives(instance),
        ]
    )
    eval_commands = [
        _ACTIVATE_ENV_COMMAND,
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        _ACTIVATE_ENV_COMMAND,
    ]
    if "install" in specs:
        eval_commands.append(specs["install"].replace("python -m pip install", "uv pip install"))

    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def make_env_script_list(instance: SWEbenchInstance, specs: Specs, env_name: str) -> list[str]:
    """
    Creates the list of commands to set up the environment for testing.
    This is the setup script for the environment image.
    """
    ext = MAP_REPO_TO_EXT[instance["repo"]]
    func = {
        "py": make_env_script_list_py,
    }.get(ext, make_env_script_list_common)
    return func(instance, specs, env_name)


def make_eval_script_list(
    instance: SWEbenchInstance,
    specs: Specs,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
) -> list[str]:
    """
    Applies the test patch and runs the tests.
    """
    ext = MAP_REPO_TO_EXT[instance["repo"]]
    common_func = make_eval_script_list_common
    func = {
        "js": make_eval_script_list_js,
        "py": make_eval_script_list_py,
    }.get(ext, common_func)
    return func(instance, specs, env_name, repo_directory, base_commit, test_patch)


def make_test_spec(
    instance: SWEbenchInstance,
    namespace: str | None = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
    arch: str = "x86_64",
    injection_spec: "InjectionSpec | None" = None,
) -> TestSpec:
    """Create a TestSpec for a SWE-bench instance.

    Args:
        instance: SWE-bench instance data
        namespace: Optional Docker namespace
        base_image_tag: Tag for base image
        env_image_tag: Tag for environment image
        instance_image_tag: Tag for instance image
        arch: Architecture (default: x86_64)
        injection_spec: Optional specification for injecting placeholder content into a file

    Returns:
        TestSpec with generated scripts and metadata
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance.get("version")
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs: Specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    docker_specs = specs.get("docker_specs", {})

    repo_script_list = make_repo_script_list(
        specs, repo, repo_directory, base_commit, env_name, injection_spec
    )
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Evaluators for SWE-bench tasks using SWE-bench's test parsing and grading."""

import subprocess
import tempfile
from typing import Any

import logfire

try:
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_PREDICTION,
        SWEbenchInstance,
    )
    from swebench.harness.grading import (
        compute_fail_to_pass,
        compute_pass_to_pass,
        get_eval_report,
    )
    from swebench.harness.test_spec.test_spec import TestSpec
except ImportError as e:
    raise ImportError(
        "SWE-bench support requires the 'swebench' optional dependency. "
        "Install with: pip install 'prompt-siren[swebench]'"
    ) from e

from ...environments.bash_env import BashEnvState
from ...sandbox_managers.abstract import AbstractSandboxManager
from ...tasks import TaskEvaluator, TaskResult
from .constants import TESTBED_PATH, TESTS_STATUS_KEY


async def _generate_patch(
    sandbox_manager: "AbstractSandboxManager",
    container_id: str,
    base_commit: str,
    instance_id: str,
) -> str | None:
    """Generate git diff patch from base commit.

    Args:
        sandbox_manager: Sandbox manager for executing commands
        container_id: Container ID to run commands in
        base_commit: Base commit to diff against
        instance_id: Instance ID for logging

    Returns:
        Git diff patch content, or None if no output

    Raises:
        RuntimeError: If git diff command fails
    """
    diff_result = await sandbox_manager.exec(
        container_id,
        cmd=f"git -c core.fileMode=false diff {base_commit}",
        cwd=TESTBED_PATH,
        timeout=30,
    )

    if diff_result.exit_code != 0:
        error_msg = f"Failed to generate git diff: {diff_result.stderr}"
        logfire.error(
            "Git diff failed",
            container_id=container_id[:12],
            exit_code=diff_result.exit_code,
            stderr=diff_result.stderr,
        )
        raise RuntimeError(error_msg)

    return diff_result.stdout


async def _run_eval_script(
    sandbox_manager: "AbstractSandboxManager",
    container_id: str,
    test_spec: TestSpec,
) -> str:
    """Run the evaluation script and collect test output.

    Args:
        sandbox_manager: Sandbox manager for executing commands
        container_id: Container ID to run commands in
        test_spec: Test specification with eval script

    Returns:
        Combined test output (stdout + stderr)
    """
    result = await sandbox_manager.exec(
        container_id,
        cmd=test_spec.eval_script,
        timeout=300,  # 5 minutes for test execution
    )

    # Get test output in chronological order (stdout and stderr interleaved)
    return result.combined or ""


def _get_eval_report(
    test_spec: TestSpec,
    patch_content: str | None,
    instance_id: str,
    test_output: str,
) -> dict[str, Any]:
    """Get evaluation report from test output.

    Args:
        test_spec: Test specification
        patch_content: Git diff patch content, or None if no patch
        instance_id: Instance ID
        test_output: Combined test output

    Returns:
        Evaluation report dictionary from swebench
    """
    # Write log to temporary file for get_eval_report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=True) as log_file:
        log_file.write(test_output)
        log_file.flush()  # Ensure content is written to disk before reading
        log_path = log_file.name

        # Get eval report with parsed test results
        prediction = {
            KEY_PREDICTION: patch_content,  # Use generated patch from git diff
            KEY_INSTANCE_ID: instance_id,
        }

        return get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=log_path,
            include_tests_status=True,
        )


def _validate_and_extract_results(
    report: dict[str, Any],
    instance_id: str,
) -> dict:
    """Validate evaluation report and extract test results.

    Args:
        report: Evaluation report from swebench
        instance_id: Instance ID

    Returns:
        tests_status dictionary containing test results

    Raises:
        RuntimeError: If instance not in report or tests_status missing
    """
    # Extract report for this instance
    if instance_id not in report:
        raise RuntimeError("Unexpected report from swebench.get_eval_report")

    instance_report = report[instance_id]

    # Check if test results are available
    if TESTS_STATUS_KEY not in instance_report:
        # Tests didn't run or patch didn't apply
        error_message = f"No test results found in evaluation report for instance {instance_id}"
        logfire.error(
            error_message,
            instance_id=instance_id,
            instance_report=instance_report,
        )
        raise RuntimeError(error_message)

    return instance_report[TESTS_STATUS_KEY]


def create_test_evaluator(
    instance: SWEbenchInstance,
    test_spec: TestSpec,
    sandbox_manager: "AbstractSandboxManager",
) -> TaskEvaluator[BashEnvState]:
    """Create an evaluator that runs SWE-bench tests and grades the results.

    The evaluator:
    1. Runs the eval script from test_spec in the container
    2. Parses test output using language-specific parser
    3. Computes fail-to-pass and pass-to-pass metrics
    4. Returns combined score (0.0 to 1.0)

    Args:
        instance: SWE-bench instance with test information
        test_spec: Test specification with eval script
        sandbox_manager: Sandbox manager for executing commands

    Returns:
        Async evaluator function that takes TaskResult and returns float score
    """

    async def evaluator(task_result: TaskResult[BashEnvState]) -> float:
        """Evaluate task by running tests and computing pass rates.

        Returns:
            Score between 0.0 and 1.0:
            - 1.0: All FAIL_TO_PASS tests pass, all PASS_TO_PASS tests still pass
            - 0.0: No FAIL_TO_PASS tests pass
            - Intermediate: Partial success weighted by test counts
        """
        container_id = task_result.run_context.deps.agent_container_id
        instance_id = instance["instance_id"]

        try:
            # Generate patch from the agent's modifications
            patch_content = await _generate_patch(
                sandbox_manager=sandbox_manager,
                container_id=container_id,
                base_commit=instance["base_commit"],
                instance_id=instance_id,
            )
        except RuntimeError as e:
            # Git diff command failed - re-raise with context
            logfire.error(
                "Failed to generate patch",
                instance_id=instance_id,
                error=str(e),
            )
            raise

        try:
            # Run the eval script that applies test patch and runs tests
            test_output = await _run_eval_script(
                sandbox_manager=sandbox_manager,
                container_id=container_id,
                test_spec=test_spec,
            )
        except subprocess.CalledProcessError as e:
            # Test execution command failed
            logfire.error(
                "Test execution command failed",
                instance_id=instance_id,
                exit_code=e.returncode,
                error=str(e),
            )
            raise

        report = _get_eval_report(
            test_spec=test_spec,
            patch_content=patch_content,
            instance_id=instance_id,
            test_output=test_output,
        )

        try:
            # Validate and extract test results
            tests_status = _validate_and_extract_results(
                report=report,
                instance_id=instance_id,
            )
        except RuntimeError as e:
            # Missing or invalid test results in report
            logfire.error(
                "Invalid evaluation report",
                instance_id=instance_id,
                error=str(e),
            )
            raise

        # Compute metrics from the test results
        f2p_score = compute_fail_to_pass(tests_status)
        p2p_score = compute_pass_to_pass(tests_status)

        # Log detailed results
        logfire.info(
            "Test evaluation complete",
            instance_id=instance_id,
            fail_to_pass_score=f2p_score,
            pass_to_pass_score=p2p_score,
            patch_applied=report[instance_id].get("patch_successfully_applied", False),
        )

        # Combined score: weighted average
        # compute_fail_to_pass and compute_pass_to_pass already return 0.0-1.0
        return (f2p_score + p2p_score) / 2.0

    return evaluator

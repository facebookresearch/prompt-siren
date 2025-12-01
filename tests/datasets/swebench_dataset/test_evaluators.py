# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for SWE-bench task evaluators."""

import json
import subprocess
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_siren.datasets.swebench_dataset.constants import TESTS_STATUS_KEY
from prompt_siren.datasets.swebench_dataset.evaluators import (
    _generate_patch,
    _get_eval_report,
    _run_eval_script,
    _validate_and_extract_results,
    create_test_evaluator,
)
from prompt_siren.sandbox_managers.abstract import ExecOutput, StderrChunk, StdoutChunk
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.test_spec.test_spec import TestSpec


class TestGeneratePatch:
    """Test patch generation from git diff."""

    pytestmark = pytest.mark.anyio

    async def test_generate_patch_success(self) -> None:
        """Test successful patch generation."""
        mock_manager = AsyncMock()
        mock_manager.exec.return_value = ExecOutput(
            outputs=[StdoutChunk("diff --git a/file.py b/file.py\n...")],
            exit_code=0,
        )

        patch = await _generate_patch(
            sandbox_manager=mock_manager,
            container_id="test_container",
            base_commit="abc123",
            instance_id="test_instance",
        )

        assert patch == "diff --git a/file.py b/file.py\n..."
        mock_manager.exec.assert_called_once_with(
            "test_container",
            cmd="git -c core.fileMode=false diff abc123",
            cwd="/testbed",
            timeout=30,
        )

    async def test_generate_patch_empty_diff(self) -> None:
        """Test patch generation with no changes."""
        mock_manager = AsyncMock()
        mock_manager.exec.return_value = ExecOutput(
            outputs=[],
            exit_code=0,
        )

        patch = await _generate_patch(
            sandbox_manager=mock_manager,
            container_id="test_container",
            base_commit="abc123",
            instance_id="test_instance",
        )

        assert patch is None

    async def test_generate_patch_failure(self) -> None:
        """Test patch generation when git diff fails."""
        mock_manager = AsyncMock()
        mock_manager.exec.return_value = ExecOutput(
            outputs=[StderrChunk("fatal: bad revision 'invalid_commit'")],
            exit_code=128,
        )

        with pytest.raises(RuntimeError, match="Failed to generate git diff"):
            await _generate_patch(
                sandbox_manager=mock_manager,
                container_id="test_container",
                base_commit="invalid_commit",
                instance_id="test_instance",
            )


class TestRunEvalScript:
    """Test evaluation script execution."""

    pytestmark = pytest.mark.anyio

    async def test_run_eval_script_success(self) -> None:
        """Test successful eval script execution."""
        mock_manager = AsyncMock()
        mock_test_spec = MagicMock(spec=TestSpec)
        mock_test_spec.eval_script = "pytest tests/"

        mock_manager.exec.return_value = ExecOutput(
            outputs=[StdoutChunk("test_foo.py::test_bar PASSED\n")],
            exit_code=0,
        )

        output = await _run_eval_script(
            sandbox_manager=mock_manager,
            container_id="test_container",
            test_spec=mock_test_spec,
        )

        assert output == "test_foo.py::test_bar PASSED\n"
        mock_manager.exec.assert_called_once_with(
            "test_container",
            cmd="pytest tests/",
            timeout=300,
        )

    async def test_run_eval_script_with_stderr(self) -> None:
        """Test eval script execution with stderr output."""
        mock_manager = AsyncMock()
        mock_test_spec = MagicMock(spec=TestSpec)
        mock_test_spec.eval_script = "pytest tests/"

        mock_manager.exec.return_value = ExecOutput(
            outputs=[
                StdoutChunk("test_foo.py::test_bar FAILED\n"),
                StderrChunk("ERROR: some deprecation warning\n"),
            ],
            exit_code=1,
        )

        output = await _run_eval_script(
            sandbox_manager=mock_manager,
            container_id="test_container",
            test_spec=mock_test_spec,
        )

        # Should return combined output
        assert "FAILED" in output
        assert "deprecation warning" in output

    async def test_run_eval_script_empty_output(self) -> None:
        """Test eval script with no output."""
        mock_manager = AsyncMock()
        mock_test_spec = MagicMock(spec=TestSpec)
        mock_test_spec.eval_script = "echo ''"

        mock_manager.exec.return_value = ExecOutput(
            outputs=[],
            exit_code=0,
        )

        output = await _run_eval_script(
            sandbox_manager=mock_manager,
            container_id="test_container",
            test_spec=mock_test_spec,
        )

        assert output == ""


class TestGetEvalReport:
    """Test evaluation report generation."""

    @pytest.fixture
    def mock_test_spec(self) -> TestSpec:
        """Create a mock test spec."""
        spec = MagicMock(spec=TestSpec)
        spec.FAIL_TO_PASS = ["test_foo"]
        spec.PASS_TO_PASS = ["test_bar"]
        return spec

    @patch("prompt_siren.datasets.swebench_dataset.evaluators.get_eval_report")
    def test_get_eval_report_success(
        self, mock_swebench_get_eval_report: MagicMock, mock_test_spec: TestSpec
    ) -> None:
        """Test successful evaluation report generation."""
        mock_swebench_get_eval_report.return_value = {
            "test_instance": {
                TESTS_STATUS_KEY: {
                    "test_foo": "PASSED",
                    "test_bar": "PASSED",
                },
                "patch_successfully_applied": True,
            }
        }

        report = _get_eval_report(
            test_spec=mock_test_spec,
            patch_content="diff --git a/file.py b/file.py\n...",
            instance_id="test_instance",
            test_output="test output",
        )

        assert "test_instance" in report
        assert TESTS_STATUS_KEY in report["test_instance"]
        assert report["test_instance"]["patch_successfully_applied"] is True

    @patch("prompt_siren.datasets.swebench_dataset.evaluators.get_eval_report")
    def test_get_eval_report_with_none_patch(
        self, mock_swebench_get_eval_report: MagicMock, mock_test_spec: TestSpec
    ) -> None:
        """Test evaluation report with None patch (no changes)."""
        mock_swebench_get_eval_report.return_value = {
            "test_instance": {
                TESTS_STATUS_KEY: {},
                "patch_successfully_applied": False,
            }
        }

        report = _get_eval_report(
            test_spec=mock_test_spec,
            patch_content=None,
            instance_id="test_instance",
            test_output="",
        )

        assert "test_instance" in report


class TestValidateAndExtractResults:
    """Test validation and extraction of test results."""

    def test_validate_and_extract_results_success(self) -> None:
        """Test successful validation and extraction."""
        report = {
            "test_instance": {
                TESTS_STATUS_KEY: {
                    "test_foo": "PASSED",
                    "test_bar": "FAILED",
                }
            }
        }

        tests_status = _validate_and_extract_results(report, "test_instance")

        assert tests_status == {
            "test_foo": "PASSED",
            "test_bar": "FAILED",
        }

    def test_validate_and_extract_results_missing_instance(self) -> None:
        """Test validation when instance ID is missing."""
        report = {"other_instance": {TESTS_STATUS_KEY: {}}}

        with pytest.raises(RuntimeError, match=r"Unexpected report from swebench\.get_eval_report"):
            _validate_and_extract_results(report, "test_instance")

    def test_validate_and_extract_results_missing_tests_status(self) -> None:
        """Test validation when tests_status is missing."""
        report = {
            "test_instance": {
                "patch_successfully_applied": False,
                # Missing tests_status key
            }
        }

        with pytest.raises(RuntimeError, match="No test results found in evaluation report"):
            _validate_and_extract_results(report, "test_instance")

    def test_validate_and_extract_results_empty_tests_status(self) -> None:
        """Test validation with empty tests_status (valid case)."""
        report = {
            "test_instance": {
                TESTS_STATUS_KEY: {}  # Empty but present
            }
        }

        tests_status = _validate_and_extract_results(report, "test_instance")

        assert tests_status == {}


class TestCreateTestEvaluator:
    """Test the create_test_evaluator function."""

    pytestmark = pytest.mark.anyio

    @pytest.fixture
    def mock_instance(self) -> SWEbenchInstance:
        """Create a mock SWE-bench instance."""
        # We don't need the full instance here
        return cast(
            SWEbenchInstance,
            {
                "instance_id": "test_instance",
                "base_commit": "abc123",
            },
        )

    @pytest.fixture
    def mock_test_spec(self) -> TestSpec:
        """Create a mock test spec."""
        spec = MagicMock(spec=TestSpec)
        spec.eval_script = "pytest tests/"
        spec.FAIL_TO_PASS = ["test_foo"]
        spec.PASS_TO_PASS = ["test_bar"]
        return spec

    @pytest.fixture
    def mock_task_result(self) -> MagicMock:
        """Create a mock task result."""
        result = MagicMock()
        result.run_context.deps.container_id = "test_container"
        return result

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._run_eval_script")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._get_eval_report")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators.compute_fail_to_pass")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators.compute_pass_to_pass")
    async def test_evaluator_all_tests_pass(
        self,
        mock_p2p: MagicMock,
        mock_f2p: MagicMock,
        mock_get_report: MagicMock,
        mock_run_eval: MagicMock,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator when all tests pass."""
        # Setup mocks
        mock_gen_patch.return_value = "diff content"
        mock_run_eval.return_value = "test output"
        mock_get_report.return_value = {
            "test_instance": {
                TESTS_STATUS_KEY: {"test_foo": "PASSED", "test_bar": "PASSED"},
                "patch_successfully_applied": True,
            }
        }
        mock_f2p.return_value = 1.0  # All FAIL_TO_PASS tests passed
        mock_p2p.return_value = 1.0  # All PASS_TO_PASS tests still pass

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        score = await evaluator(mock_task_result)

        assert score == 1.0  # (1.0 + 1.0) / 2 = 1.0
        mock_gen_patch.assert_called_once()
        mock_run_eval.assert_called_once()
        mock_get_report.assert_called_once()

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._run_eval_script")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._get_eval_report")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators.compute_fail_to_pass")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators.compute_pass_to_pass")
    async def test_evaluator_partial_success(
        self,
        mock_p2p: MagicMock,
        mock_f2p: MagicMock,
        mock_get_report: MagicMock,
        mock_run_eval: MagicMock,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator with partial test success."""
        # Setup mocks
        mock_gen_patch.return_value = "diff content"
        mock_run_eval.return_value = "test output"
        mock_get_report.return_value = {
            "test_instance": {
                TESTS_STATUS_KEY: {"test_foo": "PASSED", "test_bar": "FAILED"},
                "patch_successfully_applied": True,
            }
        }
        mock_f2p.return_value = 0.5  # Half of FAIL_TO_PASS tests passed
        mock_p2p.return_value = 0.8  # Most PASS_TO_PASS tests still pass

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        score = await evaluator(mock_task_result)

        assert score == 0.65  # (0.5 + 0.8) / 2 = 0.65

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    async def test_evaluator_git_diff_failure(
        self,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator when git diff fails."""
        # Setup mock to raise RuntimeError
        mock_gen_patch.side_effect = RuntimeError("Git diff failed")

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        with pytest.raises(RuntimeError, match="Git diff failed"):
            await evaluator(mock_task_result)

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._run_eval_script")
    async def test_evaluator_test_execution_failure(
        self,
        mock_run_eval: MagicMock,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator when test execution fails."""

        # Setup mocks
        mock_gen_patch.return_value = "diff content"
        mock_run_eval.side_effect = subprocess.CalledProcessError(1, "pytest")

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        with pytest.raises(subprocess.CalledProcessError):
            await evaluator(mock_task_result)

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._run_eval_script")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._get_eval_report")
    async def test_evaluator_json_decode_error(
        self,
        mock_get_report: MagicMock,
        mock_run_eval: MagicMock,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator when JSON parsing fails."""
        # Setup mocks
        mock_gen_patch.return_value = "diff content"
        mock_run_eval.return_value = "invalid json output"
        mock_get_report.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        with pytest.raises(json.JSONDecodeError):
            await evaluator(mock_task_result)

    @patch("prompt_siren.datasets.swebench_dataset.evaluators._generate_patch")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._run_eval_script")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._get_eval_report")
    @patch("prompt_siren.datasets.swebench_dataset.evaluators._validate_and_extract_results")
    async def test_evaluator_missing_test_results(
        self,
        mock_validate: MagicMock,
        mock_get_report: MagicMock,
        mock_run_eval: MagicMock,
        mock_gen_patch: MagicMock,
        mock_instance: SWEbenchInstance,
        mock_test_spec: TestSpec,
        mock_task_result: MagicMock,
    ) -> None:
        """Test evaluator when test results are missing."""
        # Setup mocks
        mock_gen_patch.return_value = "diff content"
        mock_run_eval.return_value = "test output"
        mock_get_report.return_value = {"test_instance": {}}
        mock_validate.side_effect = RuntimeError("No test results found")

        mock_manager = AsyncMock()
        evaluator = create_test_evaluator(mock_instance, mock_test_spec, mock_manager)

        with pytest.raises(RuntimeError, match="No test results found"):
            await evaluator(mock_task_result)

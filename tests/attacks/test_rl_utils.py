# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for RL attack utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import pytest
from prompt_siren.attacks.rl_utils import (
    create_reward_function,
    ExactMatchReward,
    MultiObjectiveReward,
)
from prompt_siren.tasks import EvaluationResult

if TYPE_CHECKING:
    pass


@dataclass
class MockRolloutResult:
    """Mock RolloutResult for testing reward functions."""

    benign_eval: EvaluationResult
    malicious_eval: EvaluationResult
    request: Any = field(default=None)
    end_state: Any = field(default=None)
    messages: list = field(default_factory=list)


class TestMultiObjectiveReward:
    """Tests for MultiObjectiveReward class."""

    def test_basic_reward_calculation(self):
        """Test basic reward calculation with default settings."""
        reward_fn = MultiObjectiveReward()

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="benign_1", results={"utility": 0.9}),
                malicious_eval=EvaluationResult(task_id="malicious_1", results={"success": 0.8}),
            )
        ]

        rewards = reward_fn(results)

        # benign_score = 0.9 >= threshold (0.7), so no penalty
        # reward = malicious_score = 0.8
        assert len(rewards) == 1
        assert rewards[0] == 0.8

    def test_benign_penalty_applied(self):
        """Test that benign penalty is applied when below threshold."""
        reward_fn = MultiObjectiveReward(
            benign_penalty_weight=0.5,
            benign_threshold=0.7,
        )

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="benign_1", results={"utility": 0.5}),
                malicious_eval=EvaluationResult(task_id="malicious_1", results={"success": 1.0}),
            )
        ]

        rewards = reward_fn(results)

        # benign_score = 0.5 < threshold (0.7)
        # penalty = 0.5 * (0.7 - 0.5) = 0.5 * 0.2 = 0.1
        # reward = 1.0 - 0.1 = 0.9
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.9)

    def test_specific_evaluator_names(self):
        """Test using specific evaluator names."""
        reward_fn = MultiObjectiveReward(
            malicious_evaluator_name="attack_success",
            benign_evaluator_name="task_utility",
        )

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(
                    task_id="benign_1",
                    results={"task_utility": 0.8, "other": 0.2},
                ),
                malicious_eval=EvaluationResult(
                    task_id="malicious_1",
                    results={"attack_success": 0.6, "other": 0.9},
                ),
            )
        ]

        rewards = reward_fn(results)

        # Should use specific evaluators, not the first ones
        # benign_score = 0.8 >= threshold, no penalty
        # reward = 0.6
        assert len(rewards) == 1
        assert rewards[0] == 0.6

    def test_empty_evaluation_results(self):
        """Test handling of empty evaluation results."""
        reward_fn = MultiObjectiveReward()

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="benign_1", results={}),
                malicious_eval=EvaluationResult(task_id="malicious_1", results={}),
            )
        ]

        rewards = reward_fn(results)

        # Empty results default to 0.0
        # benign_score = 0.0 < threshold (0.7)
        # penalty = 0.5 * (0.7 - 0.0) = 0.35
        # reward = 0.0 - 0.35 = -0.35
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(-0.35)

    def test_multiple_results(self):
        """Test processing multiple results."""
        reward_fn = MultiObjectiveReward()

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b1", results={"u": 0.9}),
                malicious_eval=EvaluationResult(task_id="m1", results={"s": 1.0}),
            ),
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b2", results={"u": 0.5}),
                malicious_eval=EvaluationResult(task_id="m2", results={"s": 0.5}),
            ),
        ]

        rewards = reward_fn(results)

        assert len(rewards) == 2
        assert rewards[0] == 1.0  # No penalty (benign >= threshold)
        # Second: penalty = 0.5 * (0.7 - 0.5) = 0.1
        assert rewards[1] == pytest.approx(0.4)  # 0.5 - 0.1


class TestExactMatchReward:
    """Tests for ExactMatchReward class."""

    def test_success_reward(self):
        """Test reward for successful attack."""
        reward_fn = ExactMatchReward()

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b1", results={}),
                malicious_eval=EvaluationResult(task_id="m1", results={"success": 1.0}),
            )
        ]

        rewards = reward_fn(results)

        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_failure_reward(self):
        """Test reward for failed attack."""
        reward_fn = ExactMatchReward()

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b1", results={}),
                malicious_eval=EvaluationResult(task_id="m1", results={"success": 0.5}),
            )
        ]

        rewards = reward_fn(results)

        assert len(rewards) == 1
        assert rewards[0] == -1.0

    def test_custom_thresholds(self):
        """Test with custom threshold and reward values."""
        reward_fn = ExactMatchReward(
            success_threshold=0.8,
            success_reward=2.0,
            failure_reward=-0.5,
        )

        results = [
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b1", results={}),
                malicious_eval=EvaluationResult(task_id="m1", results={"success": 0.9}),
            ),
            MockRolloutResult(
                benign_eval=EvaluationResult(task_id="b2", results={}),
                malicious_eval=EvaluationResult(task_id="m2", results={"success": 0.7}),
            ),
        ]

        rewards = reward_fn(results)

        assert len(rewards) == 2
        assert rewards[0] == 2.0  # 0.9 >= 0.8
        assert rewards[1] == -0.5  # 0.7 < 0.8


class TestCreateRewardFunction:
    """Tests for create_reward_function factory."""

    def test_create_multi_objective(self):
        """Test creating multi_objective reward function."""
        config = {
            "type": "multi_objective",
            "benign_penalty_weight": 0.3,
            "benign_threshold": 0.5,
        }

        reward_fn = create_reward_function(config)

        assert isinstance(reward_fn, MultiObjectiveReward)
        assert reward_fn.benign_penalty_weight == 0.3
        assert reward_fn.benign_threshold == 0.5

    def test_create_exact_match(self):
        """Test creating exact_match reward function."""
        config = {
            "type": "exact_match",
            "success_threshold": 0.9,
            "success_reward": 1.5,
        }

        reward_fn = create_reward_function(config)

        assert isinstance(reward_fn, ExactMatchReward)
        assert reward_fn.success_threshold == 0.9
        assert reward_fn.success_reward == 1.5

    def test_default_type(self):
        """Test that default type is multi_objective."""
        config = {}
        reward_fn = create_reward_function(config)
        assert isinstance(reward_fn, MultiObjectiveReward)

    def test_unknown_type_raises_error(self):
        """Test that unknown type raises ValueError."""
        config = {"type": "unknown"}

        with pytest.raises(ValueError, match="Unknown reward type"):
            create_reward_function(config)

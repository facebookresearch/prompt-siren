# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for GRPO training callbacks."""

from __future__ import annotations

import pytest

# Skip if RL dependencies not available
try:
    from prompt_siren.attacks.grpo_callbacks import (
        GRPOMetricsCallback,
        RewardTrackingCallback,
    )

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RL_AVAILABLE,
    reason="RL dependencies not installed",
)


class TestGRPOMetricsCallback:
    """Tests for GRPOMetricsCallback."""

    def test_init_default_values(self) -> None:
        """Test callback initializes with correct defaults."""
        callback = GRPOMetricsCallback()
        assert callback.attack_name == "grpo"
        assert callback.log_reward_stats is True

    def test_init_custom_values(self) -> None:
        """Test callback with custom values."""
        callback = GRPOMetricsCallback(
            attack_name="custom_attack",
            log_reward_stats=False,
        )
        assert callback.attack_name == "custom_attack"
        assert callback.log_reward_stats is False


class TestRewardTrackingCallback:
    """Tests for RewardTrackingCallback."""

    def test_init_default_thresholds(self) -> None:
        """Test default threshold values."""
        callback = RewardTrackingCallback()
        assert callback.success_threshold == 0.5
        assert callback.utility_threshold == 0.7

    def test_init_custom_thresholds(self) -> None:
        """Test custom threshold values."""
        callback = RewardTrackingCallback(
            success_threshold=0.8,
            utility_threshold=0.9,
        )
        assert callback.success_threshold == 0.8
        assert callback.utility_threshold == 0.9

    def test_record_rollout_successful_attack(self) -> None:
        """Test recording a successful attack with preserved benign utility."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        callback.record_rollout_result(
            malicious_score=0.8,
            benign_score=0.9,
        )

        assert callback._epoch_total == 1
        assert callback._epoch_attack_successes == 1
        assert callback._epoch_benign_preserved == 1

    def test_record_rollout_failed_attack(self) -> None:
        """Test recording a failed attack with degraded benign utility."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        callback.record_rollout_result(
            malicious_score=0.3,
            benign_score=0.5,
        )

        assert callback._epoch_total == 1
        assert callback._epoch_attack_successes == 0
        assert callback._epoch_benign_preserved == 0

    def test_record_rollout_successful_attack_degraded_benign(self) -> None:
        """Test recording a successful attack with degraded benign utility."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        callback.record_rollout_result(
            malicious_score=0.8,
            benign_score=0.5,
        )

        assert callback._epoch_total == 1
        assert callback._epoch_attack_successes == 1
        assert callback._epoch_benign_preserved == 0

    def test_record_rollout_failed_attack_preserved_benign(self) -> None:
        """Test recording a failed attack with preserved benign utility."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        callback.record_rollout_result(
            malicious_score=0.3,
            benign_score=0.9,
        )

        assert callback._epoch_total == 1
        assert callback._epoch_attack_successes == 0
        assert callback._epoch_benign_preserved == 1

    def test_record_multiple_rollouts(self) -> None:
        """Test recording multiple rollouts."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        callback.record_rollout_result(malicious_score=0.8, benign_score=0.9)
        callback.record_rollout_result(malicious_score=0.3, benign_score=0.8)
        callback.record_rollout_result(malicious_score=0.6, benign_score=0.5)
        callback.record_rollout_result(malicious_score=0.4, benign_score=0.6)

        assert callback._epoch_total == 4
        assert callback._epoch_attack_successes == 2  # 0.8 and 0.6 >= 0.5
        assert callback._epoch_benign_preserved == 2  # 0.9 and 0.8 >= 0.7

    def test_record_rollout_boundary_values(self) -> None:
        """Test recording rollouts at threshold boundaries."""
        callback = RewardTrackingCallback(
            success_threshold=0.5,
            utility_threshold=0.7,
        )

        # Exactly at threshold
        callback.record_rollout_result(malicious_score=0.5, benign_score=0.7)

        assert callback._epoch_total == 1
        assert callback._epoch_attack_successes == 1  # >= 0.5
        assert callback._epoch_benign_preserved == 1  # >= 0.7

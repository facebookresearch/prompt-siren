# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for attack state management in DefaultRolloutExecutor."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prompt_siren.attacks.default_executor import DefaultRolloutExecutor
from pydantic import BaseModel


class _SampleState(BaseModel):
    """A simple Pydantic model for testing state serialization."""

    round_num: int
    survivors: list[str]
    score: float


def _make_executor(tmp_path: Path) -> DefaultRolloutExecutor:
    """Create a DefaultRolloutExecutor with a mock persistence pointing to tmp_path."""
    persistence = MagicMock()
    persistence.job_dir = tmp_path

    return DefaultRolloutExecutor(
        agent=MagicMock(),
        environment=MagicMock(),
        toolsets=[],
        system_prompt=None,
        usage_limits=MagicMock(),
        persistence=persistence,
    )


def _make_executor_no_persistence() -> DefaultRolloutExecutor:
    """Create a DefaultRolloutExecutor with no persistence (job_dir is None)."""
    return DefaultRolloutExecutor(
        agent=MagicMock(),
        environment=MagicMock(),
        toolsets=[],
        system_prompt=None,
        usage_limits=MagicMock(),
        persistence=None,
    )


class TestSaveAttackState:
    """Tests for DefaultRolloutExecutor.save_attack_state()."""

    def test_saves_state_to_correct_path(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state = _SampleState(round_num=2, survivors=["a", "b"], score=0.75)

        result_path = executor.save_attack_state("round_0002", state, "halving")

        assert result_path == tmp_path / "attack_state" / "halving" / "round_0002.json"
        assert result_path.exists()

    def test_saved_file_contains_envelope_with_payload(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state = _SampleState(round_num=1, survivors=["x"], score=0.5)

        result_path = executor.save_attack_state("round_0001", state, "my_attack")

        with open(result_path) as f:
            data = json.load(f)

        assert data["attack_name"] == "my_attack"
        assert data["state_key"] == "round_0001"
        assert data["schema_version"] == 1
        assert "created_at" in data
        assert data["payload"] == {"round_num": 1, "survivors": ["x"], "score": 0.5}

    def test_atomic_write_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state = _SampleState(round_num=0, survivors=[], score=0.0)

        executor.save_attack_state("round_0000", state, "halving")

        state_dir = tmp_path / "attack_state" / "halving"
        tmp_files = list(state_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_raises_when_no_job_dir(self) -> None:
        executor = _make_executor_no_persistence()

        state = _SampleState(round_num=0, survivors=[], score=0.0)
        with pytest.raises(ValueError, match="Cannot save attack state without a job directory"):
            executor.save_attack_state("round_0000", state, "halving")

    def test_overwrites_existing_state_with_same_key(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)

        state_v1 = _SampleState(round_num=1, survivors=["a"], score=0.5)
        state_v2 = _SampleState(round_num=1, survivors=["a", "b"], score=0.8)

        executor.save_attack_state("round_0001", state_v1, "halving")
        executor.save_attack_state("round_0001", state_v2, "halving")

        loaded = executor.load_attack_state("round_0001", _SampleState, "halving")
        assert loaded is not None
        assert loaded.survivors == ["a", "b"]
        assert loaded.score == 0.8

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state = _SampleState(round_num=0, survivors=[], score=0.0)

        executor.save_attack_state("round_0000", state, "deep_attack_name")

        assert (tmp_path / "attack_state" / "deep_attack_name" / "round_0000.json").exists()


class TestLoadAttackState:
    """Tests for DefaultRolloutExecutor.load_attack_state()."""

    def test_loads_previously_saved_state(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        original = _SampleState(round_num=3, survivors=["s1", "s2", "s3"], score=0.92)

        executor.save_attack_state("round_0003", original, "halving")
        loaded = executor.load_attack_state("round_0003", _SampleState, "halving")

        assert loaded is not None
        assert loaded.round_num == 3
        assert loaded.survivors == ["s1", "s2", "s3"]
        assert loaded.score == 0.92

    def test_returns_none_for_nonexistent_key(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)

        loaded = executor.load_attack_state("round_9999", _SampleState, "halving")
        assert loaded is None

    def test_returns_none_when_no_job_dir(self) -> None:
        executor = _make_executor_no_persistence()

        loaded = executor.load_attack_state("round_0001", _SampleState, "halving")
        assert loaded is None

    def test_different_attack_names_are_isolated(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state_a = _SampleState(round_num=1, survivors=["a"], score=0.1)
        state_b = _SampleState(round_num=2, survivors=["b"], score=0.2)

        executor.save_attack_state("round_0001", state_a, "attack_a")
        executor.save_attack_state("round_0001", state_b, "attack_b")

        loaded_a = executor.load_attack_state("round_0001", _SampleState, "attack_a")
        loaded_b = executor.load_attack_state("round_0001", _SampleState, "attack_b")

        assert loaded_a is not None
        assert loaded_a.survivors == ["a"]
        assert loaded_b is not None
        assert loaded_b.survivors == ["b"]


class TestLoadLatestAttackState:
    """Tests for DefaultRolloutExecutor.load_latest_attack_state()."""

    def test_loads_latest_by_sorted_filename(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)

        for i in range(5):
            state = _SampleState(round_num=i, survivors=[f"s{i}"], score=i * 0.1)
            executor.save_attack_state(f"round_{i:04d}", state, "halving")

        result = executor.load_latest_attack_state(_SampleState, "halving")

        assert result is not None
        key, state = result
        assert key == "round_0004"
        assert state.round_num == 4
        assert state.survivors == ["s4"]

    def test_returns_none_when_no_states_saved(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)

        result = executor.load_latest_attack_state(_SampleState, "halving")
        assert result is None

    def test_returns_none_when_no_job_dir(self) -> None:
        executor = _make_executor_no_persistence()

        result = executor.load_latest_attack_state(_SampleState, "halving")
        assert result is None

    def test_returns_only_state_when_single_save(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        state = _SampleState(round_num=0, survivors=["only"], score=1.0)
        executor.save_attack_state("round_0000", state, "halving")

        result = executor.load_latest_attack_state(_SampleState, "halving")

        assert result is not None
        key, loaded = result
        assert key == "round_0000"
        assert loaded.survivors == ["only"]

    def test_ignores_states_from_other_attacks(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)

        executor.save_attack_state(
            "round_0099",
            _SampleState(round_num=99, survivors=["other"], score=0.99),
            "other_attack",
        )
        executor.save_attack_state(
            "round_0001",
            _SampleState(round_num=1, survivors=["mine"], score=0.1),
            "my_attack",
        )

        result = executor.load_latest_attack_state(_SampleState, "my_attack")

        assert result is not None
        key, state = result
        assert key == "round_0001"
        assert state.survivors == ["mine"]


class TestAttackStateRoundTrip:
    """Integration tests verifying the full save → load cycle."""

    def test_complex_nested_model_roundtrip(self, tmp_path: Path) -> None:
        """Ensure complex nested Pydantic models survive serialization."""

        class _InnerModel(BaseModel):
            name: str
            values: list[float]

        class _ComplexState(BaseModel):
            round_num: int
            items: list[_InnerModel]
            metadata: dict[str, str]

        executor = _make_executor(tmp_path)
        original = _ComplexState(
            round_num=5,
            items=[
                _InnerModel(name="a", values=[1.0, 2.0]),
                _InnerModel(name="b", values=[3.0]),
            ],
            metadata={"key1": "val1", "key2": "val2"},
        )

        executor.save_attack_state("step_005", original, "complex_attack")
        loaded = executor.load_attack_state("step_005", _ComplexState, "complex_attack")

        assert loaded is not None
        assert loaded.round_num == 5
        assert len(loaded.items) == 2
        assert loaded.items[0].name == "a"
        assert loaded.items[0].values == [1.0, 2.0]
        assert loaded.metadata == {"key1": "val1", "key2": "val2"}

    def test_save_then_load_latest_returns_same_state(self, tmp_path: Path) -> None:
        executor = _make_executor(tmp_path)
        original = _SampleState(round_num=7, survivors=["x", "y", "z"], score=0.333)

        executor.save_attack_state("round_0007", original, "halving")
        result = executor.load_latest_attack_state(_SampleState, "halving")

        assert result is not None
        key, loaded = result
        assert key == "round_0007"
        assert loaded == original

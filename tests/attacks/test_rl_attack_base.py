# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for RL attack base class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import pytest
from prompt_siren.attacks.rl_attack_base import RLAttackBase
from prompt_siren.attacks.simple_attack_base import InjectionContext
from prompt_siren.types import InjectionAttacksDict, StrContentAttack
from pydantic import BaseModel
from tests.conftest import MockEnvState


class MockRLAttackConfig(BaseModel):
    """Mock config for testing RLAttackBase."""

    test_param: str = "default"


@dataclass
class MockRLAttack(RLAttackBase[MockEnvState, str, str, StrContentAttack]):
    """Concrete implementation of RLAttackBase for testing."""

    name: ClassVar[str] = "mock_rl"
    _config: MockRLAttackConfig = field(default_factory=MockRLAttackConfig)
    _trained: bool = field(default=False, init=False)
    _loaded: bool = field(default=False, init=False)
    _saved: bool = field(default=False, init=False)

    @property
    def config(self) -> MockRLAttackConfig:
        return self._config

    async def train(
        self,
        couples: Sequence,
        executor,
    ) -> None:
        """Mock training that just sets a flag."""
        self._trained = True

    def generate_attack_from_policy(
        self,
        context: InjectionContext[MockEnvState, str, str, StrContentAttack],
    ) -> InjectionAttacksDict[StrContentAttack]:
        """Generate mock attack payload."""
        attack = StrContentAttack(content=f"mock_injection_{context.malicious_goal}")
        return dict.fromkeys(context.available_vectors, attack)

    def _save_model(self, model_dir: Path) -> None:
        """Mock save that creates a marker file."""
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.bin").write_text("mock_model")
        self._saved = True

    def _load_model(self, model_dir: Path) -> None:
        """Mock load that sets a flag."""
        self._loaded = True


class MockExecutor:
    """Mock executor for testing RLAttackBase."""

    def __init__(self, job_dir: Path | None = None):
        self._job_dir = job_dir
        self._resume_info = None

    @property
    def job_dir(self) -> Path | None:
        return self._job_dir

    @property
    def resume_info(self):
        return self._resume_info


class TestRLAttackBase:
    """Tests for RLAttackBase class."""

    def test_get_model_dir(self):
        """Test model directory path construction."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            model_dir = attack._get_model_dir(executor)

            assert model_dir == Path(tmp_dir) / "attack_model" / "mock_rl"

    def test_get_model_dir_no_job_dir_raises(self):
        """Test that missing job directory raises ValueError."""
        attack = MockRLAttack()
        executor = MockExecutor(job_dir=None)

        with pytest.raises(ValueError, match="requires a job directory"):
            attack._get_model_dir(executor)

    def test_model_exists_false_when_dir_missing(self):
        """Test that model_exists returns False when directory doesn't exist."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            assert attack._model_exists(executor) is False

    def test_model_exists_false_when_dir_empty(self):
        """Test that model_exists returns False when directory is empty."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            model_dir = attack._get_model_dir(executor)
            model_dir.mkdir(parents=True)

            assert attack._model_exists(executor) is False

    def test_model_exists_false_without_marker(self):
        """Test that model_exists returns False when files exist but no marker."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            model_dir = attack._get_model_dir(executor)
            model_dir.mkdir(parents=True)
            # Add some files but no completion marker
            (model_dir / "model.bin").write_text("content")
            (model_dir / "config.json").write_text("{}")

            assert attack._model_exists(executor) is False

    def test_model_exists_true_with_marker(self):
        """Test that model_exists returns True when completion marker exists."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            model_dir = attack._get_model_dir(executor)
            model_dir.mkdir(parents=True)
            # Add completion marker
            (model_dir / "training_complete.marker").write_text("done")

            assert attack._model_exists(executor) is True

    def test_model_exists_false_when_no_job_dir(self):
        """Test that model_exists returns False when no job directory."""
        attack = MockRLAttack()
        executor = MockExecutor(job_dir=None)

        assert attack._model_exists(executor) is False

    def test_mark_training_complete(self):
        """Test that _mark_training_complete writes the marker file."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            executor = MockExecutor(job_dir=Path(tmp_dir))
            model_dir = attack._get_model_dir(executor)
            model_dir.mkdir(parents=True)

            # Should not exist before marking
            assert attack._model_exists(executor) is False

            # Mark as complete
            attack._mark_training_complete(executor)

            # Should exist after marking
            assert attack._model_exists(executor) is True
            marker_file = model_dir / "training_complete.marker"
            assert marker_file.exists()

    def test_save_and_load_model(self):
        """Test model save and load methods."""
        attack = MockRLAttack()

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "model"

            # Save
            attack._save_model(model_dir)
            assert attack._saved is True
            assert (model_dir / "model.bin").exists()

            # Load
            attack._load_model(model_dir)
            assert attack._loaded is True

    def test_config_property(self):
        """Test config property returns correct config."""
        config = MockRLAttackConfig(test_param="custom")
        attack = MockRLAttack(_config=config)

        assert attack.config == config
        assert attack.config.test_param == "custom"

    def test_name_class_variable(self):
        """Test that name is set correctly."""
        attack = MockRLAttack()
        assert attack.name == "mock_rl"


class TestRLAttackBaseAbstract:
    """Test that RLAttackBase enforces abstract methods."""

    def test_cannot_instantiate_without_abstract_methods(self):
        """Test that RLAttackBase cannot be instantiated without implementing abstract methods."""
        # This is tested implicitly by the type system, but we verify
        # that MockRLAttack works as expected
        attack = MockRLAttack()
        assert attack is not None

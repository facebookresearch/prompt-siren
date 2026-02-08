# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Base class for reinforcement learning-based attacks.

RL attacks train a policy model on task couples to generate effective
injection payloads. This base class provides:
- Abstract interface for train() and generate_attack_from_policy()
- Model persistence utilities (save/load to job directory)
- Automatic training orchestration (train if model not found)
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import logfire
from pydantic import BaseModel

from ..tasks import TaskCouple
from ..types import InjectionAttack, InjectionAttacksDict
from .executor import EnvironmentSnapshotAtInjection, RolloutExecutor, RolloutRequest
from .results import AttackResults, CoupleAttackResult
from .simple_attack_base import InjectionContext

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class RLAttackBase(abc.ABC, Generic[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]):
    """Base class for RL-based attacks that train on task couples.

    RL attacks extend the standard attack interface with a training phase that uses
    the RolloutExecutor to:
    1. Generate injection candidates from a trainable policy (LLM)
    2. Execute rollouts to get evaluation scores
    3. Use scores as rewards to update the policy via RL algorithms
    4. Save trained model in job directory
    5. Use trained model to generate final attacks for evaluation

    Subclasses must implement:
    - train(): Train the policy on task couples
    - generate_attack_from_policy(): Generate attacks using trained policy
    - _save_model(): Save model to directory
    - _load_model(): Load model from directory
    - config property: Return the attack configuration

    Example:
        class MyRLAttack(RLAttackBase):
            name: ClassVar[str] = "my_rl_attack"

            async def train(self, couples, executor):
                # Train policy using TRL or similar
                ...

            def generate_attack_from_policy(self, context):
                # Generate injection using trained model
                ...

            def _save_model(self, model_dir):
                self.model.save_pretrained(model_dir)

            def _load_model(self, model_dir):
                self.model = AutoModel.from_pretrained(model_dir)
    """

    name: ClassVar[str]
    """Unique identifier for this attack type (used in registry)"""

    @property
    @abc.abstractmethod
    def config(self) -> BaseModel:
        """Return the attack configuration."""
        raise NotImplementedError()

    @abc.abstractmethod
    async def train(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> None:
        """Train the RL policy on task couples.

        Called when no trained model is found in the job directory.
        Should use the executor to:
        1. Discover injection points
        2. Execute rollouts to compute rewards
        3. Update policy parameters

        Args:
            couples: Task couples to train on
            executor: Rollout executor for running evaluations
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_attack_from_policy(
        self,
        context: InjectionContext[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> InjectionAttacksDict[InjectionAttackT]:
        """Generate attack payloads using the trained policy.

        Called after training (or loading) the model to produce
        final attack payloads for evaluation.

        Args:
            context: Information about the injection point

        Returns:
            Dictionary mapping injection vector IDs to attack payloads
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_model(self, model_dir: Path) -> None:
        """Save the trained model to a directory.

        Args:
            model_dir: Directory to save model files to
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _load_model(self, model_dir: Path) -> None:
        """Load a trained model from a directory.

        Args:
            model_dir: Directory containing model files
        """
        raise NotImplementedError()

    def _get_model_dir(
        self,
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> Path:
        """Get the model directory for this attack in the job directory.

        Returns:
            Path to {job_dir}/attack_model/{attack_name}/

        Raises:
            ValueError: If executor has no job directory configured
        """
        job_dir = executor.job_dir
        if job_dir is None:
            raise ValueError(
                "RLAttackBase requires a job directory for model persistence. "
                "Ensure the executor is configured with a job directory."
            )
        return job_dir / "attack_model" / self.name

    def _model_exists(
        self,
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> bool:
        """Check if a fully trained model exists in the job directory.

        This checks for a completion marker file, not just any files.
        Partial checkpoints (from TRL's save_strategy) are ignored.

        Args:
            executor: The rollout executor

        Returns:
            True if model directory exists and training completed
        """
        try:
            model_dir = self._get_model_dir(executor)
            marker_file = model_dir / "training_complete.marker"
            return marker_file.exists()
        except ValueError:
            return False

    def _mark_training_complete(
        self,
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> None:
        """Write a marker file indicating training completed successfully.

        Args:
            executor: The rollout executor
        """
        model_dir = self._get_model_dir(executor)
        marker_file = model_dir / "training_complete.marker"
        marker_file.write_text("Training completed successfully.\n")

    async def run(
        self,
        couples: Sequence[TaskCouple[EnvStateT]],
        executor: RolloutExecutor[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
    ) -> AttackResults[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        """Execute the RL attack: train if needed, then generate and evaluate attacks.

        Flow:
        1. Check if trained model exists in job directory
        2. If not, call train() to train the model
        3. Load the trained model
        4. Discover injection points for all couples
        5. Generate attacks using generate_attack_from_policy()
        6. Execute rollouts
        7. Return results

        Args:
            couples: The task couples to attack
            executor: The rollout executor

        Returns:
            AttackResults containing results for each couple
        """
        # Step 1 & 2: Train if model doesn't exist
        if not self._model_exists(executor):
            logfire.info(
                "No trained model found, starting training",
                attack_name=self.name,
                num_couples=len(couples),
            )
            await self.train(couples, executor)
            logfire.info("Training completed", attack_name=self.name)
        else:
            logfire.info(
                "Found existing trained model, loading",
                attack_name=self.name,
                model_dir=str(self._get_model_dir(executor)),
            )

        # Step 3: Load the trained model
        model_dir = self._get_model_dir(executor)
        self._load_model(model_dir)
        logfire.info("Model loaded", attack_name=self.name)

        # Handle resume: filter out already-completed couples
        if executor.resume_info is not None:
            couples = executor.resume_info.filter_remaining(couples)
            if not couples:
                return AttackResults(couple_results=[])

        # Step 4: Discover injection points
        snapshots = await executor.discover_injection_points(couples)

        try:
            # Step 5 & 6: Generate attacks and execute rollouts
            couple_data: dict[str, tuple[EnvironmentSnapshotAtInjection, InjectionAttacksDict]] = {}
            requests: list[RolloutRequest] = []

            for snapshot in snapshots:
                if snapshot.is_terminal:
                    couple_data[snapshot.couple.id] = (snapshot, {})
                    continue

                context = InjectionContext.from_snapshot(snapshot)
                attacks = self.generate_attack_from_policy(context)
                couple_data[snapshot.couple.id] = (snapshot, attacks)

                requests.append(
                    RolloutRequest(
                        snapshot=snapshot,
                        attacks=attacks,
                    )
                )

            # Execute all rollouts
            rollout_results_by_couple: dict[str, list] = {c.id: [] for c in couples}
            if requests:
                rollout_results = await executor.execute_from_snapshots(requests)
                for result in rollout_results:
                    couple_id = result.request.snapshot.couple.id
                    rollout_results_by_couple[couple_id].append(result)

            # Step 7: Build and return results
            couple_results: list[CoupleAttackResult] = []
            for couple in couples:
                _, attacks = couple_data.get(couple.id, (None, {}))
                couple_results.append(
                    CoupleAttackResult(
                        couple=couple,
                        rollout_results=rollout_results_by_couple.get(couple.id, []),
                        generated_attacks=attacks,
                    )
                )

            return AttackResults(couple_results=couple_results)

        finally:
            # Always release snapshots
            await executor.release_snapshots(snapshots)

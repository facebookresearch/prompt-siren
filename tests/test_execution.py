"""Tests for execution backends."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prompt_siren.config.experiment_config import SlurmConfig
from prompt_siren.execution import (
    create_backend,
    LocalBackend,
    SlurmBackend,
    SlurmJobInfo,
)


class TestSlurmConfig:
    """Tests for SlurmConfig model."""

    def test_default_values(self):
        """Test default SLURM configuration values."""
        config = SlurmConfig()
        assert config.partition == "learnfair"
        assert config.time_minutes == 240
        assert config.gpus_per_node == 0
        assert config.cpus_per_task == 4
        assert config.mem_gb == 32
        assert config.constraint is None
        assert config.additional_parameters == {}

    def test_custom_values(self):
        """Test custom SLURM configuration values."""
        config = SlurmConfig(
            partition="gpu",
            time_minutes=120,
            gpus_per_node=1,
            cpus_per_task=8,
            mem_gb=64,
            constraint="volta32gb",
            additional_parameters={"qos": "high"},
        )
        assert config.partition == "gpu"
        assert config.time_minutes == 120
        assert config.gpus_per_node == 1
        assert config.cpus_per_task == 8
        assert config.mem_gb == 64
        assert config.constraint == "volta32gb"
        assert config.additional_parameters == {"qos": "high"}

    def test_model_copy_update(self):
        """Test updating SLURM config with model_copy."""
        config = SlurmConfig()
        updated = config.model_copy(update={"partition": "devlab"})
        assert updated.partition == "devlab"
        assert config.partition == "learnfair"  # Original unchanged


class TestLocalBackend:
    """Tests for LocalBackend execution."""

    def test_submit_calls_function(self):
        """Test that submit calls the function directly."""
        mock_fn = MagicMock(return_value="result")
        backend = LocalBackend()

        result = backend.submit(mock_fn, "arg1", "arg2", kwarg="value")

        assert result == "result"
        mock_fn.assert_called_once_with("arg1", "arg2", kwarg="value")

    def test_submit_collects_results(self):
        """Test that submit collects results for wait()."""
        backend = LocalBackend()

        backend.submit(lambda x: x * 2, 5)
        backend.submit(lambda x: x + 1, 10)

        results = backend.wait()
        assert results == [10, 11]

    def test_submit_batch_executes_sequentially(self):
        """Test that submit_batch runs all functions."""
        backend = LocalBackend()

        results = backend.submit_batch(
            lambda x, y: x + y,
            [(1, 2), (3, 4), (5, 6)],
        )

        assert results == [3, 7, 11]

    def test_is_async_false(self):
        """Test that local backend is synchronous."""
        backend = LocalBackend()
        assert backend.is_async is False


class TestSlurmBackend:
    """Tests for SlurmBackend execution."""

    @pytest.fixture
    def slurm_config(self):
        """Create a test SLURM config."""
        return SlurmConfig(
            partition="test_partition",
            time_minutes=30,
            cpus_per_task=2,
            mem_gb=16,
        )

    @pytest.fixture
    def log_dir(self, tmp_path):
        """Create a temporary log directory."""
        return tmp_path / "slurm_logs"

    def test_submit_creates_job(self, slurm_config, log_dir):
        """Test that submit creates a submitit job."""
        mock_executor = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "12345"
        mock_job.paths.folder = "/tmp/logs"
        mock_job.paths.stdout = "/tmp/logs/12345_0_log.out"
        mock_executor.submit.return_value = mock_job

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = SlurmBackend(slurm_config, log_dir, wait_for_completion=False)
            job_info = backend.submit(lambda x: x, "test_arg")

            assert job_info.job_id == "12345"
            mock_executor.submit.assert_called_once()

    def test_submit_configures_slurm_parameters(self, slurm_config, log_dir):
        """Test that SLURM parameters are correctly configured."""
        mock_executor = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "12345"
        mock_job.paths.folder = "/tmp/logs"
        mock_job.paths.stdout = "/tmp/logs/12345_0_log.out"
        mock_executor.submit.return_value = mock_job

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = SlurmBackend(slurm_config, log_dir, wait_for_completion=False)
            backend.submit(lambda x: x, "arg")

            # Verify executor was updated with correct params
            update_call = mock_executor.update_parameters.call_args
            assert update_call is not None
            call_kwargs = update_call[1]
            assert call_kwargs["slurm_partition"] == "test_partition"
            assert call_kwargs["timeout_min"] == 30
            assert call_kwargs["cpus_per_task"] == 2
            assert call_kwargs["mem_gb"] == 16

    def test_wait_returns_results(self, slurm_config, log_dir):
        """Test waiting for job completion returns results."""
        mock_executor = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "12345"
        mock_job.paths.folder = "/tmp/logs"
        mock_job.paths.stdout = "/tmp/logs/12345_0_log.out"
        mock_job.result.return_value = "completed"
        mock_executor.submit.return_value = mock_job

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = SlurmBackend(slurm_config, log_dir, wait_for_completion=True)
            backend.submit(lambda x: x, "arg")
            results = backend.wait()

            assert len(results) == 1
            assert results[0] == "completed"

    def test_wait_handles_exceptions(self, slurm_config, log_dir):
        """Test that wait returns exceptions from failed jobs."""
        mock_executor = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "12345"
        mock_job.paths.folder = "/tmp/logs"
        mock_job.paths.stdout = "/tmp/logs/12345_0_log.out"
        mock_job.result.side_effect = ValueError("job failed")
        mock_executor.submit.return_value = mock_job

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = SlurmBackend(slurm_config, log_dir, wait_for_completion=True)
            backend.submit(lambda x: x, "arg")
            results = backend.wait()

            assert len(results) == 1
            assert isinstance(results[0], ValueError)

    def test_is_async_true(self, slurm_config, log_dir):
        """Test that SLURM backend is asynchronous."""
        mock_executor = MagicMock()

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = SlurmBackend(slurm_config, log_dir, wait_for_completion=True)
            assert backend.is_async is True


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_create_local_backend(self):
        """Test creating a local backend."""
        backend = create_backend(use_slurm=False)
        assert isinstance(backend, LocalBackend)

    def test_create_slurm_backend(self, tmp_path):
        """Test creating a SLURM backend."""
        config = SlurmConfig()
        mock_executor = MagicMock()

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = create_backend(
                use_slurm=True,
                slurm_config=config,
                log_dir=tmp_path,
            )
            assert isinstance(backend, SlurmBackend)

    def test_create_slurm_backend_uses_default_config(self, tmp_path):
        """Test that SLURM backend uses default config if not provided."""
        mock_executor = MagicMock()

        with patch("submitit.AutoExecutor", return_value=mock_executor):
            backend = create_backend(use_slurm=True, log_dir=tmp_path)
            assert isinstance(backend, SlurmBackend)

    def test_create_slurm_backend_requires_log_dir(self):
        """Test that SLURM backend requires log_dir."""
        config = SlurmConfig()
        with pytest.raises(ValueError, match="log_dir is required"):
            create_backend(use_slurm=True, slurm_config=config)


class TestSlurmJobInfo:
    """Tests for SlurmJobInfo dataclass."""

    def test_job_info_creation(self):
        """Test creating a SlurmJobInfo."""
        info = SlurmJobInfo(
            job_id="12345",
            job_dir=Path("/path/to/job"),
            log_file=Path("/path/to/log.out"),
        )
        assert info.job_id == "12345"
        assert info.job_dir == Path("/path/to/job")
        assert info.log_file == Path("/path/to/log.out")

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the browser dataset implementation."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prompt_siren.datasets.browser_dataset import (
    BrowserDataset,
    BrowserDatasetConfig,
    create_browser_dataset,
)
from prompt_siren.datasets.browser_dataset.config import (
    BrowserContainerConfig,
    SqliteSiteConfig,
)
from prompt_siren.environments.browser_env import BrowserTaskMetadata
from prompt_siren.tasks import TaskCouple


class TestBrowserContainerConfig:
    """Tests for BrowserContainerConfig.to_container_spec()."""

    def test_to_container_spec_sets_hostname_and_ports(self):
        """Test that to_container_spec() sets hostname and CDP port correctly."""
        browser_config = BrowserContainerConfig(cdp_port=9222)
        spec = browser_config.to_container_spec()

        assert spec.hostname == "browser"
        assert spec.ports is not None
        # ports is dict[int, int] mapping host_port -> container_port
        # Uses dynamic allocation (0 as host port) so Docker assigns an available port
        assert spec.ports == {0: 9222}
        # No command means "use image's ENTRYPOINT"
        assert spec.command is None

    def test_to_container_spec_with_custom_port(self):
        """Test that custom CDP port is rejected for the default image."""
        with pytest.raises(ValueError, match="cdp_port must be 9222"):
            BrowserContainerConfig(cdp_port=9999)

    def test_to_container_spec_with_custom_port_custom_image(self):
        """Test that custom CDP port is allowed for custom images."""
        browser_config = BrowserContainerConfig(image="custom/browser:latest", cdp_port=9999)
        spec = browser_config.to_container_spec()

        assert spec.ports is not None
        # ports is dict[int, int] mapping host_port -> container_port
        # Uses dynamic allocation (0 as host port) for the custom container port
        assert spec.ports == {0: 9999}


class TestSiteConfigGetUrl:
    """Tests for SiteConfig.get_url() method logic."""

    def test_get_url_omits_port_80(self):
        """Test that get_url() omits port 80 (standard HTTP port)."""
        config = SqliteSiteConfig(
            container_image="gitea/gitea:latest",
            hostname="gitea.dev-forge.io",
            db_path=Path("/data/gitea/gitea.db"),
            port=80,
        )
        assert config.get_url() == "http://gitea.dev-forge.io"

    def test_get_url_includes_non_standard_port(self):
        """Test that get_url() includes non-standard ports."""
        config = SqliteSiteConfig(
            container_image="gitea/gitea:latest",
            hostname="gitea.dev-forge.io",
            db_path=Path("/data/gitea/gitea.db"),
            port=3000,
        )
        assert config.get_url() == "http://gitea.dev-forge.io:3000"

    def test_get_url_base_url_override(self):
        """Test that base_url takes precedence over hostname/port."""
        config = SqliteSiteConfig(
            container_image="gitea/gitea:latest",
            hostname="gitea.dev-forge.io",
            db_path=Path("/data/gitea/gitea.db"),
            port=80,
            base_url="http://localhost:8080",
        )
        assert config.get_url() == "http://localhost:8080"


class TestBrowserDataset:
    """Tests for BrowserDataset class."""

    @pytest.fixture
    def dataset(self) -> BrowserDataset:
        """Create a dataset with default configuration and mock sandbox manager."""
        config = BrowserDatasetConfig()
        mock_manager = MagicMock()
        return create_browser_dataset(config, sandbox_manager=mock_manager)

    def test_task_couples_have_compatible_sites(self, dataset: BrowserDataset):
        """Test that task couples are defined with semantically compatible sites.

        Each couple must have at least one overlapping site between benign and
        malicious tasks, ensuring the attack is realistic (agent would encounter
        the malicious content while performing the benign task).
        """
        couples = dataset.task_couples

        assert len(couples) > 0
        assert all(isinstance(couple, TaskCouple) for couple in couples)

        for couple in couples:
            benign_meta = couple.benign.metadata
            malicious_meta = couple.malicious.metadata

            # Get sites from both tasks (all tasks now use 'sites' list)
            benign_sites: set[str] = set()
            malicious_sites: set[str] = set()

            if isinstance(benign_meta, BrowserTaskMetadata):
                benign_sites.update(benign_meta.sites)
            if isinstance(malicious_meta, BrowserTaskMetadata):
                malicious_sites.update(malicious_meta.sites)

            # At least one site should overlap or malicious should include benign's site
            assert benign_sites & malicious_sites or benign_sites <= malicious_sites, (
                f"Couple {couple.benign.id}:{couple.malicious.id} has incompatible sites: "
                f"benign={benign_sites}, malicious={malicious_sites}"
            )

    def test_task_ids_unique(self, dataset: BrowserDataset):
        """Test that task IDs are unique (catches accidental duplicates)."""
        benign_ids = [t.id for t in dataset.benign_tasks]
        malicious_ids = [t.id for t in dataset.malicious_tasks]

        assert len(benign_ids) == len(set(benign_ids)), "Benign task IDs should be unique"
        assert len(malicious_ids) == len(set(malicious_ids)), "Malicious task IDs should be unique"


class TestBrowserDatasetImageBuildingMode:
    """Tests for browser dataset image building."""

    def test_get_image_build_specs_works_without_sandbox_manager(self, tmp_path: Path) -> None:
        """Verify image specs can be retrieved without instantiating full dataset."""
        config = BrowserDatasetConfig()
        # This is a classmethod, doesn't need an instance
        specs = BrowserDataset.get_image_build_specs(config, str(tmp_path))

        # Should return at least some specs for configured sites
        assert isinstance(specs, list)

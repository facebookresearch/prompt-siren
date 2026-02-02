# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Configuration for browser-based dataset."""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, model_validator, Tag
from typing_extensions import assert_never, Self

from ...environments.browser_env import SiteName
from ...sandbox_managers.image_spec import (
    BuildImageSpec,
    ImageSpec,
    PullImageSpec,
    SeederFn,
)
from ...sandbox_managers.sandbox_task_setup import ContainerSpec
from ..image_tags import make_dataset_tag
from .constants import BROWSER_IMAGE_PREFIX

logger = logging.getLogger(__name__)

# Default browser container image (Headless Chrome with CDP support)
# chromedp/headless-shell is Debian-based and designed for CDP usage
DEFAULT_BROWSER_IMAGE = "chromedp/headless-shell:latest"

# CDP port for browser container
CDP_PORT = 9222


def _get_site_seeder(site_name: str) -> SeederFn | None:
    """Get the seeder function for a site.

    Args:
        site_name: Site name (e.g., "gitea", "answer", "wikijs")

    Returns:
        Seeder function if available, None for unknown sites.

    Raises:
        RuntimeError: If seeder import fails for a known site. This is fatal
            because the image would be built without the expected seeded data,
            causing cryptic failures at runtime.
    """
    # Map site names to their module paths for lazy importing
    site_modules = {
        "gitea": "prompt_siren.datasets.browser_dataset.sites.gitea",
        "answer": "prompt_siren.datasets.browser_dataset.sites.answer",
        "wikijs": "prompt_siren.datasets.browser_dataset.sites.wikijs",
    }

    module_path = site_modules.get(site_name)
    if module_path is None:
        return None

    try:
        import importlib

        module = importlib.import_module(module_path)
        return module.generate_seed
    except (ImportError, AttributeError) as e:
        # This is a known site that MUST have a seeder - fail fast rather than
        # silently building an image without seeded data.
        raise RuntimeError(
            f"Failed to import seeder for known site {site_name}: {e}. "
            f"Ensure the seeding module and all its dependencies are installed. "
            f"Required module: {module_path}"
        ) from e


def _get_site_build_context(site_name: str) -> Path | None:
    """Get the path to a site's build context directory.

    Uses importlib.resources to locate package resources. For Docker builds,
    we need a real filesystem path, so this only works when the package is
    installed as a directory (not in a zip file).

    Args:
        site_name: Site name (e.g., "gitea", "answer", "wikijs")

    Returns:
        Path to the site's build context if it exists and contains a Dockerfile, None otherwise.
    """
    try:
        # Get the Traversable for the site subdirectory
        site_traversable = (
            files("prompt_siren.datasets.browser_dataset").joinpath("sites").joinpath(site_name)
        )

        # Convert to a real path - works for directory-based installations
        # For zip-based installations, this will be a path inside the zip
        # which won't work for Docker builds (but that's an edge case)
        real_path = Path(str(site_traversable))

        if real_path.is_dir() and (real_path / "Dockerfile").exists():
            return real_path
        logger.debug(
            "No valid Docker build context found for %s at %s (directory or Dockerfile missing)",
            site_name,
            real_path,
        )
    except (TypeError, FileNotFoundError) as e:
        # Expected when package is zip-installed or context doesn't exist
        logger.debug("Could not locate Docker build context for %s: %s", site_name, e)
    except OSError as e:
        # Unexpected filesystem error
        logger.warning("Filesystem error accessing Docker build context for %s: %s", site_name, e)
    return None


class BaseSiteConfig(BaseModel):
    """Common configuration for all sites."""

    container_image: str
    """Docker image for the site container (used as base or for pulling)."""
    hostname: str
    """Realistic hostname for the site (e.g., 'gitea.dev-forge.io').

    This hostname is used in task prompts and URLs. Docker DNS resolves
    this hostname to the container when using containerized browser mode.
    """
    port: Annotated[int, Field(ge=1, le=65535)]
    """Port the site runs on inside the container (1-65535)."""
    base_url: str | None = None
    """Override base URL for the site. If not set, defaults to http://{hostname}:{port}."""
    environment: dict[str, str] | None = None
    """Environment variables to set on the site container."""
    build_context: Path | None = None
    """Path to Docker build context with pre-seeded database.

    If set, uses BuildImageSpec to build a custom image with pre-populated data.
    If None, uses PullImageSpec to pull the container_image directly.
    """

    def get_url(self) -> str:
        """Get the effective URL for this site.

        Returns base_url if set, otherwise constructs URL from hostname and port.
        Port 80 is omitted from URL as it's the default HTTP port.
        """
        if self.base_url:
            return self.base_url
        if self.port == 80:
            return f"http://{self.hostname}"
        return f"http://{self.hostname}:{self.port}"

    def _get_image_spec(self, site_name: str | None = None) -> ImageSpec:
        """Get the appropriate image spec based on configuration.

        If build_context is explicitly set, uses that (and fails if path doesn't exist).
        Otherwise, auto-detects a pre-seeded build context based on site_name if provided.

        When a build context is found, this method also attaches the appropriate seeder
        function which will be called before building the image to generate the pre-seeded
        database.

        Args:
            site_name: Optional site name for auto-detecting build context
                       (e.g., "gitea", "answer", "wikijs")

        Returns:
            BuildImageSpec if a valid build context is found, otherwise PullImageSpec.

        Raises:
            ValueError: If build_context is explicitly set but the path doesn't exist.
        """
        # Check explicit build_context first
        explicitly_configured = self.build_context is not None
        build_path = self.build_context
        known_seeded_sites = {"gitea", "answer", "wikijs"}

        # Auto-detect build context based on site name if not explicitly set
        if build_path is None and site_name is not None:
            build_path = _get_site_build_context(site_name)
            if build_path is None and site_name in known_seeded_sites:
                raise RuntimeError(
                    f"Could not locate Docker build context for known site '{site_name}'. "
                    "Pre-seeded data is required for this site. "
                    "Ensure the package is properly installed with data directories."
                )

        if build_path is not None:
            if build_path.exists():
                # Use pre-seeded image built from context.
                # Tag based on hostname to keep it unique and consistent with other datasets.
                safe_hostname = self.hostname.replace(".", "-")

                # Get the seeder function for this site
                seeder = _get_site_seeder(site_name) if site_name else None

                return BuildImageSpec(
                    context_path=str(build_path),
                    tag=make_dataset_tag(BROWSER_IMAGE_PREFIX, "site", safe_hostname),
                    seeder=seeder,
                )
            # Explicitly configured paths must exist - fail fast
            if explicitly_configured:
                raise ValueError(
                    f"Build context path {build_path} does not exist for {self.hostname}. "
                    f"Either fix the path or remove build_context to use base image {self.container_image}."
                )
            # Known sites with seeders MUST have build context - this indicates installation issue
            if site_name in known_seeded_sites:
                raise RuntimeError(
                    f"Auto-detected build context path {build_path} does not exist for "
                    f"known site '{site_name}'. Pre-seeded data is required for this site. "
                    f"Ensure the package is properly installed with data directories."
                )
            # Unknown/custom sites can fall back gracefully
            logger.info(
                "No build context found for custom site %s, using base image %s",
                self.hostname,
                self.container_image,
            )
        # Fall back to pulling the base image
        return PullImageSpec(tag=self.container_image)

    def to_container_spec(self, site_name: str | None = None) -> ContainerSpec:
        """Convert site config to ContainerSpec for sandbox manager.

        Args:
            site_name: Optional site name for auto-detecting pre-seeded build context
        """
        return ContainerSpec(
            image_spec=self._get_image_spec(site_name),
            hostname=self.hostname,
            environment=self.environment,
            use_image_cmd=True,
            # Site containers are only accessed via the internal Docker network,
            # so no host port bindings are required (avoids host port conflicts).
            ports={},
        )


class SqliteSiteConfig(BaseSiteConfig):
    """Configuration for SQLite-backed sites (easy checkpointing via file copy)."""

    db_type: Literal["sqlite"] = "sqlite"
    """Database type (always sqlite for this config)."""
    db_path: Path
    """Path to the SQLite database file inside the container."""


class PostgresqlSiteConfig(BaseSiteConfig):
    """Configuration for PostgreSQL-backed sites (requires separate DB container)."""

    db_type: Literal["postgresql"] = "postgresql"
    """Database type (always postgresql for this config)."""
    db_image: str
    """Docker image for the PostgreSQL database container."""
    db_hostname: str = "db"
    """Hostname alias for the PostgreSQL container on the internal Docker network."""
    db_environment: dict[str, str] | None = None
    """Environment variables to set on the PostgreSQL container."""

    def to_db_container_spec(self) -> ContainerSpec:
        """Convert PostgreSQL config to a DB ContainerSpec."""
        return ContainerSpec(
            image_spec=PullImageSpec(tag=self.db_image),
            hostname=self.db_hostname,
            environment=self.db_environment,
            use_image_cmd=True,
            # Database container is internal-only (no host port bindings).
            ports={},
        )


SiteConfig = Annotated[
    Annotated[SqliteSiteConfig, Tag("sqlite")] | Annotated[PostgresqlSiteConfig, Tag("postgresql")],
    Discriminator("db_type"),
]
"""Site configuration - either SQLite or PostgreSQL backed."""


class BrowserContainerConfig(BaseModel):
    """Configuration for the browser container (Chromium with CDP)."""

    image: str = DEFAULT_BROWSER_IMAGE
    """Docker image for the browser container."""
    cdp_port: Annotated[int, Field(ge=1, le=65535)] = CDP_PORT
    """CDP port for remote debugging (1-65535).

    Note: The default image binds CDP on port 9222 via its ENTRYPOINT. If you
    need a different port, use a custom image that listens on that port.
    """

    @model_validator(mode="after")
    def validate_cdp_port(self) -> Self:
        """Validate CDP port compatibility with the default browser image."""
        if self.image == DEFAULT_BROWSER_IMAGE and self.cdp_port != CDP_PORT:
            raise ValueError(
                f"cdp_port must be {CDP_PORT} when using {DEFAULT_BROWSER_IMAGE}. "
                "The image entrypoint always binds CDP on port 9222. "
                "Use a custom image that listens on your desired port instead."
            )
        return self

    def to_container_spec(self) -> ContainerSpec:
        """Convert to ContainerSpec for sandbox manager.

        Note: chromedp/headless-shell has a built-in ENTRYPOINT that runs
        headless Chrome with CDP on port 9222. We don't set command so the
        image's default entrypoint is used.

        Port Allocation:
            Uses dynamic port allocation (host_port=0) to let Docker assign
            an available port. This prevents port conflicts when running
            multiple browser environments in parallel or during reset operations.
            The actual allocated port is stored in SandboxState.agent_port_bindings.
        """
        return ContainerSpec(
            image_spec=PullImageSpec(tag=self.image),
            hostname="browser",
            use_image_cmd=True,
            # Use dynamic port allocation: 0 means Docker assigns an available port
            ports={0: self.cdp_port},
        )


class BrowserDatasetConfig(BaseModel):
    """Configuration for browser-based dataset.

    This configuration supports multiple website containers (Gitea, Apache Answer,
    Wiki.js, VWA Classifieds) managed by the sandbox manager.

    Container Management:
        Follows the SWE-bench pattern:
        - create_batch_context(): Pulls/prepares all container images upfront
        - create_task_context(): Creates fresh browser + site containers per task
        - Complete isolation between tasks (no shared state)
        - Supports true parallel execution
    """

    # Browser container configuration
    browser: BrowserContainerConfig = Field(
        default_factory=BrowserContainerConfig,
        description="Browser container configuration.",
    )

    # SQLite sites (lightweight, easy checkpointing)
    # Pre-seeded build contexts are auto-detected from docker/{site_name}/ directories
    gitea: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="gitea/gitea:latest",
            hostname="gitea.dev-forge.io",
            db_path=Path("/data/gitea/gitea.db"),
            port=80,
            environment={
                "USER_UID": "1000",
                "USER_GID": "1000",
                "GITEA__server__HTTP_PORT": "80",
                "GITEA__server__ROOT_URL": "http://gitea.dev-forge.io",
            },
        ),
        description="Gitea configuration (Git forge with issues, PRs, code review).",
    )

    answer: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="apache/answer:latest",
            hostname="answers.dev-community.io",
            db_path=Path("/data/answer.db"),
            port=80,
        ),
        description="Apache Answer configuration (Q&A platform).",
    )

    wikijs: SqliteSiteConfig = Field(
        default=SqliteSiteConfig(
            container_image="linuxserver/wikijs:latest",
            hostname="wiki.internal-docs.io",
            db_path=Path("/config/database.sqlite"),
            port=80,
            environment={
                "PUID": "0",
                "PGID": "0",
                "PORT": "80",
                "WIKIJS_PORT": "80",
            },
        ),
        description="Wiki.js configuration (wiki platform).",
    )

    # PostgreSQL site (heavier, for complex scenarios)
    classifieds: PostgresqlSiteConfig = Field(
        default=PostgresqlSiteConfig(
            container_image="ghcr.io/bgrins/vwa_classifieds_web:1",
            hostname="marketplace.local-listings.io",
            db_image="ghcr.io/bgrins/vwa_classifieds_db:1",
            port=80,
        ),
        description="VWA Classifieds configuration (marketplace with PostgreSQL).",
    )

    def get_site_config(self, site_name: SiteName) -> SiteConfig:
        """Get configuration for a specific site.

        Args:
            site_name: Name of the site

        Returns:
            SiteConfig for the requested site
        """
        match site_name:
            case "gitea":
                return self.gitea
            case "answer":
                return self.answer
            case "wikijs":
                return self.wikijs
            case "classifieds":
                return self.classifieds
            case _:
                assert_never(site_name)

    @model_validator(mode="after")
    def check_unique_hostnames(self) -> Self:
        """Validate that all site hostnames are unique."""
        hostnames = [
            self.gitea.hostname,
            self.answer.hostname,
            self.wikijs.hostname,
            self.classifieds.hostname,
        ]
        if len(hostnames) != len(set(hostnames)):
            duplicates = [h for h in hostnames if hostnames.count(h) > 1]
            raise ValueError(
                f"Site hostnames must be unique. Duplicates: {sorted(set(duplicates))}"
            )
        return self

    def get_all_site_container_specs(self) -> dict[str, ContainerSpec]:
        """Get container specs for all sites.

        Auto-detects pre-seeded build contexts from sites/{site_name}/ directories.
        Falls back to pulling base images if no pre-seeded context is found.
        PostgreSQL-backed sites include an additional "{site}-db" container spec
        for the database service.

        Returns:
            Dictionary mapping site names to their ContainerSpecs
        """
        site_configs: dict[str, SiteConfig] = {
            "gitea": self.gitea,
            "answer": self.answer,
            "wikijs": self.wikijs,
            "classifieds": self.classifieds,
        }

        specs: dict[str, ContainerSpec] = {}
        for site_name, site_config in site_configs.items():
            specs[site_name] = site_config.to_container_spec(site_name)
            if isinstance(site_config, PostgresqlSiteConfig):
                specs[f"{site_name}-db"] = site_config.to_db_container_spec()

        return specs

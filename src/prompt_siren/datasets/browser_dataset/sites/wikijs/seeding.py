# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Wiki.js seeding module.

Generates a pre-seeded Wiki.js database for the browser dataset.
This module handles:
1. Starting a fresh Wiki.js container
2. Waiting for service to be ready
3. Completing first-run setup via GraphQL
4. Seeding with documentation pages via API
5. Extracting the populated database file
6. Cleaning up the container

Usage:
    Called automatically by the build images process when a seeder is defined.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import aiohttp

from ..models import WikiSeedData

logger = logging.getLogger(__name__)

# Container configuration
CONTAINER_NAME = "prompt-siren-seed-wikijs"
CONTAINER_IMAGE = "linuxserver/wikijs:latest"
HOST_PORT = 3080
CONTAINER_PORT = 80
DB_PATH_IN_CONTAINER = "/config/database.sqlite"
HEALTH_URL = f"http://localhost:{HOST_PORT}"
SITE_URL = "http://wiki.internal-docs.io"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin123"

CONTAINER_ENV = {
    "PUID": "0",
    "PGID": "0",
    "TZ": "UTC",
    "PORT": "80",
    "WIKIJS_PORT": "80",
}

# GraphQL mutations for Wiki.js
LOGIN_MUTATION = """
mutation Login($username: String!, $password: String!, $strategy: String!) {
  authentication {
    login(username: $username, password: $password, strategy: $strategy) {
      responseResult {
        succeeded
        errorCode
        message
      }
      jwt
    }
  }
}
"""

CREATE_PAGE_MUTATION = """
mutation CreatePage($content: String!, $description: String!, $editor: String!, $isPublished: Boolean!, $isPrivate: Boolean!, $locale: String!, $path: String!, $tags: [String]!, $title: String!) {
  pages {
    create(content: $content, description: $description, editor: $editor, isPublished: $isPublished, isPrivate: $isPrivate, locale: $locale, path: $path, tags: $tags, title: $title) {
      responseResult {
        succeeded
        errorCode
        message
      }
      page {
        id
        path
        title
      }
    }
  }
}
"""

FINALIZE_MUTATION = """
mutation Finalize($adminEmail: String!, $adminPassword: String!, $adminPasswordConfirm: String!, $siteUrl: String!, $telemetry: Boolean!) {
  system {
    finalize(adminEmail: $adminEmail, adminPassword: $adminPassword, adminPasswordConfirm: $adminPasswordConfirm, siteUrl: $siteUrl, telemetry: $telemetry) {
      responseResult {
        succeeded
        errorCode
        message
      }
    }
  }
}
"""


def _get_output_path() -> Path:
    """Get the path where the seeded database should be saved."""
    site_dir = files("prompt_siren.datasets.browser_dataset.sites.wikijs")
    return Path(str(site_dir)) / "data" / "database.sqlite"


def _load_seed_data() -> WikiSeedData:
    """Load and validate seed data from JSON file."""
    data_file = files("prompt_siren.datasets.browser_dataset.sites.wikijs").joinpath(
        "seed_data.json"
    )
    return WikiSeedData.model_validate_json(data_file.read_text())


def _run_docker(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a docker command."""
    cmd = ["docker", *args]
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _container_exists(name: str) -> bool:
    """Check if a container exists (running or stopped)."""
    result = _run_docker(["ps", "-a", "-q", "-f", f"name=^{name}$"], check=False)
    return bool(result.stdout.strip())


def _remove_container(name: str) -> None:
    """Remove a container if it exists."""
    if _container_exists(name):
        logger.info(f"Removing existing container: {name}")
        result = _run_docker(["rm", "-f", name], check=False)
        if result.returncode != 0:
            logger.warning(
                f"Container removal for {name} returned {result.returncode}: {result.stderr}. "
                "Proceeding anyway, but subsequent operations may fail if container still exists."
            )


def _start_container() -> None:
    """Start the Wiki.js container.

    Raises:
        RuntimeError: If container fails to start
    """
    _remove_container(CONTAINER_NAME)

    args = [
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{HOST_PORT}:{CONTAINER_PORT}",
    ]

    # Add environment variables
    for key, value in CONTAINER_ENV.items():
        args.extend(["-e", f"{key}={value}"])

    args.append(CONTAINER_IMAGE)

    logger.info(f"Starting container: {CONTAINER_NAME}")
    result = _run_docker(args, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start container {CONTAINER_NAME}: "
            f"exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _copy_file_from_container(src_path: str, dst_path: Path) -> None:
    """Copy a file from a container to the host.

    Raises:
        RuntimeError: If file copy fails
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "db_file"
        result = _run_docker(
            ["cp", f"{CONTAINER_NAME}:{src_path}", str(tmp_file)],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to copy {src_path} from container {CONTAINER_NAME}: "
                f"{result.stderr}. This may indicate seeding did not create the database."
            )
        shutil.copy2(tmp_file, dst_path)

    logger.info(f"Copied {src_path} to {dst_path}")


async def _wait_for_service(url: str, timeout: int = 120) -> None:
    """Wait for a service to be ready.

    Args:
        url: URL to poll for readiness
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError: If service doesn't become ready within timeout
    """
    logger.info(f"Waiting for service at {url}...")
    start = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status < 500:
                        logger.info(f"Service ready at {url}")
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(f"Service at {url} not ready yet: {type(e).__name__}: {e}")
            await asyncio.sleep(2)

    raise TimeoutError(f"Service at {url} did not become ready within {timeout} seconds")


async def _setup_wikijs_admin() -> None:
    """Set up Wiki.js via finalize API.

    Wiki.js has a first-run setup wizard. We need to complete it via API.

    Raises:
        RuntimeError: If setup fails (unless already completed)
    """
    logger.info("Setting up Wiki.js installation...")

    # Wiki.js uses GraphQL for setup
    async with aiohttp.ClientSession() as session:
        # First, wait for the setup page to be available
        # Wiki.js redirects to setup on first run
        await asyncio.sleep(10)  # Give it time to initialize

        # Complete the setup via GraphQL finalize mutation
        async with session.post(
            f"{HEALTH_URL}/graphql",
            headers={"Content-Type": "application/json"},
            json={
                "query": FINALIZE_MUTATION,
                "variables": {
                    "adminEmail": ADMIN_EMAIL,
                    "adminPassword": ADMIN_PASSWORD,
                    "adminPasswordConfirm": ADMIN_PASSWORD,
                    "siteUrl": SITE_URL,
                    "telemetry": False,
                },
            },
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = (
                    data.get("data", {})
                    .get("system", {})
                    .get("finalize", {})
                    .get("responseResult", {})
                )
                if result.get("succeeded"):
                    logger.info("Wiki.js setup completed successfully")
                else:
                    error_code = result.get("errorCode")
                    # Check if setup was already done (common case on retry)
                    if error_code in ("SetupAlreadyDone", "SystemAlreadySetup"):
                        logger.info("Wiki.js setup already completed")
                    else:
                        raise RuntimeError(
                            f"Wiki.js setup failed: {result.get('message', 'Unknown error')} "
                            f"(code: {error_code}). Cannot proceed with seeding."
                        )
            else:
                text = await resp.text()
                raise RuntimeError(
                    f"Wiki.js setup request failed: {resp.status} {text}. "
                    "Cannot proceed with seeding."
                )

    # Wait for Wiki.js to restart after setup
    await asyncio.sleep(5)
    await _wait_for_service(HEALTH_URL, timeout=60)


@dataclass
class WikiJsSeeder:
    """Seeds a Wiki.js instance with documentation pages.

    Requires admin credentials to create content via the GraphQL API.
    Loads seed data from seed_data.json in the same package.
    """

    base_url: str
    admin_email: str = ADMIN_EMAIL
    admin_password: str = ADMIN_PASSWORD
    _jwt: str | None = field(default=None, repr=False)

    async def _graphql(
        self,
        session: aiohttp.ClientSession,
        query: str,
        variables: dict | None = None,
    ) -> dict:
        """Execute a GraphQL query.

        Raises:
            RuntimeError: If the HTTP request fails or GraphQL returns errors.
        """
        headers = {"Content-Type": "application/json"}
        if self._jwt:
            headers["Authorization"] = f"Bearer {self._jwt}"

        async with session.post(
            f"{self.base_url}/graphql",
            headers=headers,
            json={"query": query, "variables": variables or {}},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"GraphQL request failed: {resp.status} {text}")
            data = await resp.json()
            # GraphQL can return HTTP 200 with errors in the response body
            if data.get("errors"):
                raise RuntimeError(f"GraphQL query returned errors: {data['errors']}")
            return data

    async def _login(self, session: aiohttp.ClientSession) -> str:
        """Login and get JWT token."""
        if self._jwt:
            return self._jwt

        result = await self._graphql(
            session,
            LOGIN_MUTATION,
            {
                "username": self.admin_email,
                "password": self.admin_password,
                "strategy": "local",
            },
        )

        login_data = result.get("data", {}).get("authentication", {}).get("login", {})
        response_result = login_data.get("responseResult", {})

        if not response_result.get("succeeded"):
            raise RuntimeError(f"Login failed: {response_result.get('message', 'Unknown error')}")

        self._jwt = login_data.get("jwt")
        if not self._jwt:
            raise RuntimeError("No JWT token in login response")

        return self._jwt

    async def create_page(
        self,
        session: aiohttp.ClientSession,
        path: str,
        title: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a wiki page."""
        result = await self._graphql(
            session,
            CREATE_PAGE_MUTATION,
            {
                "content": content,
                "description": description,
                "editor": "markdown",
                "isPublished": True,
                "isPrivate": False,
                "locale": "en",
                "path": path,
                "tags": tags or [],
                "title": title,
            },
        )

        create_data = result.get("data", {}).get("pages", {}).get("create", {})
        response_result = create_data.get("responseResult", {})

        if not response_result.get("succeeded"):
            # Page might already exist, which is okay
            error_code = response_result.get("errorCode")
            if error_code == "PageDuplicateCreate":
                logger.info(f"Page {path} already exists, skipping creation")
                return {"path": path, "title": title, "exists": True}
            raise RuntimeError(
                f"Failed to create page {path}: {response_result.get('message', 'Unknown error')}"
            )

        return create_data.get("page", {})

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Wiki.js instance with test data."""
        await self._login(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Create all pages
        for page in data.pages:
            await self.create_page(
                session,
                path=page.path,
                title=page.title,
                content=page.content,
            )


async def _run_seeding() -> None:
    """Run the seeding process against the container."""
    logger.info("Seeding Wiki.js with test data...")
    seeder = WikiJsSeeder(
        base_url=HEALTH_URL,
        admin_email=ADMIN_EMAIL,
        admin_password=ADMIN_PASSWORD,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


async def generate_seed() -> None:
    """Generate pre-seeded Wiki.js database.

    This is the main entry point called by the build images process.
    It orchestrates the full seeding workflow:
    1. Start fresh container
    2. Wait for service to be ready
    3. Complete first-run setup via GraphQL
    4. Seed with documentation pages
    5. Extract database file
    6. Clean up container
    """
    try:
        _start_container()
        await _wait_for_service(HEALTH_URL)

        await _setup_wikijs_admin()
        await _run_seeding()

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        output_path = _get_output_path()
        _copy_file_from_container(DB_PATH_IN_CONTAINER, output_path)

        logger.info("Wiki.js seed generation complete")

    finally:
        _remove_container(CONTAINER_NAME)

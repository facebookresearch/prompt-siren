# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Gitea seeding module.

Generates a pre-seeded Gitea database for the browser dataset.
This module handles:
1. Starting a fresh Gitea container
2. Waiting for service to be ready
3. Configuring the admin user
4. Seeding with test data via API
5. Extracting the populated database file
6. Cleaning up the container

Usage:
    Called automatically by the build images process when a seeder is defined.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import aiohttp

from ..models import GiteaSeedData

logger = logging.getLogger(__name__)

# Container configuration
CONTAINER_NAME = "prompt-siren-seed-gitea"
CONTAINER_IMAGE = "gitea/gitea:latest"
HOST_PORT = 3000
CONTAINER_PORT = 3000
DB_PATH_IN_CONTAINER = "/data/gitea/gitea.db"
HEALTH_URL = f"http://localhost:{HOST_PORT}"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
ADMIN_EMAIL = "admin@example.com"

CONTAINER_ENV = {
    "USER_UID": "1000",
    "USER_GID": "1000",
    "GITEA__database__DB_TYPE": "sqlite3",
    "GITEA__database__PATH": "/data/gitea/gitea.db",
    "GITEA__server__HTTP_PORT": "3000",
    "GITEA__server__ROOT_URL": HEALTH_URL,
    # Skip installation wizard by pre-configuring
    "GITEA__security__INSTALL_LOCK": "true",
    # Create default admin user
    "GITEA__service__DISABLE_REGISTRATION": "false",
}


def _get_output_path() -> Path:
    """Get the path where the seeded database should be saved."""
    site_dir = files("prompt_siren.datasets.browser_dataset.sites.gitea")
    return Path(str(site_dir)) / "data" / "gitea.db"


def _load_seed_data() -> GiteaSeedData:
    """Load and validate seed data from JSON file."""
    data_file = files("prompt_siren.datasets.browser_dataset.sites.gitea").joinpath(
        "seed_data.json"
    )
    return GiteaSeedData.model_validate_json(data_file.read_text())


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
    """Start the Gitea container.

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


async def _wait_for_service(url: str, timeout: int = 300) -> None:
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


async def _setup_admin() -> None:
    """Set up Gitea admin user via CLI inside container.

    Raises:
        RuntimeError: If admin user creation fails (unless user already exists)
    """
    # Wait a bit for Gitea to fully initialize
    await asyncio.sleep(5)

    # Create admin user via gitea CLI
    logger.info("Creating Gitea admin user...")
    result = _run_docker(
        [
            "exec",
            "--user",
            "git",
            CONTAINER_NAME,
            "gitea",
            "admin",
            "user",
            "create",
            "--admin",
            "--username",
            ADMIN_USERNAME,
            "--password",
            ADMIN_PASSWORD,
            "--email",
            ADMIN_EMAIL,
        ],
        check=False,
    )

    if result.returncode != 0:
        if "user already exists" in result.stderr.lower():
            logger.info("Admin user already exists")
        else:
            raise RuntimeError(
                f"Failed to create Gitea admin user: {result.stderr}. "
                "Seeding cannot proceed without admin access."
            )


@dataclass
class GiteaSeeder:
    """Seeds a Gitea instance with users, repositories, files, and issues.

    Requires admin credentials to create content via the Gitea API.
    Loads seed data from seed_data.json in the same package.
    """

    base_url: str
    admin_username: str = ADMIN_USERNAME
    admin_password: str = ADMIN_PASSWORD
    _token: str | None = field(default=None, repr=False)

    async def _get_token(self, session: aiohttp.ClientSession) -> str:
        """Get or create an API token."""
        if self._token:
            return self._token

        auth = aiohttp.BasicAuth(self.admin_username, self.admin_password)
        async with session.post(
            f"{self.base_url}/api/v1/users/{self.admin_username}/tokens",
            auth=auth,
            json={"name": "seeding-token", "scopes": ["all"]},
        ) as resp:
            if resp.status == 201:
                data = await resp.json()
                self._token = data["sha1"]
            elif resp.status == 422:
                # Token already exists, delete and recreate
                async with session.delete(
                    f"{self.base_url}/api/v1/users/{self.admin_username}/tokens/seeding-token",
                    auth=auth,
                ) as delete_resp:
                    if delete_resp.status in (401, 403):
                        raise RuntimeError(
                            f"Authentication/authorization error deleting token: {delete_resp.status}. "
                            "Check admin credentials."
                        )
                    if delete_resp.status not in (200, 204, 404):
                        logger.warning(
                            "Token deletion returned unexpected status %d (expected 200, 204, or 404). "
                            "Attempting token recreation anyway, but this may fail.",
                            delete_resp.status,
                        )
                return await self._get_token(session)
            else:
                raise RuntimeError(f"Failed to create token: {resp.status}")

        if self._token is None:
            raise RuntimeError("Token was not set after successful creation")
        return self._token

    def _headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"token {self._token}"}

    async def create_user(
        self,
        session: aiohttp.ClientSession,
        username: str,
        email: str,
        password: str = "password123",
        full_name: str = "",
    ) -> dict:
        """Create a new user."""
        async with session.post(
            f"{self.base_url}/api/v1/admin/users",
            headers=self._headers(),
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name,
                "must_change_password": False,
            },
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            if resp.status == 422:
                # 422 typically means user already exists - verify by checking message
                error_data = await resp.json()
                error_message = str(error_data.get("message", "")).lower()
                if "already exists" in error_message or "duplicate" in error_message:
                    logger.debug(f"User {username} already exists")
                    return {"username": username}
                raise RuntimeError(f"User creation validation failed for {username}: {error_data}")
            text = await resp.text()
            raise RuntimeError(f"Failed to create user {username}: {resp.status} {text}")

    async def create_repo(
        self,
        session: aiohttp.ClientSession,
        name: str,
        description: str,
    ) -> dict:
        """Create a new repository."""
        async with session.post(
            f"{self.base_url}/api/v1/user/repos",
            headers=self._headers(),
            json={
                "name": name,
                "description": description,
                "auto_init": True,
                "default_branch": "main",
            },
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            if resp.status == 409:
                async with session.get(
                    f"{self.base_url}/api/v1/repos/{self.admin_username}/{name}",
                    headers=self._headers(),
                ) as get_resp:
                    if get_resp.status == 200:
                        return await get_resp.json()
                    get_text = await get_resp.text()
                    raise RuntimeError(
                        f"Repo {name} appears to exist (409 on create) but couldn't retrieve it: "
                        f"{get_resp.status} {get_text}"
                    )
            text = await resp.text()
            raise RuntimeError(f"Failed to create repo {name}: {resp.status} {text}")

    async def update_file(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        path: str,
        content: str,
        message: str,
    ) -> None:
        """Update or create a file in a repository."""
        sha: str | None = None
        contents_url = f"{self.base_url}/api/v1/repos/{repo_full_name}/contents/{path}"
        async with session.get(
            contents_url,
            headers=self._headers(),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                sha = data.get("sha")
            elif resp.status != 404:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed to look up file {path} before update: {resp.status} {text}"
                )

        payload: dict = {
            "content": base64.b64encode(content.encode()).decode(),
            "message": message,
        }
        if sha:
            payload["sha"] = sha

        if sha:
            method = session.put
            expected = (200, 201)
        else:
            method = session.post
            expected = (201,)

        async with method(
            contents_url,
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status in expected:
                return
            if resp.status in (409, 422) and not sha:
                # File likely exists; retry with SHA for update.
                async with session.get(contents_url, headers=self._headers()) as get_resp:
                    if get_resp.status != 200:
                        text = await get_resp.text()
                        raise RuntimeError(
                            f"Failed to fetch file {path} after create conflict: "
                            f"{get_resp.status} {text}"
                        )
                    data = await get_resp.json()
                    sha = data.get("sha")
                if not sha:
                    raise RuntimeError(f"File {path} exists but no SHA was returned by API")
                payload["sha"] = sha
                async with session.put(
                    contents_url,
                    headers=self._headers(),
                    json=payload,
                ) as put_resp:
                    if put_resp.status in (200, 201):
                        return
                    text = await put_resp.text()
                    raise RuntimeError(
                        f"Failed to update file {path} after create conflict: "
                        f"{put_resp.status} {text}"
                    )
            text = await resp.text()
            action = "create" if not sha else "update"
            raise RuntimeError(f"Failed to {action} file {path}: {resp.status} {text}")

    async def create_issue(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        title: str,
        body: str,
    ) -> dict:
        """Create an issue."""
        async with session.post(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/issues",
            headers=self._headers(),
            json={"title": title, "body": body},
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create issue: {resp.status} {text}")

    async def create_issue_comment(
        self,
        session: aiohttp.ClientSession,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> dict:
        """Create an issue comment."""
        async with session.post(
            f"{self.base_url}/api/v1/repos/{repo_full_name}/issues/{issue_number}/comments",
            headers=self._headers(),
            json={"body": body},
        ) as resp:
            if resp.status == 201:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create comment: {resp.status} {text}")

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Gitea instance with test data."""
        await self._get_token(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Create users
        for user in data.users:
            await self.create_user(
                session,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
            )

        # Create repositories with files
        repos: dict[str, dict] = {}
        for repo_def in data.repositories:
            repo = await self.create_repo(
                session,
                name=repo_def.name,
                description=repo_def.description,
            )
            repos[repo_def.name] = repo

            # Add files
            for file_path, content in repo_def.files.items():
                await self.update_file(
                    session,
                    repo["full_name"],
                    file_path,
                    content,
                    f"Add {file_path}",
                )

        # Create issues with comments
        for issue_def in data.issues:
            repo = repos.get(issue_def.repo)
            if not repo:
                raise RuntimeError(
                    f"Issue references unknown repository '{issue_def.repo}'. "
                    f"Available repositories: {list(repos.keys())}"
                )
            issue = await self.create_issue(
                session,
                repo["full_name"],
                issue_def.title,
                issue_def.body,
            )

            for comment in issue_def.comments:
                await self.create_issue_comment(
                    session,
                    repo["full_name"],
                    issue["number"],
                    comment,
                )


async def _run_seeding() -> None:
    """Run the seeding process against the container."""
    logger.info("Seeding Gitea with test data...")
    seeder = GiteaSeeder(
        base_url=HEALTH_URL,
        admin_username=ADMIN_USERNAME,
        admin_password=ADMIN_PASSWORD,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


async def generate_seed() -> None:
    """Generate pre-seeded Gitea database.

    This is the main entry point called by the build images process.
    It orchestrates the full seeding workflow:
    1. Start fresh container
    2. Wait for service to be ready
    3. Configure admin user
    4. Seed with test data
    5. Extract database file
    6. Clean up container
    """
    try:
        _start_container()
        await _wait_for_service(HEALTH_URL)

        await _setup_admin()
        await _run_seeding()

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        output_path = _get_output_path()
        _copy_file_from_container(DB_PATH_IN_CONTAINER, output_path)

        logger.info("Gitea seed generation complete")

    finally:
        _remove_container(CONTAINER_NAME)

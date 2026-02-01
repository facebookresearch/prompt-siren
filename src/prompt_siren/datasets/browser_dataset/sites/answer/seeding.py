# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Apache Answer seeding module.

Generates a pre-seeded Apache Answer database for the browser dataset.
This module handles:
1. Starting a fresh Answer container
2. Waiting for service to be ready
3. Running installation wizard via API
4. Seeding with test data via API
5. Extracting the populated database file
6. Cleaning up the container

Usage:
    Called automatically by the build images process when a seeder is defined.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import aiohttp

from ..models import AnswerSeedData

logger = logging.getLogger(__name__)

# Container configuration
CONTAINER_NAME = "prompt-siren-seed-answer"
CONTAINER_IMAGE = "apache/answer:latest"
HOST_PORT = 9080
CONTAINER_PORT = 80
DB_PATH_IN_CONTAINER = "/data/answer.db"
HEALTH_URL = f"http://localhost:{HOST_PORT}"
SITE_URL = "http://answers.dev-community.io"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
ADMIN_EMAIL = "admin@example.com"
AUTO_INSTALL = True

CONTAINER_ENV = {
    "ANSWER_DEBUG": "true",
    "AUTO_INSTALL": "true",
    "DB_TYPE": "sqlite3",
    "DB_FILE": DB_PATH_IN_CONTAINER,
    "LANGUAGE": "en-US",
    "SITE_NAME": "Dev Community Q&A",
    "SITE_URL": SITE_URL,
    "CONTACT_EMAIL": ADMIN_EMAIL,
    "ADMIN_NAME": ADMIN_USERNAME,
    "ADMIN_PASSWORD": ADMIN_PASSWORD,
    "ADMIN_EMAIL": ADMIN_EMAIL,
    "EXTERNAL_CONTENT_DISPLAY": "always_display",
}


def _get_output_path() -> Path:
    """Get the path where the seeded database should be saved."""
    site_dir = files("prompt_siren.datasets.browser_dataset.sites.answer")
    return Path(str(site_dir)) / "data" / "answer.db"


def _load_seed_data() -> AnswerSeedData:
    """Load and validate seed data from JSON file."""
    data_file = files("prompt_siren.datasets.browser_dataset.sites.answer").joinpath(
        "seed_data.json"
    )
    return AnswerSeedData.model_validate_json(data_file.read_text())


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
    """Start the Answer container.

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


async def _setup_answer_admin() -> None:
    """Set up Answer via installation API.

    Raises:
        RuntimeError: If any installation step fails
    """
    if AUTO_INSTALL:
        logger.info("AUTO_INSTALL enabled; skipping installation API steps")
        return
    logger.info("Setting up Answer installation...")

    async with aiohttp.ClientSession() as session:
        # Step 1: Check installation status
        async with session.get(f"{HEALTH_URL}/installation/base-info") as resp:
            if resp.status != 200:
                logger.debug(f"Installation info check returned: {resp.status}")

        # Step 2: Configure database (SQLite)
        logger.info("Configuring Answer database...")
        async with session.post(
            f"{HEALTH_URL}/installation/db/check",
            json={
                "db_type": "sqlite3",
                "db_file": "/data/answer.db",
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Answer database configuration failed: {resp.status} {text}. "
                    "Cannot proceed with seeding."
                )

        # Step 3: Create config file
        logger.info("Creating Answer configuration...")
        async with session.post(
            f"{HEALTH_URL}/installation/config-file",
            json={
                "lang": "en_US",
                "site_name": "Dev Community Q&A",
                "site_url": HEALTH_URL,
                "contact_email": ADMIN_EMAIL,
                "admin_name": ADMIN_USERNAME,
                "admin_password": ADMIN_PASSWORD,
                "admin_email": ADMIN_EMAIL,
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Answer configuration failed: {resp.status} {text}. "
                    "Cannot proceed with seeding."
                )

        # Step 4: Initialize database
        logger.info("Initializing Answer database...")
        async with session.post(
            f"{HEALTH_URL}/installation/init",
            json={
                "db_type": "sqlite3",
                "db_file": "/data/answer.db",
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Answer initialization failed: {resp.status} {text}. "
                    "Cannot proceed with seeding."
                )

    logger.info("Answer installation completed successfully")

    # Wait for Answer to restart after installation
    await asyncio.sleep(5)
    await _wait_for_service(HEALTH_URL, timeout=60)


@dataclass
class AnswerSeeder:
    """Seeds an Apache Answer instance with questions, answers, and comments.

    Requires admin credentials to create content via the Answer API.
    Loads seed data from seed_data.json in the same package.
    """

    base_url: str
    admin_username: str = ADMIN_USERNAME
    admin_password: str = ADMIN_PASSWORD
    _token: str | None = field(default=None, repr=False)

    @staticmethod
    def _extract_id(payload: dict) -> str:
        """Extract an object id from Answer API responses."""
        data = payload.get("data", {})
        if isinstance(data, dict):
            if data.get("id"):
                return str(data["id"])
            info = data.get("info", {})
            if isinstance(info, dict) and info.get("id"):
                return str(info["id"])
        return ""

    async def _login(self, session: aiohttp.ClientSession) -> str:
        """Login and get access token."""
        if self._token:
            return self._token

        url = f"{self.base_url}/answer/api/v1/user/login/email"
        payload = {
            "e_mail": f"{self.admin_username}@example.com",
            "pass": self.admin_password,
        }
        last_error: str | None = None

        max_attempts = 30
        for attempt in range(1, max_attempts + 1):
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except aiohttp.ContentTypeError:
                        text = await resp.text()
                        last_error = (
                            f"Login returned non-JSON response (attempt {attempt}/10): {text[:200]}"
                        )
                    else:
                        self._token = data.get("data", {}).get("access_token")
                        if self._token:
                            return self._token
                        last_error = f"No access token in response: {data}"
                else:
                    text = await resp.text()
                    last_error = f"Failed to login: {resp.status} {text}"

            await asyncio.sleep(2)

        raise RuntimeError(f"Failed to login after {max_attempts} attempts: {last_error}")

    def _headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self._token}"}

    async def create_question(
        self,
        session: aiohttp.ClientSession,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Create a new question."""
        payload = {
            "title": title,
            "content": content,
            "tags": [{"slug_name": tag} for tag in (tags or [])],
        }

        async with session.post(
            f"{self.base_url}/answer/api/v1/question",
            headers=self._headers(),
            json=payload,
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create question: {resp.status} {text}")

    async def create_answer(
        self,
        session: aiohttp.ClientSession,
        question_id: str,
        content: str,
    ) -> dict:
        """Create an answer to a question."""
        payload = {
            "question_id": question_id,
            "content": content,
        }
        max_attempts = 5
        last_error = "Unknown error"
        for attempt in range(1, max_attempts + 1):
            async with session.post(
                f"{self.base_url}/answer/api/v1/answer",
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                text = await resp.text()
                last_error = f"{resp.status} {text}"
                if resp.status == 403:
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        data = {}
                    if (
                        data.get("reason") == "error.answer.restrict_answer"
                        and attempt < max_attempts
                    ):
                        await asyncio.sleep(2)
                        continue
                raise RuntimeError(f"Failed to create answer: {last_error}")
        raise RuntimeError(f"Failed to create answer after {max_attempts} attempts: {last_error}")

    async def create_comment(
        self,
        session: aiohttp.ClientSession,
        object_id: str,
        content: str,
    ) -> dict:
        """Create a comment on a question or answer."""
        async with session.post(
            f"{self.base_url}/answer/api/v1/comment",
            headers=self._headers(),
            json={
                "object_id": object_id,
                "original_text": content,
            },
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to create comment: {resp.status} {text}")

    async def update_user_bio(
        self,
        session: aiohttp.ClientSession,
        bio: str,
    ) -> dict:
        """Update the current user's bio."""
        async with session.put(
            f"{self.base_url}/answer/api/v1/user/info",
            headers=self._headers(),
            json={"bio": bio},
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Failed to update bio: {resp.status} {text}")

    async def seed(self, session: aiohttp.ClientSession) -> None:
        """Seed the Answer instance with test data."""
        await self._login(session)

        # Load and validate seed data
        data = _load_seed_data()

        # Update user bio with injection vector
        if data.user_bio:
            await self.update_user_bio(session, bio=data.user_bio)

        # Create all questions with their answers and comments
        for q_def in data.questions:
            question = await self.create_question(
                session,
                title=q_def.title,
                content=q_def.content,
                tags=q_def.tags,
            )

            q_id = self._extract_id(question)
            if not q_id:
                raise RuntimeError(
                    f"Answer API returned success for question '{q_def.title}' but no ID. "
                    f"Response: {question}. This may indicate an API schema change or server error."
                )

            for idx, answer_def in enumerate(q_def.answers):
                if idx > 0:
                    logger.info(
                        "Skipping extra answer for '%s' to avoid restrict_answer", q_def.title
                    )
                    continue
                answer = await self.create_answer(
                    session,
                    question_id=q_id,
                    content=answer_def.content,
                )

                a_id = self._extract_id(answer)
                if not a_id:
                    raise RuntimeError(
                        f"Answer API returned success for answer to '{q_def.title}' but no ID. "
                        f"Response: {answer}. This may indicate an API schema change or server error."
                    )

                for comment in answer_def.comments:
                    await self.create_comment(
                        session,
                        object_id=a_id,
                        content=comment,
                    )


async def _run_seeding() -> None:
    """Run the seeding process against the container."""
    logger.info("Seeding Answer with test data...")
    seeder = AnswerSeeder(
        base_url=HEALTH_URL,
        admin_username=ADMIN_USERNAME,
        admin_password=ADMIN_PASSWORD,
    )

    async with aiohttp.ClientSession() as session:
        await seeder.seed(session)


async def generate_seed() -> None:
    """Generate pre-seeded Answer database.

    This is the main entry point called by the build images process.
    It orchestrates the full seeding workflow:
    1. Start fresh container
    2. Wait for service to be ready
    3. Run installation wizard via API
    4. Seed with test data
    5. Extract database file
    6. Clean up container
    """
    try:
        _start_container()
        await _wait_for_service(HEALTH_URL)

        await _setup_answer_admin()
        await _run_seeding()

        # Give it a moment to persist
        await asyncio.sleep(2)

        # Copy database out
        output_path = _get_output_path()
        _copy_file_from_container(DB_PATH_IN_CONTAINER, output_path)

        logger.info("Answer seed generation complete")

    finally:
        _remove_container(CONTAINER_NAME)

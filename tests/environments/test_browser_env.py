# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for BrowserEnvironment."""

from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
from prompt_siren.environments.browser_env import (
    _setup_page_with_capture,
    apply_injections,
    BrowserEnvironment,
    BrowserTaskMetadata,
    InjectionError,
)
from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_task_setup import ContainerSpec
from prompt_siren.tasks import BenignTask, MaliciousTask, TaskCouple
from prompt_siren.types import StrContentAttack
from pydantic import HttpUrl

pytestmark = pytest.mark.anyio

# Default CDP port used in tests (simulating dynamically allocated port)
DEFAULT_TEST_CDP_HOST_PORT = 32768


def create_mock_sandbox_state(agent_port_bindings: dict[int, int] | None = None) -> MagicMock:
    """Create a mock sandbox state with proper port bindings.

    Args:
        agent_port_bindings: Port bindings to use. Defaults to {9222: DEFAULT_TEST_CDP_HOST_PORT}
                            which simulates dynamic allocation of the CDP port.
    """
    mock_state = MagicMock()
    mock_state.agent_port_bindings = agent_port_bindings or {9222: DEFAULT_TEST_CDP_HOST_PORT}
    return mock_state


@pytest.fixture
def mock_sandbox_manager() -> MagicMock:
    """Create a mock sandbox manager for testing."""
    manager = MagicMock()
    manager.clone = AsyncMock()
    manager.setup_batch = AsyncMock()
    manager.setup_task = AsyncMock()
    return manager


@pytest.fixture
def browser_container_spec() -> ContainerSpec:
    """Create browser container spec.

    Uses dynamic port allocation (host_port=0) to match production config.
    The container port is 9222 (CDP port).
    """
    return ContainerSpec(
        image_spec=PullImageSpec(tag="chromedp/headless-shell:latest"),
        hostname="browser",
        ports={0: 9222},  # Dynamic allocation: 0 means Docker assigns a port
    )


@pytest.fixture
def site_container_specs() -> dict[str, ContainerSpec]:
    """Create site container specs.

    NOTE: PR1 includes only Gitea. PR2 will add Answer, WikiJS, etc.
    """
    return {
        "gitea": ContainerSpec(
            image_spec=PullImageSpec(tag="gitea/gitea:latest"),
            hostname="gitea.dev-forge.io",
            ports={80: 80},
        ),
    }


async def _mock_render_fn(page, attacks):
    """Mock render function for testing."""
    return "rendered"


@pytest.fixture
def browser_env(
    mock_sandbox_manager: MagicMock,
    browser_container_spec: ContainerSpec,
    site_container_specs: dict[str, ContainerSpec],
) -> BrowserEnvironment:
    """Create a BrowserEnvironment instance for testing.

    NOTE: PR1 includes only Gitea vectors. PR2 will add Answer, WikiJS, etc.
    """
    injection_ids = ["gitea_issue_content", "gitea_readme"]
    return BrowserEnvironment(
        name="test-browser",
        all_injection_ids=injection_ids,
        sandbox_manager=mock_sandbox_manager,
        browser_container_spec=browser_container_spec,
        site_container_specs=site_container_specs,
        render_fn=_mock_render_fn,
    )


class TestGetInjectableIds:
    """Tests for get_injectable_ids method."""

    async def test_finds_single_injection_id(self, browser_env: BrowserEnvironment):
        """Test finding a single injection ID in page content."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>Issue: {gitea_issue_content}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert "gitea_issue_content" in result
        assert len(result) == 1

    async def test_finds_multiple_injection_ids(self, browser_env: BrowserEnvironment):
        """Test finding multiple injection IDs in page content."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>{gitea_issue_content} and {gitea_readme}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert "gitea_issue_content" in result
        assert "gitea_readme" in result
        assert len(result) == 2

    async def test_does_not_match_partial_braces(self, browser_env: BrowserEnvironment):
        """Test that incomplete braces don't match."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>gitea_issue_content without braces</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        assert result == []

    async def test_only_matches_known_injection_ids(self, browser_env: BrowserEnvironment):
        """Test that only IDs in all_injection_ids are matched, not arbitrary {text}."""
        mock_page = MagicMock()
        mock_page.content = AsyncMock(
            return_value="<html><body>{unknown_vector} and {gitea_readme}</body></html>"
        )

        result = await browser_env.get_injectable_ids(mock_page)

        # Should only find gitea_readme (in all_injection_ids), not unknown_vector
        assert result == ["gitea_readme"]


class TestGetSitesFromTask:
    """Tests for _get_sites_from_task method."""

    def test_single_site_from_benign_task(self, browser_env: BrowserEnvironment):
        """Test extracting single site from BenignTask."""
        task = BenignTask(
            id="test_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )

        result = browser_env._get_sites_from_task(task)

        assert result == ["gitea"]

    # NOTE: Cross-site tests removed for PR1 (only Gitea). PR2 will add them back.

    def test_combines_sites_from_task_couple_same_site(self, browser_env: BrowserEnvironment):
        """Test that TaskCouple deduplicates same site from both tasks."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple)

        # Should deduplicate
        assert result == ["gitea"]

    # NOTE: Test deduplicates_same_site moved above as test_combines_sites_from_task_couple_same_site

    def test_include_malicious_false_returns_only_benign_sites(
        self, browser_env: BrowserEnvironment
    ):
        """Test that include_malicious=False excludes malicious task sites."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        result = browser_env._get_sites_from_task(couple, include_malicious=False)

        # Should only return benign task's sites
        assert result == ["gitea"]

    def test_extracts_first_site_for_url_resolution(self, browser_env: BrowserEnvironment):
        """Test that first site can be used for URL resolution."""
        task = BenignTask(
            id="gitea_task",
            prompt="Do something on gitea",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )

        result = browser_env._get_sites_from_task(task)

        # First element is primary site for URL resolution
        assert result[0] == "gitea"


class TestCreateTaskSetup:
    """Tests for _create_task_setup method."""

    def test_creates_setup_for_single_site_task(self, browser_env: BrowserEnvironment):
        """Test creating TaskSetup for a single-site benign task."""
        task = BenignTask(
            id="gitea_find_issue",
            prompt="Find the issue",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )

        setup = browser_env._create_task_setup(task)

        assert setup.task_id == "gitea_find_issue"
        assert setup.agent_container.name == "browser"
        assert "gitea" in setup.service_containers
        assert setup.network_config is not None
        assert setup.network_config.name == "browser-net-gitea_find_issue"
        assert setup.network_config.internal is False

    # NOTE: Cross-site task test removed for PR1 (only Gitea). PR2 will add it back.

    def test_creates_setup_for_task_couple(self, browser_env: BrowserEnvironment):
        """Test creating TaskSetup for a TaskCouple."""
        benign = BenignTask(
            id="benign_task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        malicious = MaliciousTask(
            id="malicious_task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        setup = browser_env._create_task_setup(couple)

        # Couple ID format is "benign_id:malicious_id"
        assert setup.task_id == "benign_task:malicious_task"
        # Should include gitea container (deduplicated since both tasks use same site)
        assert "gitea" in setup.service_containers
        assert len(setup.service_containers) == 1

    def test_sanitizes_task_id_for_network_name(self, browser_env: BrowserEnvironment):
        """Test that task IDs with special characters are sanitized for network names."""
        benign = BenignTask(
            id="benign/task",
            prompt="Do something",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        malicious = MaliciousTask(
            id="malicious:task",
            goal="Attack",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )
        couple = TaskCouple(benign=benign, malicious=malicious)

        setup = browser_env._create_task_setup(couple)

        # Colons and slashes should be replaced with dashes
        assert setup.network_config is not None
        assert ":" not in setup.network_config.name
        assert "/" not in setup.network_config.name
        assert setup.network_config.name == "browser-net-benign-task-malicious-task"

    def test_raises_for_unknown_site_containers(
        self,
        mock_sandbox_manager: MagicMock,
        browser_container_spec: ContainerSpec,
    ):
        """Test that unknown sites raise ValueError.

        NOTE: This test creates its own BrowserEnvironment with an empty site_container_specs
        to test the error handling for unknown sites.
        """
        env_with_no_sites = BrowserEnvironment(
            name="test-browser-no-sites",
            all_injection_ids=[],
            sandbox_manager=mock_sandbox_manager,
            browser_container_spec=browser_container_spec,
            site_container_specs={},  # No site containers configured
            render_fn=_mock_render_fn,
        )

        # gitea is a valid SiteName, but we have no container spec for it
        task = BenignTask(
            id="gitea_task",
            prompt="Do something on gitea",
            evaluators={},
            metadata=BrowserTaskMetadata(
                sites=["gitea"], start_url=HttpUrl("http://gitea.dev-forge.io")
            ),
        )

        with pytest.raises(ValueError, match="requires site 'gitea' but no container spec"):
            env_with_no_sites._create_task_setup(task)


class TestApplyInjections:
    """Tests for apply_injections function."""

    async def test_does_nothing_when_attacks_none(self):
        """Test that apply_injections does nothing when attacks is None."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        await apply_injections(mock_page, None)

        mock_page.evaluate.assert_not_called()

    async def test_does_nothing_when_attacks_empty(self):
        """Test that apply_injections does nothing when attacks dict is empty."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        await apply_injections(mock_page, {})

        mock_page.evaluate.assert_not_called()

    async def test_replaces_single_placeholder(self):
        """Test that a single injection placeholder is replaced."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        attacks = {"vector1": StrContentAttack(content="injected payload")}
        await apply_injections(mock_page, attacks)

        mock_page.evaluate.assert_called_once()
        call_args = mock_page.evaluate.call_args
        assert call_args[0][1] == ["{vector1}", "injected payload"]

    async def test_replaces_multiple_placeholders(self):
        """Test that multiple injection placeholders are replaced."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        attacks = {
            "vector1": StrContentAttack(content="payload1"),
            "vector2": StrContentAttack(content="payload2"),
        }
        await apply_injections(mock_page, attacks)

        assert mock_page.evaluate.call_count == 2

    async def test_handles_special_characters_in_content(self):
        """Test that special characters in attack content are passed safely."""
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        # Content with characters that could be problematic if not handled
        attacks = {"vector1": StrContentAttack(content="alert('xss'); ${malicious}")}
        await apply_injections(mock_page, attacks)

        call_args = mock_page.evaluate.call_args
        # Content is passed as parameter, not interpolated into JS
        assert call_args[0][1][1] == "alert('xss'); ${malicious}"

    async def test_raises_injection_error_on_evaluate_failure(self):
        """Test that InjectionError is raised when page.evaluate fails."""
        mock_page = MagicMock()
        mock_page.url = "http://example.com/test"
        mock_page.evaluate = AsyncMock(side_effect=RuntimeError("Page navigated away"))

        attacks = {"vector1": StrContentAttack(content="payload")}

        with pytest.raises(InjectionError) as exc_info:
            await apply_injections(mock_page, attacks)

        assert "vector1" in str(exc_info.value)
        assert "http://example.com/test" in str(exc_info.value)


class TestSetupPageWithCapture:
    """Tests for _setup_page_with_capture function."""

    async def test_closes_page_on_navigation_failure(self):
        """Test that page is closed if navigation fails."""
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.route = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=RuntimeError("Navigation failed"))
        mock_page.close = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with pytest.raises(RuntimeError, match="Navigation failed"):
            await _setup_page_with_capture(mock_browser, "http://example.com")

        # Page should be closed on failure
        mock_page.close.assert_called_once()

    async def test_closes_page_on_route_setup_failure(self):
        """Test that page is closed if route setup fails."""
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.route = AsyncMock(side_effect=RuntimeError("Route setup failed"))
        mock_page.close = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        with pytest.raises(RuntimeError, match="Route setup failed"):
            await _setup_page_with_capture(mock_browser, "http://example.com")

        mock_page.close.assert_called_once()

    async def test_returns_page_and_captures_on_success(self):
        """Test successful page setup returns page and capture list."""
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.route = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.close = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)

        page, captured = await _setup_page_with_capture(mock_browser, "http://example.com")

        assert page is mock_page
        assert captured == []
        mock_page.close.assert_not_called()


class TestResetEnvState:
    """Tests for reset_env_state method."""

    async def test_closes_browser_connection(self, browser_env: BrowserEnvironment):
        """Test that reset closes the old browser connection."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_old_browser = MagicMock()
        mock_old_browser.close = AsyncMock()

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=create_mock_sandbox_state())

        mock_task_setup = MagicMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_old_browser,
            playwright=mock_playwright,
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=mock_task_setup,
            start_url="http://example.com",
        )

        await browser_env.reset_env_state(env_state)

        mock_old_browser.close.assert_called_once()

    async def test_destroys_old_containers(self, browser_env: BrowserEnvironment):
        """Test that reset destroys old containers in background."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_old_sandbox_state = create_mock_sandbox_state()
        mock_new_sandbox_state = create_mock_sandbox_state()

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=mock_new_sandbox_state)

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=mock_playwright,
            sandbox_state=mock_old_sandbox_state,
            sandbox_manager=mock_sandbox_manager,
            task_setup=MagicMock(),
            start_url="http://example.com",
        )

        await browser_env.reset_env_state(env_state)
        mock_sandbox_manager.replace_sandbox.assert_called_once_with(
            mock_old_sandbox_state, env_state.task_setup
        )

    async def test_creates_fresh_containers(self, browser_env: BrowserEnvironment):
        """Test that reset creates fresh containers."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_new_sandbox_state = create_mock_sandbox_state()
        mock_task_setup = MagicMock()

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=mock_new_sandbox_state)

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=mock_playwright,
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=mock_task_setup,
            start_url="http://example.com",
        )

        new_state = await browser_env.reset_env_state(env_state)

        mock_sandbox_manager.replace_sandbox.assert_called_once_with(
            env_state.sandbox_state, mock_task_setup
        )
        assert new_state.sandbox_state is mock_new_sandbox_state

    async def test_reconnects_browser_via_cdp(self, browser_env: BrowserEnvironment):
        """Test that reset reconnects to browser via CDP using dynamic port."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=create_mock_sandbox_state())

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=mock_playwright,
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=MagicMock(),
            start_url="http://example.com",
        )

        new_state = await browser_env.reset_env_state(env_state)

        # Should connect via CDP using dynamically allocated port from sandbox state
        mock_playwright.chromium.connect_over_cdp.assert_called_once()
        (cdp_url,) = mock_playwright.chromium.connect_over_cdp.call_args.args
        parsed = urlparse(cdp_url)
        assert parsed.scheme == "http"
        assert parsed.hostname in {"localhost", "127.0.0.1"}
        assert parsed.port == DEFAULT_TEST_CDP_HOST_PORT
        assert new_state.browser is mock_new_browser

    async def test_navigates_to_start_url(self, browser_env: BrowserEnvironment):
        """Test that reset navigates to the original start URL."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=create_mock_sandbox_state())

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=mock_playwright,
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=MagicMock(),
            start_url="http://gitea.dev-forge.io/issues",
        )

        new_state = await browser_env.reset_env_state(env_state)

        mock_new_page.goto.assert_called_once_with("http://gitea.dev-forge.io/issues")
        assert new_state.start_url == "http://gitea.dev-forge.io/issues"

    async def test_preserves_playwright_instance(self, browser_env: BrowserEnvironment):
        """Test that reset reuses the existing Playwright instance."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        mock_new_browser = MagicMock()
        mock_new_page = MagicMock()
        mock_new_page.route = AsyncMock()
        mock_new_page.goto = AsyncMock()
        mock_new_browser.new_page = AsyncMock(return_value=mock_new_page)

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_new_browser)

        mock_sandbox_manager = MagicMock()
        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=create_mock_sandbox_state())

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=mock_playwright,
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=MagicMock(),
            start_url="http://example.com",
        )

        new_state = await browser_env.reset_env_state(env_state)

        # Playwright instance should be preserved
        assert new_state.playwright is mock_playwright

    async def test_raises_if_no_ports_defined(self, mock_sandbox_manager: MagicMock):
        """Test that reset raises error if browser spec has no ports."""
        from prompt_siren.environments.browser_env import BrowserEnvState

        # Create env with browser spec that has no ports
        browser_spec_no_ports = ContainerSpec(
            image_spec=PullImageSpec(tag="chromedp/headless-shell:latest"),
            hostname="browser",
            ports={},  # Empty ports
        )
        env = BrowserEnvironment(
            name="test",
            all_injection_ids=[],
            sandbox_manager=mock_sandbox_manager,
            browser_container_spec=browser_spec_no_ports,
            site_container_specs={},
            render_fn=_mock_render_fn,
        )

        mock_sandbox_manager.replace_sandbox = AsyncMock(return_value=create_mock_sandbox_state())

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()

        env_state = BrowserEnvState(
            page=MagicMock(),
            browser=mock_browser,
            playwright=MagicMock(),
            sandbox_state=create_mock_sandbox_state(),
            sandbox_manager=mock_sandbox_manager,
            task_setup=MagicMock(),
            start_url="http://example.com",
        )

        with pytest.raises(RuntimeError, match="Browser container spec must have ports"):
            await env.reset_env_state(env_state)

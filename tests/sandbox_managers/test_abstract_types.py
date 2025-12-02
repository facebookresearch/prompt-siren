# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for sandbox manager abstract types."""

from prompt_siren.sandbox_managers.image_spec import PullImageSpec
from prompt_siren.sandbox_managers.sandbox_state import SandboxState
from prompt_siren.sandbox_managers.sandbox_task_setup import (
    ContainerSetup,
    ContainerSpec,
    NetworkConfig,
    TaskSetup,
)


class TestTaskSetup:
    """Tests for TaskSetup."""

    def test_single_container_setup_creation(self):
        """Test that a single container setup can be created."""
        agent_spec = ContainerSpec(image_spec=PullImageSpec(tag="python:3.12"))
        agent_container = ContainerSetup(name="agent", spec=agent_spec)

        setup = TaskSetup(
            task_id="task1",
            agent_container=agent_container,
            service_containers={},
            network_config=None,
        )

        assert setup.task_id == "task1"
        assert setup.agent_container.name == "agent"
        assert setup.service_containers == {}
        assert setup.network_config is None

    def test_multi_container_setup_creation(self):
        """Test that a multi-container setup can be created."""
        agent_spec = ContainerSpec(image_spec=PullImageSpec(tag="python:3.12"))
        attack_spec = ContainerSpec(image_spec=PullImageSpec(tag="alpine:latest"))

        agent_container = ContainerSetup(name="agent", spec=agent_spec)
        attack_container = ContainerSetup(name="attack_server", spec=attack_spec)
        network_config = NetworkConfig(name="test-network", internal=True)

        setup = TaskSetup(
            task_id="task1",
            agent_container=agent_container,
            service_containers={"attack_server": attack_container},
            network_config=network_config,
        )

        assert setup.task_id == "task1"
        assert setup.agent_container.name == "agent"
        assert "attack_server" in setup.service_containers
        assert setup.service_containers["attack_server"].name == "attack_server"
        assert setup.network_config is not None
        assert setup.network_config.name == "test-network"
        assert setup.network_config.internal is True


class TestSandboxState:
    """Tests for SandboxState."""

    def test_single_container_state_creation(self):
        """Test that a single container state can be created."""
        state = SandboxState(
            agent_container_id="agent123",
            service_containers={},
            execution_id="",
            network_id=None,
        )

        assert state.agent_container_id == "agent123"
        assert state.service_containers == {}
        assert state.network_id is None

    def test_multi_container_state_creation(self):
        """Test that a multi-container state can be created."""
        state = SandboxState(
            agent_container_id="agent123",
            service_containers={"attack_server": "attack456"},
            execution_id="",
            network_id="network789",
        )

        assert state.agent_container_id == "agent123"
        assert state.service_containers["attack_server"] == "attack456"
        assert state.network_id == "network789"

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Configuration validation exceptions."""


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails.

    Attributes:
        component_type: Type of component that failed validation ("agent", "environment", "attack")
        component_name: Name/identifier of the specific component (e.g., "plain", "agentdojo")
        original_error: The underlying validation error
    """

    def __init__(
        self,
        component_type: str,
        component_name: str,
        original_error: Exception,
    ):
        """Initialize ConfigValidationError.

        Args:
            component_type: Type of component (agent/environment/attack)
            component_name: Name of the specific component
            original_error: The original validation error
        """
        self.component_type = component_type
        self.component_name = component_name
        self.original_error = original_error
        super().__init__(
            f"Invalid configuration for {component_type} '{component_name}': {original_error}"
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for browser dataset injection system."""

import pytest
from prompt_siren.datasets.browser_dataset.injection import (
    ALL_BROWSER_VECTORS,
    get_vectors_for_sites,
    GITEA_VECTORS,
)


class TestGetVectorsForSites:
    """Tests for get_vectors_for_sites function."""

    def test_unknown_site_raises(self):
        """Test that unknown sites raise ValueError to catch typos."""
        with pytest.raises(ValueError, match=r"Unknown site.*unknown_site"):
            get_vectors_for_sites(["gitea", "unknown_site"])

    def test_valid_sites_return_vectors(self):
        """Test that valid sites return their vectors."""
        vectors = get_vectors_for_sites(["gitea"])
        assert vectors == GITEA_VECTORS

    def test_vector_ids_unique(self):
        """Test that all vector IDs are unique (catches accidental duplicates)."""
        assert len(ALL_BROWSER_VECTORS) == len(set(ALL_BROWSER_VECTORS))

"""Test that the rf namespace is correctly populated."""

from __future__ import annotations

import tidy3d.rf


def test_rf_import():
    """Test that tidy3d.rf can be imported."""
    assert tidy3d.rf is not None

from __future__ import annotations

import pytest

import tidy3d as td


@pytest.fixture
def ignore_size_px_precedence_warning():
    """Suppress intentional size_px precedence warnings from legacy cartesian coverage."""
    with td.log.suppress_output():
        yield

"""Regression checks for model __repr__ implementations."""

from __future__ import annotations

import tidy3d as td


def test_required_field_model_repr_is_clean() -> None:
    """__repr__ should not recurse when the class can't be instantiated with no args."""
    b = td.Box(center=(1, 2, 3), size=(2, 2, 2))
    r = repr(b)

    # These are implementation details injected by Tidy3dBaseModel and shouldn't
    # leak into the concise repr.
    assert "type=" not in r
    assert "attrs=" not in r

    # Sanity check that we still include key information.
    assert r.startswith("Box(")
    assert "center=" in r
    assert "size=" in r

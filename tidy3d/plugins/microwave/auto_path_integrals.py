"""Backwards compatibility - import from tidy3d.components.microwave.path_integrals.integrals instead."""

from __future__ import annotations

from tidy3d.components.microwave.path_integrals.integrals.auto import (
    path_integrals_from_lumped_element,
)

__all__ = [
    "path_integrals_from_lumped_element",
]

"""Compatibility shim for :mod:`tidy3d._common.components.viz.visualization_spec`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.viz.visualization_spec import (
    MATPLOTLIB_IMPORTED,
    VisualizationSpec,
    is_valid_color,
)

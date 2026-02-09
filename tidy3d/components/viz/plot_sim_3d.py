"""Compatibility shim for :mod:`tidy3d._common.components.viz.plot_sim_3d`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.viz.plot_sim_3d import (
    plot_scene_3d,
    plot_sim_3d,
)

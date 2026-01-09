"""Compatibility shim for :mod:`tidy3d._common.components.viz.flex_style`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.viz.flex_style import (
    _ORIGINAL_PARAMS,
    apply_tidy3d_params,
    restore_matplotlib_rcparams,
)

"""Compatibility shim for :mod:`tidy3d._common.components.viz.axes_utils`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.viz.axes_utils import (
    _create_unit_aware_locator,
    add_ax_if_none,
    equal_aspect,
    make_ax,
    set_default_labels_and_title,
)

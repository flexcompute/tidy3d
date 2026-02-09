"""Compatibility shim for :mod:`tidy3d._common.components.autograd.utils`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.utils import (
    asarray1d,
    contains,
    get_static,
    hasbox,
    is_tidy_box,
    pack_complex_vec,
    split_list,
)

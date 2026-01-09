"""Compatibility shim for :mod:`tidy3d._common.packaging`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.packaging import (
    F,
    _check_tidy3d_extras_available,
    check_import,
    check_tidy3d_extras_licensed_feature,
    disable_local_subpixel,
    get_numpy_major_version,
    requires_vtk,
    supports_local_subpixel,
    tidy3d_extras,
    verify_packages_import,
    vtk,
)

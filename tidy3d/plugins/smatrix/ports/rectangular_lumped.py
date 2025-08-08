"""Deprecation shim for LumpedPort; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.ports.rectangular_lumped import LumpedPort as _LumpedPort

__all__ = ["LumpedPort"]

warnings.warn(
    "tidy3d.plugins.smatrix.ports.rectangular_lumped.LumpedPort is deprecated; "
    "use tidy3d.plugins.rf.ports.rectangular_lumped.LumpedPort",
    DeprecationWarning,
    stacklevel=2,
)

LumpedPort = _LumpedPort

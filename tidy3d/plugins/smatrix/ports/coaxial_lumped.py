"""Deprecation shim for CoaxialLumpedPort; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.ports.coaxial_lumped import (
    CoaxialLumpedPort as _CoaxialLumpedPort,
)

__all__ = ["CoaxialLumpedPort"]

warnings.warn(
    "tidy3d.plugins.smatrix.ports.coaxial_lumped.CoaxialLumpedPort is deprecated; "
    "use tidy3d.plugins.rf.ports.coaxial_lumped.CoaxialLumpedPort",
    DeprecationWarning,
    stacklevel=2,
)

CoaxialLumpedPort = _CoaxialLumpedPort

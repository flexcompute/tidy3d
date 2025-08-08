"""Deprecation shim for lumped port base; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.ports.base_lumped import (
    AbstractLumpedPort as _AbstractLumpedPort,
)

__all__ = ["AbstractLumpedPort"]

warnings.warn(
    "tidy3d.plugins.smatrix.ports.base_lumped.AbstractLumpedPort is deprecated; "
    "use tidy3d.plugins.rf.ports.base_lumped.AbstractLumpedPort",
    DeprecationWarning,
    stacklevel=2,
)

AbstractLumpedPort = _AbstractLumpedPort

"""Deprecation shim for WavePort; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.ports.wave import WavePort as _WavePort

__all__ = ["WavePort"]

warnings.warn(
    "tidy3d.plugins.smatrix.ports.wave.WavePort is deprecated; "
    "use tidy3d.plugins.rf.ports.wave.WavePort",
    DeprecationWarning,
    stacklevel=2,
)

WavePort = _WavePort

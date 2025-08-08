"""Deprecation shim for terminal port base; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.ports.base_terminal import (
    AbstractTerminalPort as _AbstractTerminalPort,
)

__all__ = ["AbstractTerminalPort"]

warnings.warn(
    "tidy3d.plugins.smatrix.ports.base_terminal.AbstractTerminalPort is deprecated; "
    "use tidy3d.plugins.rf.ports.base_terminal.AbstractTerminalPort",
    DeprecationWarning,
    stacklevel=2,
)

AbstractTerminalPort = _AbstractTerminalPort

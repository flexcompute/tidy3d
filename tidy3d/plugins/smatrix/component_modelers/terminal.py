"""Deprecation shim for TerminalComponentModeler; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.component_modelers.terminal import (
    TerminalComponentModeler as _TerminalComponentModeler,
)

__all__ = ["TerminalComponentModeler"]

warnings.warn(
    "tidy3d.plugins.smatrix.component_modelers.terminal.TerminalComponentModeler is deprecated; "
    "use tidy3d.plugins.rf.component_modelers.terminal.TerminalComponentModeler",
    DeprecationWarning,
    stacklevel=2,
)

TerminalComponentModeler = _TerminalComponentModeler

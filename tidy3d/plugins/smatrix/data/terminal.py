"""Deprecation shim for terminal data APIs; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.data.terminal import (
    MicrowaveSMatrixData as _MicrowaveSMatrixData,
)
from tidy3d.plugins.rf.data.terminal import (
    TerminalComponentModelerData as _TerminalComponentModelerData,
)

__all__ = ["MicrowaveSMatrixData", "TerminalComponentModelerData"]

warnings.warn(
    "tidy3d.plugins.smatrix.data.terminal.* is deprecated; use tidy3d.plugins.rf.data.terminal.*",
    DeprecationWarning,
    stacklevel=2,
)

MicrowaveSMatrixData = _MicrowaveSMatrixData
TerminalComponentModelerData = _TerminalComponentModelerData

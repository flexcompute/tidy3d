"""Tidy3d core package imports"""

from __future__ import annotations

# TODO(FXC-3827): Drop this import once the legacy shim is removed in Tidy3D 2.12.
from tidy3d._common.web.core import environment

__all__ = ["environment"]

"""Compatibility shim for :mod:`tidy3d._common.exceptions`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.exceptions import (
    AdjointError,
    AuthenticationError,
    ConfigError,
    DataError,
    FileError,
    SetupError,
    Tidy3dError,
    Tidy3dImportError,
    Tidy3dKeyError,
    Tidy3dNotImplementedError,
    ValidationError,
    WebError,
)

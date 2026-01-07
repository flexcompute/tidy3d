"""Compatibility shim for :mod:`tidy3d._common.components.file_util`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.file_util import (
    compress_file_to_gzip,
    extract_gzip_file,
    replace_values,
)

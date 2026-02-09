"""Compatibility shim for :mod:`tidy3d._common.web.core.file_util`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.file_util import (
    _json_string_key,
    compress_file_to_gzip,
    extract_gzip_file,
    read_simulation_from_hdf5,
    read_simulation_from_hdf5_gz,
    read_simulation_from_json,
)

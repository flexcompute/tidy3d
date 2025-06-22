"""Compatibility layer for handling differences between package versions."""

from __future__ import annotations

import importlib

from packaging.version import parse as parse_version

try:
    from xarray.structure import alignment
except ImportError:
    from xarray.core import alignment


_SHAPELY_VERSION = parse_version(importlib.metadata.version("shapely"))


def _shapely_is_older_than(version: str) -> bool:
    if _SHAPELY_VERSION < parse_version(version):
        return True
    return False


__all__ = ["alignment"]

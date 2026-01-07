"""Compatibility layer for handling differences between package versions."""

from __future__ import annotations

import functools
import importlib

from packaging.version import parse as parse_version

try:
    from xarray.structure import alignment
except ImportError:
    from xarray.core import alignment

try:
    from numpy import trapezoid as np_trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as np_trapezoid


@functools.lru_cache(maxsize=8)
def _shapely_is_older_than(version: str) -> bool:
    return parse_version(importlib.metadata.version("shapely")) < parse_version(version)


__all__ = ["alignment", "np_trapezoid"]

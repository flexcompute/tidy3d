"""Compatibility layer for handling differences between package versions."""

from __future__ import annotations

import importlib
from functools import cache

from packaging.version import parse

try:
    from xarray.structure import alignment
except ImportError:
    from xarray.core import alignment

try:
    from numpy import trapezoid as np_trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as np_trapezoid

try:
    from typing import Self, TypeAlias  # Python >= 3.11
except ImportError:  # Python <3.11
    from typing import TypeAlias

    from typing_extensions import Self


@cache
def _package_is_older_than(package: str, version: str) -> bool:
    return parse(importlib.metadata.version(package)) < parse(version)


__all__ = ["Self", "TypeAlias", "_package_is_older_than", "alignment", "np_trapezoid"]

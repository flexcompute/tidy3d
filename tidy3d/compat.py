"""Compatibility layer for handling differences between package versions."""

from __future__ import annotations

try:
    from xarray.structure import alignment
except ImportError:
    from xarray.core import alignment

try:
    from typing import Self, TypeAlias  # Python >= 3.11
except ImportError:  # Python <3.11
    from typing_extensions import Self, TypeAlias

__all__ = ["Self", "TypeAlias", "alignment"]

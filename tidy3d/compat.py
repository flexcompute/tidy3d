"""Compatibility layer for handling differences between package versions."""

try:
    from xarray.structure import alignment
except ImportError:
    from xarray.core import alignment

__all__ = ["alignment"]

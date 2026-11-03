"""Public array facade over the shared array implementation."""

from __future__ import annotations

from typing import Any

from flexcompute.data.arrays.base import (
    DataArray as CoreDataArray,
)

from tidy3d.exceptions import DataError

from . import _io as _io_methods
from ._plot import _SanitizedPlotProxy


class DataArray(CoreDataArray):
    """Subclass of ``xr.DataArray`` that requires _dims to match the keys of the coords."""

    __slots__ = ()
    _core_array_type = CoreDataArray
    _data_error = DataError
    _legacy_layout_hint = "an earlier Tidy3D version"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only an explicit, same-name shared base marks a public facade. A user
        # subclass inherits the marker but has none in its own class namespace.
        for base in cls.__bases__:
            if (
                base.__module__.startswith("flexcompute.data.arrays.")
                and base.__name__ == cls.__name__
            ):
                cls._core_array_type = base
                break

    @property
    def plot(self) -> _SanitizedPlotProxy:
        """xarray-style plotting accessor on a plotting-safe DataArray copy."""
        return _SanitizedPlotProxy(self)

    _from_hdf5_handle = classmethod(_io_methods._from_hdf5_handle)
    from_hdf5 = classmethod(_io_methods.from_hdf5)
    from_file = classmethod(_io_methods.from_file)


# Keep runtime annotation lookup on the public I/O helpers working.
_io_methods.DataArray = DataArray

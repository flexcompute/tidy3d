"""Plot accessors that route xarray plotting through a plotting-safe copy."""

from __future__ import annotations

from typing import Any

import xarray as xr

from tidy3d.components.data.plot_safe import static_dataarray_for_plot
from tidy3d.exceptions import DataError


class _SanitizedPlotProxy:
    """Proxy xarray's plot accessor through a plotting-safe DataArray copy."""

    __slots__ = ("_data_array",)

    def __init__(self, data_array: xr.DataArray) -> None:
        self._data_array = data_array

    def _accessor(self) -> xr.plot.accessor.DataArrayPlotAccessor:
        """Construct xarray's plot accessor on a static plotting-safe copy."""
        return xr.DataArray.plot(static_dataarray_for_plot(self._data_array))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call xarray's plotting entrypoint on the sanitized accessor."""
        return self._accessor()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to xarray's plotting accessor."""
        return getattr(self._accessor(), name)


class _StructuredSpatialPlotProxy(_SanitizedPlotProxy):
    """Plot proxy that preserves structured spatial plot validation."""

    def __call__(self, *args: Any, field: bool = True, grid: bool = False, **kwargs: Any) -> Any:
        """Validate structured spatial-only kwargs before plotting."""
        if grid:
            raise DataError("The 'grid' argument is only supported for unstructured data.")
        if not field:
            raise DataError("The 'field' argument is only supported for unstructured data.")
        return super().__call__(*args, **kwargs)

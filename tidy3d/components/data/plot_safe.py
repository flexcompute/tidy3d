"""Make a traced DataArray safe to hand to xarray's plotting.

This needs nothing from the data layer beyond xarray itself, so it lives below
it: ``data_array`` calls it from its plot accessor, and importing that must not
pull in the spatial dataset types that :mod:`tidy3d.components.data.utils`
gathers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.autograd.utils import get_static

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["static_dataarray_for_plot"]


def static_dataarray_for_plot(field_data: xr.DataArray) -> xr.DataArray:
    """Return a plotting-safe DataArray with traced data and coordinates stripped.

    Plotting traced xarray objects can fail when xarray materializes traced
    coordinates as ``dtype=object``. Rebuild the array using static payload and
    coordinates while preserving the original array type when possible.
    """

    static_coords = {}
    for name, coord in field_data.coords.items():
        static_coords[name] = (
            coord.dims,
            get_static(coord.data),
            dict(coord.attrs),
        )

    return type(field_data)(
        get_static(field_data.data),
        coords=static_coords,
        dims=field_data.dims,
        name=field_data.name,
        attrs=dict(field_data.attrs),
    )

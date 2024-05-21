# utilities for working with autograd

from autograd.extend import Box, primitive, defvjp
from autograd.builtins import dict as dict_ag
from autograd.tracer import getval

import xarray as xr
import numpy as np

import typing
from .types import Size1D, Bound, ArrayFloat2D

# TODO: should we use ArrayBox? Box is more general

# Types for floats, or collections of floats that can also be autograd tracers
TracedFloat = typing.Union[float, Box]
TracedSize1D = typing.Union[Size1D, Box]
TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]
TracedVertices = typing.Union[ArrayFloat2D, Box]

# The data type that we pass in and out of the web.run() @autograd.primitive
AutogradFieldMap = dict_ag[tuple[str, ...], TracedFloat]


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)
    # if isinstance(x, Box):
    #     return get_static(x._value)
    # return x


# TODO: could we move this into a DataArray method?
def integrate_within_bounds(arr: xr.DataArray, dims: list[str], bounds: Bound) -> xr.DataArray:
    """integrate a data array within bounds, assumes bounds are [2, N] for N dims."""

    _arr = arr.copy()

    # order bounds with dimension first (N, 2)
    bounds = np.array(bounds).T

    # loop over all dimensions
    for dim, (bmin, bmax) in zip(dims, bounds):
        bmin = get_static(bmin)
        bmax = get_static(bmax)

        coord_values = _arr.coords[dim].values

        # reset all coordinates outside of bounds to the bounds, so that dL = 0 in integral
        coord_values[coord_values < bmin] = bmin
        coord_values[coord_values > bmax] = bmax
        _arr = _arr.assign_coords(**{dim: coord_values})

    # uses trapezoidal rule
    # https://docs.xarray.dev/en/stable/generated/xarray.DataArray.integrate.html
    return _arr.integrate(coord=dims)


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
    "integrate_within_bounds",
]

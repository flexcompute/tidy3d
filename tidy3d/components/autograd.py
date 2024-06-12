# utilities for working with autograd

import copy
import typing

import numpy as np
import xarray as xr
from autograd.builtins import dict as dict_ag
from autograd.extend import Box, defvjp, primitive
from autograd.tracer import getval

from tidy3d.components.type_util import _add_schema

from .types import ArrayFloat2D, ArrayLike, Bound, Size1D

# add schema to the Box
_add_schema(Box, title="AutogradBox", field_type_str="autograd.tracer.Box")

# make sure Boxes in tidy3d properly define VJPs for copy operations, for computational graph
_copy = primitive(copy.copy)
_deepcopy = primitive(copy.deepcopy)

defvjp(_copy, lambda ans, x: lambda g: _copy(g))
defvjp(_deepcopy, lambda ans, x, memo: lambda g: _deepcopy(g, memo))

Box.__copy__ = lambda v: _copy(v)
Box.__deepcopy__ = lambda v, memo: _deepcopy(v, memo)

# Types for floats, or collections of floats that can also be autograd tracers
TracedFloat = typing.Union[float, Box]
TracedSize1D = typing.Union[Size1D, Box]
TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]
TracedVertices = typing.Union[ArrayFloat2D, Box]


# The data type that we pass in and out of the web.run() @autograd.primitive
AutogradTraced = typing.Union[Box, ArrayLike]
AutogradFieldMap = dict_ag[tuple[str, ...], AutogradTraced]


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    return getval(x)


# TODO: could we move this into a DataArray method?
def integrate_within_bounds(arr: xr.DataArray, dims: list[str], bounds: Bound) -> xr.DataArray:
    """integrate a data array within bounds, assumes bounds are [2, N] for N dims."""

    _arr = arr.copy()

    # order bounds with dimension first (N, 2)
    bounds = np.array(bounds).T

    all_coords = {}

    # loop over all dimensions
    for dim, (bmin, bmax) in zip(dims, bounds):
        bmin = get_static(bmin)
        bmax = get_static(bmax)

        coord_values = _arr.coords[dim].values

        # reset all coordinates outside of bounds to the bounds, so that dL = 0 in integral
        coord_values[coord_values < bmin] = bmin
        coord_values[coord_values > bmax] = bmax

        all_coords[dim] = coord_values

    _arr = _arr.assign_coords(**all_coords)

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

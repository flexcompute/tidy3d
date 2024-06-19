# utilities for working with autograd

import copy
import typing

import autograd as ag
import numpy as np
import pydantic.v1 as pd
import xarray as xr
from autograd.builtins import dict as dict_ag
from autograd.extend import Box, defvjp, primitive
from autograd.tracer import getval

from tidy3d.components.type_util import _add_schema

from .types import ArrayFloat2D, ArrayLike, Bound, Complex, Size1D

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
TracedPositiveFloat = typing.Union[pd.PositiveFloat, Box]
TracedSize1D = typing.Union[Size1D, Box]
TracedSize = typing.Union[tuple[TracedSize1D, TracedSize1D, TracedSize1D], Box]
TracedCoordinate = typing.Union[tuple[TracedFloat, TracedFloat, TracedFloat], Box]
TracedVertices = typing.Union[ArrayFloat2D, Box]

# poles
TracedComplex = typing.Union[Complex, Box]
TracedPoleAndResidue = typing.Tuple[TracedComplex, TracedComplex]

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


def split_fn_z(_fn_of_z):
    """Split a function of a complex variable into two functions for real and imag outputs of the real and imag parts of input."""

    def fn(x, y):
        z = x + 1j * y
        value = _fn_of_z(z)
        return np.real(value), np.imag(value)

    def u(x, y):
        return fn(x, y)[0]

    def v(x, y):
        return fn(x, y)[1]

    return u, v


def stitch_vjp(fn, z, g):
    x, y = np.real(z), np.imag(z)
    g_x, g_y = np.real(g), np.imag(g)
    u, v = split_fn_z(fn)
    vjp_value = (
        g_x * ag.grad(u, 0)(x, y)
        - 1j * g_x * ag.grad(u, 1)(x, y)
        - g_y * ag.grad(v, 0)(x, y)
        + 1j * g_y * ag.grad(v, 1)(x, y)
    )
    import pdb

    pdb.set_trace()
    if isinstance(z, complex):
        return vjp_value + 0j
    else:
        return np.real(vjp_value)


__all__ = [
    "Box",
    "primitive",
    "defvjp",
    "get_static",
    "integrate_within_bounds",
    "stitch_vjp",
]

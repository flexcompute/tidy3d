# utilities for autograd derivative passing
from __future__ import annotations

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ..base import Tidy3dBaseModel
from ..data.data_array import ScalarFieldDataArray
from ..types import Bound, tidycomplex
from .types import PathType
from .utils import get_static

# we do this because importing these creates circular imports
FieldData = dict[str, ScalarFieldDataArray]
PermittivityData = dict[str, ScalarFieldDataArray]


class DerivativeInfo(Tidy3dBaseModel):
    """Stores derivative information passed to the ``.compute_derivatives`` methods."""

    paths: list[PathType] = pd.Field(
        ...,
        title="Paths to Traced Fields",
        description="List of paths to the traced fields that need derivatives calculated.",
    )

    E_der_map: FieldData = pd.Field(
        ...,
        title="Electric Field Gradient Map",
        description='Dataset where the field components ``("Ex", "Ey", "Ez")`` store the '
        "multiplication of the forward and adjoint electric fields. The tangential components "
        "of this dataset is used when computing adjoint gradients for shifting boundaries. "
        "All components are used when computing volume-based gradients.",
    )

    D_der_map: FieldData = pd.Field(
        ...,
        title="Displacement Field Gradient Map",
        description='Dataset where the field components ``("Ex", "Ey", "Ez")`` store the '
        "multiplication of the forward and adjoint displacement fields. The normal component "
        "of this dataset is used when computing adjoint gradients for shifting boundaries.",
    )

    eps_data: PermittivityData = pd.Field(
        ...,
        title="Permittivity Dataset",
        description="Dataset of relative permittivity values along all three dimensions. "
        "Used for automatically computing permittivity inside or outside of a simple geometry.",
    )

    eps_in: tidycomplex = pd.Field(
        title="Permittivity Inside",
        description="Permittivity inside of the ``Structure``. "
        "Typically computed from ``Structure.medium.eps_model``."
        "Used when it can not be computed from ``eps_data`` or when ``eps_approx==True``.",
    )

    eps_out: tidycomplex = pd.Field(
        ...,
        title="Permittivity Outside",
        description="Permittivity outside of the ``Structure``. "
        "Typically computed from ``Simulation.medium.eps_model``."
        "Used when it can not be computed from ``eps_data`` or when ``eps_approx==True``.",
    )

    bounds: Bound = pd.Field(
        ...,
        title="Geometry Bounds",
        description="Bounds corresponding to the structure, used in ``Medium`` calculations.",
    )

    frequency: float = pd.Field(
        ...,
        title="Frequency of adjoint simulation",
        description="Frequency at which the adjoint gradient is computed.",
    )

    eps_approx: bool = pd.Field(
        False,
        title="Use Permittivity Approximation",
        description="If ``True``, approximates outside permittivity using ``Simulation.medium``"
        "and the inside permittivity using ``Structure.medium``. "
        "Only set ``True`` for ``GeometryGroup`` handling where it is difficult to automatically "
        "evaluate the inside and outside relative permittivity for each geometry.",
    )

    def updated_paths(self, paths: list[PathType]) -> DerivativeInfo:
        """Update this ``DerivativeInfo`` with new set of paths."""
        return self.updated_copy(paths=paths)


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
    dims_integrate = [dim for dim in dims if len(_arr.coords[dim]) > 1]
    return _arr.integrate(coord=dims_integrate)


__all__ = [
    "integrate_within_bounds",
    "DerivativeInfo",
]

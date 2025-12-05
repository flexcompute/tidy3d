"""Semiconductor doping definitions."""

from __future__ import annotations

from typing import Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from tidy3d.components.autograd import TracedSize
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.geometry.base import Box
from tidy3d.constants import MICROMETER, PERCMCUBE, inf
from tidy3d.exceptions import SetupError


class AbstractDopingBox(Box):
    """Derived class from Box to deal with dopings"""

    # Override size so that we can set default values
    size: TracedSize = pd.Field(
        (inf, inf, inf),
        title="Size",
        description="Size in x, y, and z directions.",
        units=MICROMETER,
    )

    def _get_indices_in_box(self, coords: dict, meshgrid: bool = True):
        """Returns locations inside box"""

        # work out whether x,y, and z are present
        dim_missing = len(list(coords.keys())) < 3
        if dim_missing:
            for var_name in "xyz":
                if var_name not in coords:
                    coords[var_name] = [0]

        if meshgrid:
            X, Y, Z = np.meshgrid(coords["x"], coords["y"], coords["z"], indexing="ij")
        else:
            X = coords["x"]
            Y = coords["y"]
            Z = coords["z"]

        new_bounds = [list(self.bounds[0]), list(self.bounds[1])]
        for d in range(3):
            if new_bounds[0][d] == new_bounds[1][d]:
                new_bounds[0][d] = -np.inf
                new_bounds[1][d] = np.inf

        # let's assume some of these coordinates may lay outside the box
        indices_in_box = np.logical_and(new_bounds[0][0] <= X, new_bounds[1][0] >= X)
        indices_in_box = np.logical_and(indices_in_box, new_bounds[0][1] <= Y)
        indices_in_box = np.logical_and(indices_in_box, new_bounds[1][1] >= Y)
        indices_in_box = np.logical_and(indices_in_box, new_bounds[0][2] <= Z)
        indices_in_box = np.logical_and(indices_in_box, new_bounds[1][2] >= Z)

        return indices_in_box, X, Y, Z

    def _post_init_validators(self) -> None:
        # check the doping box is 3D
        if len(self.zero_dims) > 0:
            raise SetupError(
                "The doping box must be 3D. If you want a 2D doping box, please set one of the dimensions to a large or infinite size."
            )


class ConstantDoping(AbstractDopingBox):
    """
    Sets constant doping :math:`N` in the specified box with a :py:attr:`~.Box.size` and :py:attr:`concentration`. For translationally invariant behavior in one dimension, the box must have infinite size in the
    homogenous (invariant) direction.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> constant_box1 = td.ConstantDoping(center=(0, 0, 0), size=(2, 2, 2), concentration=1e18)
    >>> constant_box2 = td.ConstantDoping.from_bounds(rmin=box_coords[0], rmax=box_coords[1], concentration=1e18)
    """

    concentration: pd.NonNegativeFloat = pd.Field(
        default=0,
        title="Doping concentration density.",
        description="Doping concentration density.",
        units=PERCMCUBE,
    )

    def _get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

        indices_in_box, X, _, _ = self._get_indices_in_box(coords=coords, meshgrid=meshgrid)

        contrib = np.zeros(X.shape)
        contrib[indices_in_box] = self.concentration

        return contrib.squeeze()


class GaussianDoping(AbstractDopingBox):
    """Sets a gaussian doping in the specified box. For translationally invariant behavior in one dimension, the box must have infinite size in the
    homogenous (invariant) direction.

    Notes
    -----
    The Gaussian doping concentration :math:`N` is defined in the following manner:

    - :math:`N=N_{\\text{max}}` at locations more than :math:`\\text{width}` um away from the sides of the box.
    - :math:`N=N_{\\text{ref}}` at location on the box sides.
    - a Gaussian variation between :math:`N_{\\text{max}}` and  :math:`N_{\\text{ref}}`  at locations less than :math:`\\text{width}`
    um away from the sides.

    By definition, all sides of the box will have concentration :math:`N_{\\text{ref}}` (except the side specified
    as source) and the center of the box (:math:`\\text{width}` away from the box sides) will have a concentration
    :math:`N_{\\text{max}}`.

    .. math::

        N = \\{N_{\\text{max}}\\} \\exp \\left[
        - \\ln \\left( \\frac{\\{N_{\\text{max}}\\}}{\\{N_{\\text{ref}}\\}} \\right)
        \\left( \\frac{(x|y|z) - \\{(x|y|z)_{\\text{box}}\\}}{\\text{width}} \\right)^2
        \\right]

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> gaussian_box1 = td.GaussianDoping(
    ...     center=(0, 0, 0),
    ...     size=(2, 2, 2),
    ...     ref_con=1e15,
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    >>> gaussian_box2 = td.GaussianDoping.from_bounds(
    ...     rmin=box_coords[0],
    ...     rmax=box_coords[1],
    ...     ref_con=1e15,
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    """

    ref_con: pd.PositiveFloat = pd.Field(
        title="Reference concentration.",
        description="Reference concentration. This is the minimum concentration in the box "
        "and it is attained at the edges/faces of the box.",
        units=PERCMCUBE,
    )

    concentration: pd.PositiveFloat = pd.Field(
        title="Concentration",
        description="The concentration at the center of the box.",
        units=PERCMCUBE,
    )

    width: pd.PositiveFloat = pd.Field(
        title="Width of the gaussian.",
        description="Width of the gaussian. The concentration will transition from "
        "``concentration`` at the center of the box to ``ref_con`` at the edge/face "
        "of the box in a distance equal to ``width``. ",
        units=MICROMETER,
    )

    source: str = pd.Field(
        "xmin",
        title="Source face",
        description="Specifies the side of the box acting as the source, i.e., "
        "the face specified does not have a gaussian evolution normal to it, instead "
        "the concentration is constant from this face. Accepted values for ``source`` "
        "are [``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``]",
    )

    @cached_property
    def sigma(self):
        """The sigma parameter of the pseudo-gaussian"""
        return np.sqrt(-self.width * self.width / 2 / np.log(self.ref_con / self.concentration))

    def _get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

        indices_in_box, X, Y, Z = self._get_indices_in_box(coords=coords, meshgrid=meshgrid)

        x_contrib = np.zeros(X.shape)
        x_contrib[indices_in_box] = 1.0
        if self.source != "xmin":
            x0 = self.bounds[0][0]
            indices = np.logical_and(x0 <= X, x0 + self.width >= X)
            indices = np.logical_and(indices, indices_in_box)
            x_contrib[indices] = np.exp(
                -(X[indices] - x0 - self.width)
                * (X[indices] - x0 - self.width)
                / 2
                / self.sigma
                / self.sigma
            )
        # higher x face
        if self.source != "xmax":
            x1 = self.bounds[1][0]
            indices = np.logical_and(x1 - self.width <= X, x1 >= X)
            indices = np.logical_and(indices, indices_in_box)
            x_contrib[indices] = np.exp(
                -(X[indices] - x1 + self.width)
                * (X[indices] - x1 + self.width)
                / 2
                / self.sigma
                / self.sigma
            )

        y_contrib = np.zeros(X.shape)
        y_contrib[indices_in_box] = 1.0
        if self.source != "ymin":
            y0 = self.bounds[0][1]
            indices = np.logical_and(y0 <= Y, y0 + self.width >= Y)
            indices = np.logical_and(indices, indices_in_box)
            y_contrib[indices] = np.exp(
                -(Y[indices] - y0 - self.width)
                * (Y[indices] - y0 - self.width)
                / 2
                / self.sigma
                / self.sigma
            )
        # higher y face
        if self.source != "ymax":
            y1 = self.bounds[1][1]
            indices = np.logical_and(y1 - self.width <= Y, y1 >= Y)
            indices = np.logical_and(indices, indices_in_box)
            y_contrib[indices] = np.exp(
                -(Y[indices] - y1 + self.width)
                * (Y[indices] - y1 + self.width)
                / 2
                / self.sigma
                / self.sigma
            )

        z_contrib = np.zeros(X.shape)
        z_contrib[indices_in_box] = 1.0
        if self.source != "zmin":
            z0 = self.bounds[0][2]
            indices = np.logical_and(z0 <= Z, z0 + self.width >= Z)
            indices = np.logical_and(indices, indices_in_box)
            z_contrib[indices] = np.exp(
                -(Z[indices] - z0 - self.width)
                * (Z[indices] - z0 - self.width)
                / 2
                / self.sigma
                / self.sigma
            )
        # higher z face
        if self.source != "zmax":
            z1 = self.bounds[1][2]
            indices = np.logical_and(z1 - self.width <= Z, z1 >= Z)
            indices = np.logical_and(indices, indices_in_box)
            z_contrib[indices] = np.exp(
                -(Z[indices] - z1 + self.width)
                * (Z[indices] - z1 + self.width)
                / 2
                / self.sigma
                / self.sigma
            )

        total_contrib = x_contrib * y_contrib * z_contrib * self.concentration

        return total_contrib.squeeze()


class CustomDoping(AbstractDopingBox):
    """Sets a custom doping in the specified box.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 5)
    >>> z = np.linspace(-1, 1, 5)
    >>> data = np.random.rand(5, 5, 5)*1e18
    >>> concentration = td.SpatialDataArray(
    ...     data=data,
    ...     coords={'x': x, 'y': y, 'z': z},
    ... )
    >>> custom_box1 = td.CustomDoping(
    ...     center=(0, 0, 0),
    ...     size=(2, 2, 2),
    ...     concentration=concentration
    ... )
    >>> custom_box2 = td.CustomDoping.from_bounds(
    ...     rmin=box_coords[0],
    ...     rmax=box_coords[1],
    ...     concentration=concentration
    ... )
    """

    concentration: SpatialDataArray = pd.Field(
        ...,
        title="Doping concentration data array.",
        description="Doping concentration data array.",
        units=PERCMCUBE,
    )

    def _get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

        indices_in_box, X, Y, Z = self._get_indices_in_box(coords=coords, meshgrid=meshgrid)

        contrib = np.zeros(X.shape)
        # interpolate
        if meshgrid:
            interp_result = self.concentration.interp(coords)
            contrib[indices_in_box] = interp_result.values[indices_in_box]
        else:
            # X, Y, Z are 1D arrays of coordinates
            # interp_result = self.concentration.interp(coords)
            # contrib = np.zeros(X.shape)
            # contrib[indices_in_box] = interp_result.values[indices_in_box]

            interp_coords = coords
            interp_da = {
                name: xr.DataArray(data, dims="new_dim") for name, data in interp_coords.items()
            }
            interp_res = self.concentration.interp(
                **interp_da, kwargs={"fill_value": 0, "bounds_error": False}
            )
            contrib[indices_in_box] = interp_res.values[indices_in_box]

        return contrib.squeeze()


DopingBoxType = Union[ConstantDoping, GaussianDoping, CustomDoping]

"""File containing classes required for the setup of a DEVSIM case."""

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import Union
from tidy3d.exceptions import SetupError

from ...constants import PERCMCUBE


class AbstractDopingBox(Box):
    """Derived class from Box to deal with dopings"""

    def _normal_dim(self):
        """Returns the normal direction if the box is 2D. False otherwise"""

        normal_dim = None
        for dim in range(3):
            if self.size[dim] == np.inf:
                if normal_dim is not None:
                    raise SetupError("Only 3D and 2D boxes are considered for doping.")
                normal_dim = dim

        return normal_dim

    def _get_indices_in_box(self, coords: dict, meshgrid: bool = True):
        """Returns locations inside box"""

        # work out whether x,y, and z are present
        dim_missing = len(list(coords.keys())) < 3
        if dim_missing:
            for var_name in "xyz":
                if var_name not in coords.keys():
                    coords[var_name] = [0]

        # work out whether the dimensions are 2D
        normal_axis = None
        # normal_position = None
        for dim in range(3):
            var_name = "xyz"[dim]
            if len(coords[var_name]) == 1:
                normal_axis = dim
                # normal_position = coords[var_name][0]

        # if provided coordinates are 3D, check if box is 2D
        if normal_axis is None:
            normal_axis = self._normal_dim()

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
        indices_in_box = np.logical_and(X >= new_bounds[0][0], X <= new_bounds[1][0])
        indices_in_box = np.logical_and(indices_in_box, Y >= new_bounds[0][1])
        indices_in_box = np.logical_and(indices_in_box, Y <= new_bounds[1][1])
        indices_in_box = np.logical_and(indices_in_box, Z >= new_bounds[0][2])
        indices_in_box = np.logical_and(indices_in_box, Z <= new_bounds[1][2])

        return indices_in_box, X, Y, Z, normal_axis

    @pd.root_validator(skip_on_failure=True)
    def check_dimensions(cls, values):
        """Make sure dimensionality is specified correctly. I.e.,
        a 2D box must be defined with an inf size in the normal direction."""

        size = values["size"]
        for dim in range(3):
            if size[dim] == 0:
                zero_dim_name = "xyz"[dim]

                raise SetupError(
                    f"The doping box has been set up with 0 size in the {zero_dim_name} direction. "
                    "If this was intended to be translationally invariant, the box must have a large "
                    "or infinite ('td.inf') size in the perpendicular direction."
                )

        return values


class ConstantDoping(AbstractDopingBox):
    """
    Sets constant doping :math:`N` in the specified box with a :parameter`size` and :parameter:`concentration`.

    For translationally invariant behavior in one dimension, the box must have infinite size in the
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
        description="Doping concentration density in #/cm^3.",
        units=PERCMCUBE,
    )

    def _get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

        indices_in_box, X, _, _, normal_axis = self._get_indices_in_box(
            coords=coords, meshgrid=meshgrid
        )

        contrib = np.zeros(X.shape)
        contrib[indices_in_box] = self.concentration

        if normal_axis is not None and meshgrid:
            slices = [slice(None)] * X.ndim
            slices[normal_axis] = 0
            return contrib[tuple(slices)]
        else:
            return contrib


class GaussianDoping(AbstractDopingBox):
    """Sets a gaussian doping in the specified box.

    For translationally invariant behavior in one dimension, the box must have infinite size in the
    homogenous (invariant) direction.

    Notes
    -----
    The Gaussian doping concentration :math:`N` is defined in the following manner:

    - :math:`N=N_{\\text{max}}` at locations more than :math:``width`` um away from the sides of the box.
    - :math:`N=N_{\\text{ref}}` at location on the box sides.
    - a Gaussian variation between :math:`N_{\\text{max}}` and  :math:`N_{\\text{ref}}`  at locations less than ``width``
    um away from the sides.

    By definition, all sides of the box will have concentration :math:`N_{\\text{ref}}` (except the side specified
    as source) and the center of the box (``width`` away from the box sides) will have a concentration
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
    )

    concentration: pd.PositiveFloat = pd.Field(
        title="Concentration",
        description="The concentration at the center of the box.",
    )

    width: pd.PositiveFloat = pd.Field(
        title="Width of the gaussian.",
        description="Width of the gaussian. The concentration will transition from "
        "'concentration' at the center of the box to 'ref_con' at the edge/face "
        "of the box in a distance equal to 'width'. ",
    )

    source: str = pd.Field(
        "xmin",
        title="Source face",
        description="Specifies the side of the box acting as the source, i.e., "
        "the face specified does not have a gaussian evolution normal to it, instead "
        "the concentration is constant from this face. Accepted values for 'source' "
        "are ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']",
    )

    @cached_property
    def sigma(self):
        """The sigma parameter of the pseudo-gaussian"""
        return np.sqrt(-self.width * self.width / 2 / np.log(self.ref_con / self.concentration))

    def _get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

        indices_in_box, X, Y, Z, normal_axis = self._get_indices_in_box(
            coords=coords, meshgrid=meshgrid
        )

        x_contrib = np.ones(X.shape)
        if normal_axis != 0:
            x_contrib = np.zeros(X.shape)
            x_contrib[indices_in_box] = 1
            # lower x face
            if self.source != "xmin":
                x0 = self.bounds[0][0]
                indices = np.logical_and(X >= x0, X <= x0 + self.width)
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
                indices = np.logical_and(X >= x1 - self.width, X <= x1)
                indices = np.logical_and(indices, indices_in_box)
                x_contrib[indices] = np.exp(
                    -(X[indices] - x1 + self.width)
                    * (X[indices] - x1 + self.width)
                    / 2
                    / self.sigma
                    / self.sigma
                )

        y_contrib = np.ones(X.shape)
        if normal_axis != 1:
            y_contrib = np.zeros(X.shape)
            y_contrib[indices_in_box] = 1
            # lower y face
            if self.source != "ymin":
                y0 = self.bounds[0][1]
                indices = np.logical_and(Y >= y0, Y <= y0 + self.width)
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
                indices = np.logical_and(Y >= y1 - self.width, Y <= y1)
                indices = np.logical_and(indices, indices_in_box)
                y_contrib[indices] = np.exp(
                    -(Y[indices] - y1 + self.width)
                    * (Y[indices] - y1 + self.width)
                    / 2
                    / self.sigma
                    / self.sigma
                )

        z_contrib = np.ones(X.shape)
        if normal_axis != 2:
            z_contrib = np.zeros(X.shape)
            z_contrib[indices_in_box] = 1
            # lower z face
            if self.source != "zmin":
                z0 = self.bounds[0][2]
                indices = np.logical_and(Z >= z0, Z <= z0 + self.width)
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
                indices = np.logical_and(Z >= z1 - self.width, Z <= z1)
                indices = np.logical_and(indices, indices_in_box)
                z_contrib[indices] = np.exp(
                    -(Z[indices] - z1 + self.width)
                    * (Z[indices] - z1 + self.width)
                    / 2
                    / self.sigma
                    / self.sigma
                )

        total_contrib = x_contrib * y_contrib * z_contrib * self.concentration

        if normal_axis is not None and meshgrid:
            slices = [slice(None)] * X.ndim
            slices[normal_axis] = 0
            return total_contrib[tuple(slices)]
        else:
            return total_contrib


DopingBoxType = Union[ConstantDoping, GaussianDoping]

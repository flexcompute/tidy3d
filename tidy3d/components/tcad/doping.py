"""File containing classes required for the setup of a DEVSIM case."""

from typing import Optional

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.autograd.types import TracedSize
from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import Bound, Union
from tidy3d.constants import MICROMETER


class AbstractDopingBox(Box):
    """Derived class from Box which redefines size so that
    we have some default values"""

    # redefining size so that it doesn't fail validation when the box
    # is defined through box_coords
    size: TracedSize = pd.Field(
        (1, 1, 1),
        title="Size",
        description="Size in x, y, and z directions.",
        units=MICROMETER,
    )

    # equivalent to Box().bounds but defined here so that we can actually
    # define boxes through this
    box_coords: Optional[Bound] = pd.Field(title="Doping box coordinates", description="")

    @pd.root_validator(skip_on_failure=True)
    def check_bounds(cls, values):
        """When 'box_coords' is provided, make sure to rewrite 'size' and 'center' appropriately"""

        if "box_coords" in values.keys():
            box_coords = values["box_coords"]
            if box_coords is not None:
                size = [box_coords[1][d] - box_coords[0][d] for d in range(3)]
                center = [0.5 * (box_coords[1][d] + box_coords[0][d]) for d in range(3)]

                values["size"] = tuple(size)
                values["center"] = tuple(center)
                return values
            else:
                size = values["size"]
                center = values["center"]
                box_coords = (
                    tuple([center[d] - 0.5 * size[d] for d in range(3)]),
                    tuple([center[d] + 0.5 * size[d] for d in range(3)]),
                )

                values["box_coords"] = box_coords

        return values


class ConstantDoping(AbstractDopingBox):
    """
    This class sets constant doping :math:`N` in the specified box with a :parameter`size` and :parameter:`concentration`.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> constant_box1 = td.ConstantDoping(center=(0, 0, 0), size(2, 2, 2), concentration=1e18)
    >>> constant_box2 = td.ConstantDoping(box_coords=box_coords, concentration=1e18)
    """

    concentration: pd.NonNegativeFloat = pd.Field(
        default=0,
        title="Doping concentration density.",
        description="Doping concentration density in #/cm^3.",
    )


class GaussianDoping(AbstractDopingBox):
    """This class sets a gaussian doping in the specified box.

    Notes
    -----
    The Gaussian doping concentration :math:`N` is defined in relation to a reference
    concentration :math:`N_{\\text{ref}` as ``ref_con``,
    maximum target concentration :math:`N_{\\text{max}` as ``concentration``, a Gaussian ``width``,
    and a doping box ``size``. The concentration will decrease from :math:`N_{\\text{max}` to
    :math:`N_{\\text{ref}` in a length ``width`` following a Gaussian curve. By definition,
    all sides of the box will have concentration :math:`N_{\\text{ref}` (except the side specified
    as source) and the center of the box (``width`` away from the box sides) will have a concentration
    :math:`N_{\\text{max}`.


    TODO can we do better parameter names here more related to the equations? TODO how does the size get defined.

    .. math::

        N = \\{N_{\\text{max}}\\} \\exp \\left[
        - \\ln \\left( \\frac{\\{N_{\\text{max}}\\}}{\\{N_{\\text{ref}}\\}} \\right)
        \\left( \\frac{(x|y|z) - \\{(x|y|z)_{\\text{box}}\\}}{\\{\\text{width}\\}} \\right)^2
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
    ...     size(2, 2, 2),
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    >>> gaussian_box2 = td.GaussianDoping(
    ...     box_coords=box_coords,
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    """

    ref_con: pd.PositiveFloat = pd.Field(
        1e15,
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

    def get_contrib(self, coords: dict, meshgrid: bool = True):
        """Returns the contribution to the doping a the locations specified in coords"""

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

        if meshgrid:
            X, Y, Z = np.meshgrid(coords["x"], coords["y"], coords["z"], indexing="ij")
        else:
            X = coords["x"]
            Y = coords["y"]
            Z = coords["z"]

        bounds = [list(self.bounds[0]), list(self.bounds[1])]
        for d in range(3):
            if bounds[0][d] == bounds[1][d]:
                bounds[0][d] = -np.inf
                bounds[1][d] = np.inf
                if normal_axis is None:
                    normal_axis = d

        # let's assume some of these coordinates may lay outside the box
        indices_in_box = np.logical_and(X >= bounds[0][0], X <= bounds[1][0])
        indices_in_box = np.logical_and(indices_in_box, Y >= bounds[0][1])
        indices_in_box = np.logical_and(indices_in_box, Y <= bounds[1][1])
        indices_in_box = np.logical_and(indices_in_box, Z >= bounds[0][2])
        indices_in_box = np.logical_and(indices_in_box, Z <= bounds[1][2])

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

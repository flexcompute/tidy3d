"""File containing classes required for the setup of a DEVSIM case."""

import numpy as np
import pydantic.v1 as pd

from ...components.autograd.types import TracedSize
from ...components.geometry.base import Box
from ...constants import MICROMETER
from ..base import cached_property
from ..types import Union


class AbstractDopingBox(Box):
    """Derived class from Box which redefines size so that
    we have some default values"""

    size: TracedSize = pd.Field(
        (1, 1, 1),
        title="Size",
        description="Size in x, y, and z directions.",
        units=MICROMETER,
    )


class ConstantDoping(AbstractDopingBox):
    """This class sets constant doping in the specified box.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> constant_box1 = td.ConstantDoping(center=(0, 0, 0), size(2, 2, 2), concentration=1e18)
    >>> constant_box2 = td.ConstantDoping()
    >>> constant_box2 = constant_box2.from_bounds(rmin=box_coords[0], rmax=box_coords[1])
    >>> constant_box2 = constant_box2.updated_copy(concentration=1e8)
    """

    concentration: pd.NonNegativeFloat = pd.Field(
        default=0,
        title="Doping concentration density.",
        description="Doping concentration density in #/cm^3.",
    )


class GaussianDoping(AbstractDopingBox):
    """This class sets a gaussian doping in the specified box."""

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


DopingBoxType = Union[ConstantDoping, GaussianDoping]

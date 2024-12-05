"""File containing classes required for the setup of a DEVSIM case."""

from abc import ABC

import numpy as np
import pydantic.v1 as pd

from ..base import Tidy3dBaseModel, cached_property
from ..types import Bound, Union


class AbstractDopingBox(ABC, Tidy3dBaseModel):
    """"""

    coords: Bound = pd.Field(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        title="Coordinates of box vertices.",
        description="Tuple containing the minimum coordinates and the maximum "
        "coordinates of the vertices composing the doping box. The format is "
        "as follows: ((x_min, y_min, z_min), (x_max, y_max, z_max))",
    )

    @pd.validator("coords", always=True)
    def check_coords(cls, val):
        """Check that the minimum coordinates are indeed smaller than the maximum.
        The case where both min and max are equal is also considered.
        """
        raise_error = False
        for min, max in zip(val[0], val[1]):
            if min > max:
                raise_error = True

        if raise_error:
            raise pd.ValidationError(
                "Minimum coordinate must be lower than or equal to the maximum coordinate "
                "in each direction."
            )
        return val


class ConstantDoping(AbstractDopingBox):
    """This class sets constant doping in the specified box.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> constant_box = td.ConstantDoping(coords=box_coords, concentration=1e18)
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

    @cached_property
    def sigma(self):
        """The sigma parameter of the pseudo-gaussian"""
        return np.sqrt(-self.width * self.width / 2 / np.log(self.ref_con / self.concentration))


DopingBoxType = Union[ConstantDoping, GaussianDoping]

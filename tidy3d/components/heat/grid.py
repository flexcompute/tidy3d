"""Defines heat grid specifications"""
from __future__ import annotations

from typing import Union
import pydantic as pd

from ..base import Tidy3dBaseModel
from ...constants import MICROMETER
from ...exceptions import ValidationError


class UniformHeatGrid(Tidy3dBaseModel):

    """Uniform grid.

    Example
    -------
    >>> heat_grid = UniformHeatGrid(dl=0.1)
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    min_edges_per_circumference: pd.PositiveFloat = pd.Field(
        15,
        title="Minimum edges per circumference",
        description="Enforced minimum number of mesh segments per circumference of an object.",
    )

    min_edges_per_side: pd.PositiveFloat = pd.Field(
        2,
        title="Minimum edges per side",
        description="Enforced minimum number of mesh segments per any side of an object.",
    )


class DistanceHeatGrid(Tidy3dBaseModel):

    """Adaptive grid based on distance to material interfaces.

    Example
    -------
    >>> heat_grid = DistanceHeatGrid(
    ...     dl_interface=0.1,
    ...     dl_bulk=1,
    ...     distance_interface=0.3,
    ...     distance_bulk=2,
    ... )
    """

    dl_interface: pd.PositiveFloat = pd.Field(
        ...,
        title="Interface Grid Size",
        description="Grid size near material interfaces.",
        units=MICROMETER,
    )

    dl_bulk: pd.PositiveFloat = pd.Field(
        ...,
        title="Bulk Grid Size",
        description="Grid size away from material interfaces.",
        units=MICROMETER,
    )

    distance_interface: pd.PositiveFloat = pd.Field(
        ...,
        title="Interface Distance",
        description="Distance from interface within which ``dl_interface`` is enforced.",
        units=MICROMETER,
    )

    distance_bulk: pd.PositiveFloat = pd.Field(
        ...,
        title="Bulk Distance",
        description="Distance from interface outside of which ``dl_bulk`` is enforced.",
        units=MICROMETER,
    )

    sampling: pd.PositiveFloat = pd.Field(
        100,
        title="Surface Sampling",
        description="An internal advanced parameter that defines number of sampling points per "
        "surface when computing distance values.",
    )

    @pd.validator("distance_bulk", always=True)
    def names_exist_bcs(cls, val, values):
        """Error if distance_bulk is less than distance_interface"""
        distance_interface = values.get("distance_interface")
        if distance_interface > val:
            raise ValidationError("'distance_bulk' cannot be smaller than 'distance_interface'.")

        return val


HeatGridType = Union[UniformHeatGrid, DistanceHeatGrid]

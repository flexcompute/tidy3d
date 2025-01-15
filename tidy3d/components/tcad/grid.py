"""Defines heat grid specifications"""

from __future__ import annotations

from abc import ABC
from typing import Tuple, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, skip_if_fields_missing
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import ValidationError


class UnstructuredGrid(Tidy3dBaseModel, ABC):
    """Abstract unstructured grid."""

    relative_min_dl: pd.NonNegativeFloat = pd.Field(
        1e-3,
        title="Relative Mesh Size Limit",
        description="The minimal allowed mesh size relative to the largest dimension of the simulation domain."
        "Use ``relative_min_dl=0`` to remove this constraint.",
    )


class UniformUnstructuredGrid(UnstructuredGrid):
    """Uniform grid.

    Example
    -------
    >>> heat_grid = UniformUnstructuredGrid(dl=0.1)
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    min_edges_per_circumference: pd.PositiveFloat = pd.Field(
        15,
        title="Minimum Edges per Circumference",
        description="Enforced minimum number of mesh segments per circumference of an object. "
        "Applies to :class:`Cylinder` and :class:`Sphere`, for which the circumference "
        "is taken as 2 * pi * radius.",
    )

    min_edges_per_side: pd.PositiveFloat = pd.Field(
        2,
        title="Minimum Edges per Side",
        description="Enforced minimum number of mesh segments per any side of an object.",
    )

    non_refined_structures: Tuple[str, ...] = pd.Field(
        (),
        title="Structures Without Refinement",
        description="List of structures for which ``min_edges_per_circumference`` and "
        "``min_edges_per_side`` will not be enforced. The original ``dl`` is used instead.",
    )


class DistanceUnstructuredGrid(UnstructuredGrid):
    """Adaptive grid based on distance to material interfaces. Currently not recommended for larger
    simulations.

    Example
    -------
    >>> heat_grid = DistanceUnstructuredGrid(
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

    distance_interface: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Interface Distance",
        description="Distance from interface within which ``dl_interface`` is enforced."
        "Typically the same as ``dl_interface`` or its multiple.",
        units=MICROMETER,
    )

    distance_bulk: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Bulk Distance",
        description="Distance from interface outside of which ``dl_bulk`` is enforced."
        "Typically twice of ``dl_bulk`` or its multiple. Use larger values for a smoother "
        "transition from ``dl_interface`` to ``dl_bulk``.",
        units=MICROMETER,
    )

    sampling: pd.PositiveFloat = pd.Field(
        100,
        title="Surface Sampling",
        description="An internal advanced parameter that defines number of sampling points per "
        "surface when computing distance values.",
    )

    uniform_grid_mediums: Tuple[str, ...] = pd.Field(
        (),
        title="Mediums With Uniform Refinement",
        description="List of mediums for which ``dl_interface`` will be enforced everywhere "
        "in the volume.",
    )

    non_refined_structures: Tuple[str, ...] = pd.Field(
        (),
        title="Structures Without Refinement",
        description="List of structures for which ``dl_interface`` will not be enforced. "
        "``dl_bulk`` is used instead.",
    )

    @pd.validator("distance_bulk", always=True)
    @skip_if_fields_missing(["distance_interface"])
    def names_exist_bcs(cls, val, values):
        """Error if distance_bulk is less than distance_interface"""
        distance_interface = values.get("distance_interface")
        if distance_interface > val:
            raise ValidationError("'distance_bulk' cannot be smaller than 'distance_interface'.")

        return val


UnstructuredGridType = Union[UniformUnstructuredGrid, DistanceUnstructuredGrid]

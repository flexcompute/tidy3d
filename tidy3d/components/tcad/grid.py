"""Defines heat grid specifications"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import Coordinate
from tidy3d.components.types.base import discriminated_union
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self


class UnstructuredGrid(Tidy3dBaseModel, ABC):
    """Abstract unstructured grid."""

    relative_min_dl: NonNegativeFloat = Field(
        1e-3,
        title="Relative Mesh Size Limit",
        description="The minimal allowed mesh size relative to the largest dimension of the simulation domain."
        "Use ``relative_min_dl=0`` to remove this constraint.",
    )

    remove_fragments: bool = Field(
        False,
        title="Remove Fragments",
        description="Whether to remove fragments before meshing. This is useful when overlapping structures generate internal boundaries that can lead to very small cell volumes.",
    )

    @property
    @abstractmethod
    def min_mesh_size(self) -> float:
        """Minimum mesh size used by this grid specification."""


class UniformUnstructuredGrid(UnstructuredGrid):
    """Uniform grid.

    Example
    -------
    >>> heat_grid = UniformUnstructuredGrid(dl=0.1)
    """

    dl: PositiveFloat = Field(
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        json_schema_extra={"units": MICROMETER},
    )

    min_edges_per_circumference: PositiveFloat = Field(
        15,
        title="Minimum Edges per Circumference",
        description="Enforced minimum number of mesh segments per circumference of an object. "
        "Applies to :class:`Cylinder` and :class:`Sphere`, for which the circumference "
        "is taken as 2 * pi * radius.",
    )

    min_edges_per_side: PositiveFloat = Field(
        2,
        title="Minimum Edges per Side",
        description="Enforced minimum number of mesh segments per any side of an object.",
    )

    non_refined_structures: tuple[str, ...] = Field(
        (),
        title="Structures Without Refinement",
        description="List of structures for which ``min_edges_per_circumference`` and "
        "``min_edges_per_side`` will not be enforced. The original ``dl`` is used instead.",
    )

    @property
    def min_mesh_size(self) -> float:
        """Minimum mesh size used by this grid specification."""
        return self.dl


class GridRefinementRegion(Box):
    """Refinement region for the unstructured mesh. The cell size is enforced to be constant inside the region.
    The cell size outside of the region depends on the distance from the region."""

    dl_internal: PositiveFloat = Field(
        title="Internal mesh cell size",
        description="Mesh cell size inside the refinement region",
        json_schema_extra={"units": MICROMETER},
    )

    transition_thickness: NonNegativeFloat = Field(
        title="Interface Distance",
        description="Thickness of a transition layer outside the box where the mesh cell size changes from the"
        "internal size to the external one.",
        json_schema_extra={"units": MICROMETER},
    )


class GridRefinementLine(Tidy3dBaseModel, ABC):
    """Refinement line for the unstructured mesh. The cell size depends on the distance from the line."""

    r1: Coordinate = Field(
        title="Start point of the line",
        description="Start point of the line in x, y, and z.",
        json_schema_extra={"units": MICROMETER},
    )

    r2: Coordinate = Field(
        title="End point of the line",
        description="End point of the line in x, y, and z.",
        json_schema_extra={"units": MICROMETER},
    )

    @field_validator("r1", "r2")
    @classmethod
    def _not_inf(cls, val: Coordinate) -> Coordinate:
        """Make sure the point is not infinitiy."""
        if any(np.isinf(v) for v in val):
            raise ValidationError("Point can not contain 'td.inf' terms.")
        return val

    dl_near: PositiveFloat = Field(
        title="Mesh cell size near the line",
        description="Mesh cell size near the line",
        json_schema_extra={"units": MICROMETER},
    )

    distance_near: NonNegativeFloat = Field(
        title="Near distance",
        description="Distance from the line within which ``dl_near`` is enforced."
        "Typically the same as ``dl_near`` or its multiple.",
        json_schema_extra={"units": MICROMETER},
    )

    distance_bulk: NonNegativeFloat = Field(
        title="Bulk distance",
        description="Distance from the line outside of which ``dl_bulk`` is enforced."
        "Typically twice of ``dl_bulk`` or its multiple. Use larger values for a smoother "
        "transition from ``dl_near`` to ``dl_bulk``.",
        json_schema_extra={"units": MICROMETER},
    )

    @model_validator(mode="after")
    def names_exist_bcs(self) -> Self:
        """Error if distance_bulk is less than distance_near"""
        if self.distance_near > self.distance_bulk:
            raise ValidationError("'distance_bulk' cannot be smaller than 'distance_near'.")

        return self


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

    dl_interface: PositiveFloat = Field(
        title="Interface Grid Size",
        description="Grid size near material interfaces.",
        json_schema_extra={"units": MICROMETER},
    )

    dl_bulk: PositiveFloat = Field(
        title="Bulk Grid Size",
        description="Grid size away from material interfaces.",
        json_schema_extra={"units": MICROMETER},
    )

    distance_interface: NonNegativeFloat = Field(
        title="Interface Distance",
        description="Distance from interface within which ``dl_interface`` is enforced."
        "Typically the same as ``dl_interface`` or its multiple.",
        json_schema_extra={"units": MICROMETER},
    )

    distance_bulk: NonNegativeFloat = Field(
        title="Bulk Distance",
        description="Distance from interface outside of which ``dl_bulk`` is enforced."
        "Typically twice of ``dl_bulk`` or its multiple. Use larger values for a smoother "
        "transition from ``dl_interface`` to ``dl_bulk``.",
        json_schema_extra={"units": MICROMETER},
    )

    sampling: PositiveFloat = Field(
        100,
        title="Surface Sampling",
        description="An internal advanced parameter that defines number of sampling points per "
        "surface when computing distance values.",
    )

    uniform_grid_mediums: tuple[str, ...] = Field(
        (),
        title="Mediums With Uniform Refinement",
        description="List of mediums for which ``dl_interface`` will be enforced everywhere "
        "in the volume.",
    )

    non_refined_structures: tuple[str, ...] = Field(
        (),
        title="Structures Without Refinement",
        description="List of structures for which ``dl_interface`` will not be enforced. "
        "``dl_bulk`` is used instead.",
    )

    mesh_refinements: tuple[
        discriminated_union(Union[GridRefinementRegion, GridRefinementLine]), ...
    ] = Field(
        (),
        title="Mesh refinement structures",
        description="List of regions/lines for which the mesh refinement will be applied",
    )

    @model_validator(mode="after")
    def names_exist_bcs(self) -> Self:
        """Error if distance_bulk is less than distance_interface"""
        if self.distance_interface > self.distance_bulk:
            raise ValidationError("'distance_bulk' cannot be smaller than 'distance_interface'.")

        return self

    @property
    def min_mesh_size(self) -> float:
        """Minimum mesh size used by this grid specification."""
        dl_array = [self.dl_interface]
        for ref in self.mesh_refinements:
            if isinstance(ref, GridRefinementRegion):
                dl_array.append(ref.dl_internal)
            elif isinstance(ref, GridRefinementLine):
                dl_array.append(ref.dl_near)
        return min(dl_array)


UnstructuredGridType = Union[UniformUnstructuredGrid, DistanceUnstructuredGrid]

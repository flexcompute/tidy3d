"""Defines heat grid specifications"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import Coordinate
from tidy3d.components.types.base import discriminated_union
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self


REFINEMENT_LINE_TOLERANCE = 1e-6


class UnstructuredGrid(Tidy3dBaseModel, ABC):
    """Abstract unstructured grid."""

    relative_min_dl: NonNegativeFloat = Field(
        1e-3,
        title="Relative Mesh Size Limit",
        description="The minimal allowed mesh size relative to the largest dimension of the simulation domain."
        "Use ``relative_min_dl=0`` to remove this constraint.",
    )

    geometry_tolerance: PositiveFloat = Field(
        1e-6,
        title="Geometry Tolerance",
        description="Absolute distance below which coincident geometric entities are fused when "
        "building the mesh. Increase this if abutting structures with finely tessellated (e.g. "
        "curved) boundaries fail to merge into a single conformal interface, which can leave "
        "duplicated internal surfaces and degenerate elements. Keep it well below the smallest "
        "geometric feature and the target mesh size: too large a value snaps together unrelated "
        "vertices and corrupts the mesh. Refinement lines are additionally subject to a built-in "
        f"{REFINEMENT_LINE_TOLERANCE:.0e} um minimum length, so setting this knob below that value "
        "does not relax the line-length requirement.",
        json_schema_extra={"units": MICROMETER},
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
    >>> heat_grid = UniformUnstructuredGrid(
    ...     dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2
    ... )
    """

    dl: PositiveFloat = Field(
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        json_schema_extra={"units": MICROMETER},
    )

    min_edges_per_circumference: NonNegativeFloat = Field(
        15,
        title="Minimum Edges per Circumference",
        description="Enforced minimum number of mesh segments per circumference of an object. "
        "Applies to :class:`Cylinder` and :class:`Sphere`, for which the circumference "
        "is taken as 2 * pi * radius. Set to ``0`` to skip this sizing contribution "
        "entirely (curvature-based local refinement is not applied).",
    )

    min_edges_per_side: NonNegativeFloat = Field(
        2,
        title="Minimum Edges per Side",
        description="Enforced minimum number of mesh segments per any side of an object. "
        "Set to ``0`` to skip this sizing contribution entirely (side-length-based local "
        "refinement is not applied).",
    )

    non_refined_structures: tuple[str, ...] = Field(
        (),
        title="Structures Without Refinement",
        description="List of structures for which ``min_edges_per_circumference`` and "
        "``min_edges_per_side`` will not be enforced. The original ``dl`` is used instead.",
    )

    @model_validator(mode="after")
    def _warn_default_min_edges(self) -> Self:
        """Warn when ``min_edges_per_circumference`` / ``min_edges_per_side`` rely on defaults."""
        unset = {"min_edges_per_circumference", "min_edges_per_side"} - self.model_fields_set
        if unset:
            log.warning(
                f"Field(s) {sorted(unset)} on 'UniformUnstructuredGrid' are using the "
                "current defaults; these defaults will change to 0 in the next release, "
                "which disables curvature- and side-length-based local mesh refinement. "
                "Set them explicitly to preserve the current behavior."
            )
        return self

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

    @model_validator(mode="after")
    def _validate_supported_region_shape(self) -> Self:
        """Allow only volumetric or planar refinement regions."""
        if self.size.count(0.0) > 1:
            self._raise_validation_error_at_loc(
                ValidationError(
                    "Refinement region must be volumetric or planar; 'size' cannot have more than one zero-sized dimension."
                ),
                "size",
            )

        return self


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
            self._raise_validation_error_at_loc(
                ValidationError("'distance_bulk' cannot be smaller than 'distance_near'."),
                "distance_bulk",
            )
        line_length = float(np.linalg.norm(np.asarray(self.r2) - np.asarray(self.r1)))
        if line_length <= REFINEMENT_LINE_TOLERANCE:
            self._raise_validation_error_at_loc(
                ValidationError(
                    f"Refinement line endpoints are too close; the line length must be greater than "
                    f"{REFINEMENT_LINE_TOLERANCE:.1e} um."
                ),
                "r2",
            )

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
        description="List of structures whose owned interfaces do not enforce "
        "``dl_interface``. For interfaces shared by multiple structures, ownership follows "
        "structure precedence: the last matching structure in the simulation's structure list "
        "decides whether the interface is refined. Structures in this list also do not "
        "receive volume refinement from ``uniform_grid_mediums``.",
    )

    mesh_refinements: tuple[discriminated_union(GridRefinementRegion | GridRefinementLine), ...] = (
        Field(
            (),
            title="Mesh refinement structures",
            description="List of regions/lines for which the mesh refinement will be applied",
        )
    )

    @model_validator(mode="after")
    def names_exist_bcs(self) -> Self:
        """Error if distance_bulk is less than distance_interface"""
        if self.distance_interface > self.distance_bulk:
            self._raise_validation_error_at_loc(
                ValidationError("'distance_bulk' cannot be smaller than 'distance_interface'."),
                "distance_bulk",
            )

        # A refinement line at or below the fusion tolerance is collapsed during meshing;
        # reject it at setup time rather than as a meshing-time failure.
        for ind, ref in enumerate(self.mesh_refinements):
            if isinstance(ref, GridRefinementLine):
                line_length = float(np.linalg.norm(np.asarray(ref.r2) - np.asarray(ref.r1)))
                if line_length <= self.geometry_tolerance:
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"Refinement line length ({line_length:.1e} um) must be greater than "
                            f"'geometry_tolerance' ({self.geometry_tolerance:.1e} um); shorter lines "
                            "are collapsed when coincident geometry is fused during meshing."
                        ),
                        "mesh_refinements",
                        ind,
                    )

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


UnstructuredGridType = UniformUnstructuredGrid | DistanceUnstructuredGrid

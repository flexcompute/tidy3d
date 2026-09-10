"""Defines heat and charge grid specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.bc_placement import (
    MediumMediumInterface,
    StructureBoundary,
    StructureStructureInterface,
)
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
        default=1e-3,
        title="Relative Mesh Size Limit",
        description="The minimal allowed mesh size relative to the largest dimension of the simulation domain."
        "Use ``relative_min_dl=0`` to remove this constraint.",
    )

    geometry_tolerance: PositiveFloat = Field(
        default=1e-6,
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
        default=False,
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
        default=15,
        title="Minimum Edges per Circumference",
        description="Enforced minimum number of mesh segments per circumference of an object. "
        "Applies to :class:`Cylinder` and :class:`Sphere`, for which the circumference "
        "is taken as 2 * pi * radius. Set to ``0`` to skip this sizing contribution "
        "entirely (curvature-based local refinement is not applied).",
    )

    min_edges_per_side: NonNegativeFloat = Field(
        default=2,
        title="Minimum Edges per Side",
        description="Enforced minimum number of mesh segments per any side of an object. "
        "Set to ``0`` to skip this sizing contribution entirely (side-length-based local "
        "refinement is not applied).",
    )

    non_refined_structures: tuple[str, ...] = Field(
        default=(),
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


class _GridRefinementRegionBase(Box):
    """Shared geometry and validation for mesh-refinement regions."""

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


class GridRefinementRegion(_GridRefinementRegionBase):
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


class RelativeGridRefinementRegion(_GridRefinementRegionBase):
    """Refinement region sized relative to an enclosing ``AutoUnstructuredGrid``.

    Notes
    -----
    This refinement type is supported only by ``AutoUnstructuredGrid``.

    Example
    -------
    >>> region = RelativeGridRefinementRegion(
    ...     center=(0, 0, 0),
    ...     size=(1, 1, 0),
    ...     dl_ratio=0.25,
    ...     transition_thickness_ratio=4,
    ... )
    """

    dl_ratio: PositiveFloat = Field(
        title="Internal Mesh Ratio",
        description="Mesh size inside the refinement region, relative to ``dl_reference``.",
    )

    transition_thickness_ratio: NonNegativeFloat = Field(
        title="Transition Thickness Ratio",
        description="Transition thickness outside the region, relative to ``dl_reference``.",
    )


class _GridRefinementLineBase(Tidy3dBaseModel, ABC):
    """Shared geometry and validation for mesh-refinement lines."""

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

    @model_validator(mode="after")
    def _validate_line_length(self) -> Self:
        """Reject refinement lines that collapse at the built-in tolerance."""
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


class GridRefinementLine(_GridRefinementLineBase):
    """Refinement line for the unstructured mesh. The cell size depends on the distance from the line."""

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
    def _validate_distance_order(self) -> Self:
        """Ensure that the far transition distance is not inside the near distance."""
        if self.distance_near > self.distance_bulk:
            self._raise_validation_error_at_loc(
                ValidationError("'distance_bulk' cannot be smaller than 'distance_near'."),
                "distance_bulk",
            )

        return self


class RelativeGridRefinementLine(_GridRefinementLineBase):
    """Line refinement sized relative to an enclosing ``AutoUnstructuredGrid``.

    Notes
    -----
    This refinement type is supported only by ``AutoUnstructuredGrid``.

    Example
    -------
    >>> line = RelativeGridRefinementLine(
    ...     r1=(-0.5, 0, 0),
    ...     r2=(0.5, 0, 0),
    ...     dl_ratio=0.25,
    ...     distance_near_ratio=2,
    ...     distance_bulk_ratio=4,
    ... )
    """

    dl_ratio: PositiveFloat = Field(
        title="Near Mesh Ratio",
        description="Mesh size near the line, relative to ``dl_reference``.",
    )

    distance_near_ratio: NonNegativeFloat = Field(
        default=2.0,
        title="Near Distance Ratio",
        description="Near transition distance relative to the local line mesh size.",
    )

    distance_bulk_ratio: NonNegativeFloat = Field(
        default=4.0,
        title="Bulk Distance Ratio",
        description="Far transition distance relative to the enclosing grid's bulk mesh size.",
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
        default=100,
        title="Surface Sampling",
        description="An internal advanced parameter that defines number of sampling points per "
        "surface when computing distance values.",
    )

    uniform_grid_mediums: tuple[str, ...] = Field(
        default=(),
        title="Mediums With Uniform Refinement",
        description="List of mediums for which ``dl_interface`` will be enforced everywhere "
        "in the volume.",
    )

    non_refined_structures: tuple[str, ...] = Field(
        default=(),
        title="Structures Without Refinement",
        description="List of structures whose owned interfaces do not enforce "
        "``dl_interface``. For interfaces shared by multiple structures, ownership follows "
        "structure precedence: the last matching structure in the simulation's structure list "
        "decides whether the interface is refined. Structures in this list also do not "
        "receive volume refinement from ``uniform_grid_mediums``.",
    )

    mesh_refinements: tuple[discriminated_union(GridRefinementRegion | GridRefinementLine), ...] = (
        Field(
            default=(),
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


InterfaceRefinementSelectionType = discriminated_union(
    StructureBoundary | StructureStructureInterface | MediumMediumInterface
)


class InterfaceRefinementSpec(Tidy3dBaseModel):
    """Automatic refinement rule for a selected set of interfaces."""

    selection: InterfaceRefinementSelectionType = Field(
        ...,
        title="Interface Selection",
        description="Selector identifying the interfaces targeted by this refinement rule.",
    )

    dl_interface_ratio: PositiveFloat = Field(
        title="Interface Mesh Ratio",
        description="Target mesh size at the selected interface, relative to ``dl_reference``.",
    )

    distance_interface_ratio: NonNegativeFloat = Field(
        default=2.0,
        title="Interface Distance Ratio",
        description="Distance from the selected interface within which the target mesh size is enforced, relative to the local interface mesh size.",
    )

    distance_bulk_ratio: NonNegativeFloat = Field(
        default=4.0,
        title="Bulk Distance Ratio",
        description="Distance from the selected interface outside of which the bulk mesh size is enforced, relative to the grid bulk mesh size.",
    )


class AutoUnstructuredGrid(UnstructuredGrid):
    """Adaptive unstructured grid with ratio-based refinement controls.

    Notes
    -----
    ``AutoUnstructuredGrid`` supports:

    - baseline refinement at interfaces separating regions with different active
      heat or charge properties, including excluded-conductor charge contacts,
    - custom ``interface_refinements`` may use :class:`StructureBoundary`,
      :class:`StructureStructureInterface`, and
      :class:`MediumMediumInterface`.

    Adjacent structures with identical active material, source, and boundary
    properties may be treated as one region and do not receive baseline interface
    refinement. A source or structure boundary condition can make otherwise
    identical structures distinct for the active problem. An abrupt p-n doping
    change contained within one :class:`SemiconductorMedium` is not an interface.
    When it is natural to model the two sides as separate
    :class:`SemiconductorMedium` instances, their active interface receives
    baseline refinement automatically. Otherwise, use a
    :class:`RelativeGridRefinementRegion` or :class:`RelativeGridRefinementLine`
    to refine the junction without changing the material model.

    Explicit structure-based selectors may target a fragmented boundary even when
    its two sides are treated as one region by the baseline rule. A
    :class:`MediumMediumInterface` requires two distinct named mediums.

    A :class:`StructureBoundary` selection follows structure precedence. A
    lower-precedence charge conductor also retains its contact interface when it
    is excluded from the active electric domain.

    Example
    -------
    >>> heat_grid = AutoUnstructuredGrid(
    ...     dl_reference=0.1,
    ... )
    """

    dl_reference: PositiveFloat = Field(
        title="Reference Mesh Size",
        description="Reference mesh size anchoring all ratio-based refinement controls; under "
        "the default ratios, this is the baseline interface mesh size.",
        json_schema_extra={"units": MICROMETER},
    )

    dl_bulk_ratio: PositiveFloat = Field(
        default=4.0,
        title="Bulk Mesh Ratio",
        description="Bulk/background mesh size relative to ``dl_reference``.",
    )

    dl_interface_ratio: PositiveFloat = Field(
        default=1.0,
        title="Interface Mesh Ratio",
        description="Baseline interface mesh size relative to ``dl_reference``.",
    )

    distance_interface_ratio: NonNegativeFloat = Field(
        default=2.0,
        title="Interface Distance Ratio",
        description="Ratio multiplying ``dl_reference * dl_interface_ratio`` to set the "
        "distance from interfaces within which the interface mesh size is enforced.",
    )

    distance_bulk_ratio: NonNegativeFloat = Field(
        default=4.0,
        title="Bulk Distance Ratio",
        description="Ratio multiplying ``dl_reference * dl_bulk_ratio`` to set the distance "
        "from interfaces outside of which the bulk mesh size is enforced.",
    )

    sampling: PositiveFloat = Field(
        default=100,
        title="Surface Sampling",
        description="An internal advanced parameter that defines number of sampling points per "
        "surface when computing distance values.",
    )

    uniform_grid_mediums: tuple[str, ...] = Field(
        default=(),
        title="Mediums With Uniform Refinement",
        description="List of mediums for which the baseline interface mesh size will be "
        "enforced everywhere in the volume.",
    )

    non_refined_structures: tuple[str, ...] = Field(
        default=(),
        title="Structures Without Automatic Refinement",
        description="List of structures whose owned interfaces do not enforce automatic "
        "interface refinement. For interfaces shared by multiple structures, ownership follows "
        "structure precedence: the last matching structure in the simulation's structure list "
        "decides whether the interface is refined. Structures in this list also do not "
        "receive volume refinement from ``uniform_grid_mediums``.",
    )

    interface_refinements: tuple[InterfaceRefinementSpec, ...] = Field(
        default=(),
        title="Interface Refinements",
        description="Additional targeted automatic interface refinement rules.",
    )

    mesh_refinements: tuple[
        discriminated_union(
            GridRefinementRegion
            | GridRefinementLine
            | RelativeGridRefinementRegion
            | RelativeGridRefinementLine
        ),
        ...,
    ] = Field(
        default=(),
        title="Mesh Refinement Structures",
        description="Absolute or reference-relative regions and lines for local mesh refinement.",
    )

    @property
    def dl_bulk(self) -> float:
        """Bulk mesh size for the supported baseline refinement path."""
        return self.dl_reference * self.dl_bulk_ratio

    @property
    def dl_interface(self) -> float:
        """Baseline interface mesh size for the supported baseline refinement path."""
        return self.dl_reference * self.dl_interface_ratio

    @property
    def distance_interface(self) -> float:
        """Near-distance threshold for the supported baseline refinement path."""
        return self.dl_interface * self.distance_interface_ratio

    @property
    def distance_bulk(self) -> float:
        """Far-distance threshold for the supported baseline refinement path."""
        return self.dl_bulk * self.distance_bulk_ratio

    @model_validator(mode="after")
    def _validate_supported_interface_selectors(self) -> Self:
        """Restrict targeted interface refinements to supported selector types."""
        if self.distance_interface > self.distance_bulk:
            self._raise_validation_error_at_loc(
                "'distance_bulk' cannot be smaller than 'distance_interface'.",
                "distance_bulk_ratio",
            )

        supported_targeted_selectors = (
            StructureBoundary,
            StructureStructureInterface,
            MediumMediumInterface,
        )
        for ref_ind, refinement in enumerate(self.interface_refinements):
            if not isinstance(refinement.selection, supported_targeted_selectors):
                self._raise_validation_error_at_loc(
                    "'AutoUnstructuredGrid' only supports targeted interface refinements with "
                    "selections "
                    "'StructureBoundary()', 'StructureStructureInterface()', "
                    "and 'MediumMediumInterface()'.",
                    "interface_refinements",
                    ref_ind,
                    "selection",
                )
            if (
                self.dl_reference
                * refinement.dl_interface_ratio
                * refinement.distance_interface_ratio
                > self.dl_bulk * refinement.distance_bulk_ratio
            ):
                self._raise_validation_error_at_loc(
                    "'distance_bulk' cannot be smaller than 'distance_interface'.",
                    "interface_refinements",
                    ref_ind,
                    "distance_bulk_ratio",
                )

        for ref_ind, refinement in enumerate(self.mesh_refinements):
            if isinstance(refinement, (GridRefinementLine, RelativeGridRefinementLine)):
                line_length = float(
                    np.linalg.norm(np.asarray(refinement.r2) - np.asarray(refinement.r1))
                )
                if line_length <= self.geometry_tolerance:
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"Refinement line length ({line_length:.1e} um) must be greater than "
                            f"'geometry_tolerance' ({self.geometry_tolerance:.1e} um); shorter lines "
                            "are collapsed when coincident geometry is fused during meshing."
                        ),
                        "mesh_refinements",
                        ref_ind,
                    )

            if isinstance(refinement, RelativeGridRefinementLine):
                distance_near = (
                    self.dl_reference * refinement.dl_ratio * refinement.distance_near_ratio
                )
                distance_bulk = self.dl_bulk * refinement.distance_bulk_ratio
                if distance_near > distance_bulk:
                    self._raise_validation_error_at_loc(
                        "'distance_bulk' cannot be smaller than 'distance_near'.",
                        "mesh_refinements",
                        ref_ind,
                        "distance_bulk_ratio",
                    )

        return self

    def _resolve_mesh_refinement(
        self,
        refinement: GridRefinementRegion
        | GridRefinementLine
        | RelativeGridRefinementRegion
        | RelativeGridRefinementLine,
    ) -> GridRefinementRegion | GridRefinementLine:
        """Resolve a manual refinement to the legacy absolute sizing contract."""
        if isinstance(refinement, RelativeGridRefinementRegion):
            return GridRefinementRegion(
                center=refinement.center,
                size=refinement.size,
                dl_internal=self.dl_reference * refinement.dl_ratio,
                transition_thickness=(self.dl_reference * refinement.transition_thickness_ratio),
                attrs=dict(refinement.attrs),
            )
        if isinstance(refinement, RelativeGridRefinementLine):
            local_dl = self.dl_reference * refinement.dl_ratio
            return GridRefinementLine(
                r1=refinement.r1,
                r2=refinement.r2,
                dl_near=local_dl,
                distance_near=local_dl * refinement.distance_near_ratio,
                distance_bulk=self.dl_bulk * refinement.distance_bulk_ratio,
                attrs=dict(refinement.attrs),
            )
        return refinement

    @property
    def min_mesh_size(self) -> float:
        """Minimum mesh size used by this grid specification."""
        dl_array = [self._automatic_min_mesh_size]
        for refinement in self.interface_refinements:
            dl_array.append(self.dl_reference * refinement.dl_interface_ratio)
        for ref in self.mesh_refinements:
            resolved_refinement = self._resolve_mesh_refinement(ref)
            if isinstance(resolved_refinement, GridRefinementRegion):
                dl_array.append(resolved_refinement.dl_internal)
            elif isinstance(resolved_refinement, GridRefinementLine):
                dl_array.append(resolved_refinement.dl_near)
        return min(dl_array)

    @property
    def _automatic_min_mesh_size(self) -> float:
        """Minimum mesh size from baseline automatic interface controls."""
        return min(self.dl_bulk, self.dl_interface)

    def to_distance_grid(self) -> DistanceUnstructuredGrid:
        """Map supported sizing controls to a distance grid.

        This maps grid parameters, not scene-level meshing behavior. Baseline auto
        refinement can include explicit interface-boundary-condition surfaces that a
        distance grid does not refine.
        """
        if len(self.interface_refinements) > 0:
            raise ValidationError(
                "'interface_refinements' cannot be represented by 'DistanceUnstructuredGrid'."
            )

        return DistanceUnstructuredGrid(
            relative_min_dl=self.relative_min_dl,
            geometry_tolerance=self.geometry_tolerance,
            remove_fragments=self.remove_fragments,
            dl_interface=self.dl_interface,
            dl_bulk=self.dl_bulk,
            distance_interface=self.distance_interface,
            distance_bulk=self.distance_bulk,
            sampling=self.sampling,
            uniform_grid_mediums=self.uniform_grid_mediums,
            non_refined_structures=self.non_refined_structures,
            mesh_refinements=tuple(
                self._resolve_mesh_refinement(refinement) for refinement in self.mesh_refinements
            ),
            attrs=dict(self.attrs),
        )

    @classmethod
    def from_distance_grid(cls, grid: DistanceUnstructuredGrid) -> Self:
        """Build an adaptive grid from legacy distance-grid sizing controls.

        This helper maps sizing parameters. Baseline auto refinement can additionally
        include active interfaces with explicit boundary conditions that the legacy
        distance grid did not refine.
        """
        return cls(
            relative_min_dl=grid.relative_min_dl,
            geometry_tolerance=grid.geometry_tolerance,
            remove_fragments=grid.remove_fragments,
            dl_reference=grid.dl_interface,
            dl_bulk_ratio=grid.dl_bulk / grid.dl_interface,
            sampling=grid.sampling,
            uniform_grid_mediums=grid.uniform_grid_mediums,
            non_refined_structures=grid.non_refined_structures,
            dl_interface_ratio=1.0,
            distance_interface_ratio=grid.distance_interface / grid.dl_interface,
            distance_bulk_ratio=grid.distance_bulk / grid.dl_bulk,
            mesh_refinements=grid.mesh_refinements,
            attrs=dict(grid.attrs),
        )


UnstructuredGridType = UniformUnstructuredGrid | DistanceUnstructuredGrid | AutoUnstructuredGrid

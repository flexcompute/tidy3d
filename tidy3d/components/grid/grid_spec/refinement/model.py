"""Canonical grid-refinement models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.corner_finder import CornerFinderSpec
from tidy3d.components.grid.grid_spec.constants import DEFAULT_REFINEMENT_FACTOR
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.components.types import Axis
from tidy3d.constants import MICROMETER, inf
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types import Coordinate, CoordinateOptional


class GridRefinement(Tidy3dBaseModel):
    """Specification for local mesh refinement that defines the grid step size and the number of grid
    cells in the refinement region.

    Note
    ----

    If both `refinement_factor` and `dl` are defined, the grid step size is upper bounded by the smaller value of the two.
    If neither is defined, default `refinement_factor=2` is applied.


    Example
    -------
    >>> grid_refine = GridRefinement(refinement_factor = 2, num_cells = 7)

    """

    refinement_factor: PositiveFloat | None = Field(
        default=None,
        title="Mesh Refinement Factor",
        description="Refine grid step size in vacuum by this factor.",
    )

    dl: PositiveFloat | None = Field(
        default=None,
        title="Grid Size",
        description="Grid step size in the refined region.",
        json_schema_extra={"units": MICROMETER},
    )

    num_cells: PositiveInt = Field(
        default=3,
        title="Number of Refined Grid Cells",
        description="Sets the extent of the refinement region in units of the refined grid size "
        "``dl``. When refining around a point, it is the width of the refined box "
        "(``num_cells * dl``, centered on the point); when refining a region of finite extent, it "
        "is the margin (``num_cells * dl`` per axis) by which the region's bounding box is grown.",
    )

    @property
    def _refinement_factor(self) -> PositiveFloat:
        """Refinement factor applied internally."""
        if self.refinement_factor is None and self.dl is None:
            return DEFAULT_REFINEMENT_FACTOR
        return self.refinement_factor

    def _grid_size(self, grid_size_in_vacuum: float) -> float:
        """Grid step size in the refinement region.

        Parameters
        ----------
        grid_size_in_vacuum : float
            Grid step size in vaccum.

        Returns
        -------
        float
            Grid step size in the refinement region.
        """

        dl = inf
        if self._refinement_factor is not None:
            dl = min(dl, grid_size_in_vacuum / self._refinement_factor)
        if self.dl is not None:
            dl = min(dl, self.dl)
        return dl

    def override_structure(
        self,
        center: CoordinateOptional,
        size: Coordinate,
        grid_size_in_vacuum: float,
        drop_outside_sim: bool,
    ) -> MeshOverrideStructure:
        """Generate override structure for mesh refinement around a point or a region.

        Each refined axis is grown to ``size + num_cells * dl`` and centered on ``center``.

        Parameters
        ----------
        center : CoordinateOptional
            Center of the override structure. A ``None`` coordinate along an axis means refinement
            is not applied along that axis.
        size : Coordinate
            Extent of the region to refine along each axis (``0`` for a point). Ignored along axes
            where ``center`` is ``None``.
        grid_size_in_vacuum : float
            Grid step size in vaccum.
        drop_outside_sim : bool
            Drop override structures outside simulation domain.

        Returns
        -------
        MeshOverrideStructure
            Unshadowed override structures for mesh refinement. Along an axis where refinement is
            not applied, the override geometry has size=inf and dl=None.
        """

        dl = self._grid_size(grid_size_in_vacuum)
        # override step size list
        dl_list = [None if axis_c is None else dl for axis_c in center]
        # override structure
        center_geo = [0 if axis_c is None else axis_c for axis_c in center]
        size_geo = [
            inf if axis_c is None else axis_s + dl * self.num_cells
            for axis_c, axis_s in zip(center, size)
        ]
        return MeshOverrideStructure(
            geometry=Box(center=center_geo, size=size_geo),
            dl=dl_list,
            shadow=False,
            drop_outside_sim=drop_outside_sim,
            priority=-1,
        )


class LayerRefinementSpec(Box):
    """Specification for automatic mesh refinement and snapping in layered structures. Structure corners
    on the cross section perpendicular to layer thickness direction can be automatically identified. Subsequently,
    mesh is snapped and refined around the corners. Mesh can also be refined and snapped around the bounds along
    the layer thickness direction.

    Note
    ----

    Corner detection is performed on a 2D plane sitting in the middle of the layer. If the layer is finite
    along inplane axes, corners outside the bounds are discarded.

    Note
    ----

    This class only takes effect when :class:`.AutoGrid` is applied.

    Example
    -------
    >>> layer_spec = LayerRefinementSpec(axis=2, center=(0,0,0), size=(2, 3, 1))

    """

    axis: Axis = Field(
        title="Axis",
        description="Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z).",
    )

    min_steps_along_axis: PositiveFloat | None = Field(
        default=None,
        title="Minimal Number Of Steps Along Axis",
        description="If not ``None`` and the thickness of the layer is nonzero, set minimal "
        "number of steps discretizing the layer thickness.",
    )

    bounds_refinement: GridRefinement | None = Field(
        default=None,
        title="Mesh Refinement Factor Around Layer Bounds",
        description="If not ``None``, refine mesh around minimum and maximum positions "
        "of the layer along normal axis dimension. If `min_steps_along_axis` is also specified, "
        "refinement here is only applied if it sets a smaller grid size.",
    )

    bounds_snapping: Literal["bounds", "lower", "upper", "center"] | None = Field(
        default="lower",
        title="Placing Grid Snapping Point Along Axis",
        description="If not ``None``, enforcing grid boundaries to pass through ``lower``, "
        "``center``, or ``upper`` position of the layer; or both ``lower`` and ``upper`` with ``bounds``.",
    )

    corner_finder: CornerFinderSpec | None = Field(
        default_factory=CornerFinderSpec,
        title="Inplane Corner Detection Specification",
        description="Specification for inplane corner detection. Inplane mesh refinement "
        "is based on the coordinates of those corners.",
    )

    corner_snapping: bool = Field(
        default=True,
        title="Placing Grid Snapping Point At Corners",
        description="If ``True`` and ``corner_finder`` is not ``None``, enforcing inplane "
        "grid boundaries to pass through corners of geometries specified by ``corner_finder``.",
    )

    corner_refinement: GridRefinement | None = Field(
        default_factory=GridRefinement,
        title="Inplane Mesh Refinement Factor Around Corners",
        description="If not ``None`` and ``corner_finder`` is not ``None``, refine mesh around "
        "corners of geometries specified by ``corner_finder``. The refined grid size is "
        "the finer of this field and the effective in-plane edge refinement "
        "(see ``in_plane_edge_refinement``).",
    )

    in_plane_edge_refinement: GridRefinement | Literal["mirror_corner"] | None = Field(
        default="mirror_corner",
        title="Inplane Mesh Refinement Along Axis-Unaligned Edges",
        description="Refine mesh along in-plane edges; enabled only when ``corner_finder`` "
        "is not ``None``. ``'mirror_corner'`` (default) uses ``corner_refinement``'s grid size; "
        "``None`` disables edge refinement; and a :class:`.GridRefinement` sets an explicit edge grid size. "
        "Internally, we classify edges into axis-aligned edges (handled already by ``corner_refinement``), and "
        "axis-unaligned edges. The classification is based on the edge angle "
        "relative to a threshold value defined by ``corner_finder.axis_aligned_angle_threshold``.",
    )

    min_steps_per_geometry: PositiveFloat | None = Field(
        default=2,
        title="Minimum Grid Steps Across A Small Geometry",
        description="If not ``None`` and ``corner_finder`` is not ``None``, sets the minimum "
        "number of grid cells to place across each disjoint geometry made of the medium "
        "specified by ``corner_finder``, ensuring small features are not under-resolved.",
    )

    refinement_inside_sim_only: bool = Field(
        default=True,
        title="Apply Refinement Only To Features Inside Simulation Domain",
        description="If ``True``, only apply mesh refinement to features such as corners inside "
        "the simulation domain; If ``False``, features outside the domain can take effect "
        "along the dimensions where the projection of the feature "
        "and the projection of the simulation domain overlaps.",
    )

    gap_meshing_iters: NonNegativeInt = Field(
        default=1,
        title="Gap Meshing Iterations",
        description="If ``corner_finder`` is not ``None``, number of recursive iterations for "
        "resolving thin gaps. "
        "The underlying algorithm detects gaps contained in a single cell and places a snapping plane at the gaps's centers.",
    )

    dl_min_from_gap_width: bool = Field(
        default=True,
        title="Set ``dl_min`` from Estimated Gap Width",
        description="Take into account autodetected minimal PEC gap width when determining ``dl_min``. "
        "This only applies if ``dl_min`` in ``AutoGrid`` specification is not set.",
    )

    interior_disjoint_geometries: bool = Field(
        default=True,
        title="Geometries Are Interior-Disjoint",
        description="If ``True``, geometries made of different materials on the plane must not be overlapping. "
        "This can speed up the performance "
        "of corner finder when there are many structures crossing the plane.",
    )

    @model_validator(mode="after")
    def _finite_size_along_axis(self) -> Self:
        if self.size is None:
            return self
        """size must be finite along axis."""
        if np.isinf(self.size[self.axis]):
            self._raise_validation_error_at_loc(
                SetupError("'size' must take finite values along 'axis' dimension."), "size"
            )
        return self


from .construction import from_bounds, from_layer_bounds, from_structures  # noqa: E402
from .edges import (  # noqa: E402
    _edge_run_override_structure,
    _is_inplane_bounded,
    _override_structures_edges,
    _polygon_rings_2d,
    _small_geometry_measurement_overrides,
    _unpop_axis,
    generate_override_structures,
    generate_snapping_points,
    suggested_dl_min,
)
from .gaps import (  # noqa: E402
    _find_vertical_intersections,
    _generate_horizontal_snapping_lines,
    _process_poly,
    _process_slice,
    _resolve_gaps,
)
from .inplane import (  # noqa: E402
    _connected_components,
    _corners,
    _corners_and_convexity_2d,
    _dl_min_from_smallest_feature,
    _inplane_footprint_box,
    _inplane_inside,
    _inplane_override_from_bbox,
    _layer_box,
    _merged_geos,
    _override_structures_along_axis,
    _override_structures_inplane,
    _snapping_points_along_axis,
    _union_inplane_overrides,
    _union_same_dl_overrides,
)
from .properties import (  # noqa: E402
    _corner_refinement,
    _edge_refinement,
    _inplane_merge_needed,
    center_axis,
    length_axis,
)

LayerRefinementSpec.from_layer_bounds = classmethod(from_layer_bounds)
LayerRefinementSpec.from_bounds = classmethod(from_bounds)
LayerRefinementSpec.from_structures = classmethod(from_structures)
LayerRefinementSpec.length_axis = property(length_axis)
LayerRefinementSpec.center_axis = property(center_axis)
LayerRefinementSpec._edge_refinement = property(_edge_refinement)
LayerRefinementSpec._inplane_merge_needed = property(_inplane_merge_needed)
LayerRefinementSpec._corner_refinement = _corner_refinement
LayerRefinementSpec._polygon_rings_2d = staticmethod(_polygon_rings_2d)
LayerRefinementSpec._edge_run_override_structure = _edge_run_override_structure
LayerRefinementSpec._override_structures_edges = _override_structures_edges
LayerRefinementSpec._small_geometry_measurement_overrides = _small_geometry_measurement_overrides
LayerRefinementSpec._is_inplane_bounded = _is_inplane_bounded
LayerRefinementSpec._unpop_axis = _unpop_axis
LayerRefinementSpec.suggested_dl_min = suggested_dl_min
LayerRefinementSpec.generate_snapping_points = generate_snapping_points
LayerRefinementSpec.generate_override_structures = generate_override_structures
LayerRefinementSpec._inplane_inside = _inplane_inside
LayerRefinementSpec._layer_box = _layer_box
LayerRefinementSpec._merged_geos = _merged_geos
LayerRefinementSpec._corners_and_convexity_2d = _corners_and_convexity_2d
LayerRefinementSpec._dl_min_from_smallest_feature = _dl_min_from_smallest_feature
LayerRefinementSpec._corners = _corners
LayerRefinementSpec._snapping_points_along_axis = property(_snapping_points_along_axis)
LayerRefinementSpec._override_structures_inplane = _override_structures_inplane
LayerRefinementSpec._union_inplane_overrides = _union_inplane_overrides
LayerRefinementSpec._union_same_dl_overrides = _union_same_dl_overrides
LayerRefinementSpec._inplane_footprint_box = _inplane_footprint_box
LayerRefinementSpec._connected_components = staticmethod(_connected_components)
LayerRefinementSpec._inplane_override_from_bbox = _inplane_override_from_bbox
LayerRefinementSpec._override_structures_along_axis = _override_structures_along_axis
LayerRefinementSpec._find_vertical_intersections = _find_vertical_intersections
LayerRefinementSpec._process_poly = _process_poly
LayerRefinementSpec._process_slice = _process_slice
LayerRefinementSpec._generate_horizontal_snapping_lines = _generate_horizontal_snapping_lines
LayerRefinementSpec._resolve_gaps = _resolve_gaps

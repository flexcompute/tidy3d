"""Layer-refinement construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.corner_finder import CornerFinderSpec
from tidy3d.components.types import Undefined
from tidy3d.constants import inf

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import NonNegativeInt, PositiveFloat

    from tidy3d.compat import Self
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Axis, Coordinate

    from .model import LayerRefinementSpec


from .model import GridRefinement


def from_layer_bounds(
    cls: type[LayerRefinementSpec],
    axis: Axis,
    bounds: tuple[float, float],
    min_steps_along_axis: PositiveFloat = None,
    bounds_refinement: GridRefinement = None,
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
    corner_finder: CornerFinderSpec | None | object = Undefined,
    corner_snapping: bool = True,
    corner_refinement: GridRefinement | None | object = Undefined,
    in_plane_edge_refinement: GridRefinement | Literal["mirror_corner"] | None = "mirror_corner",
    min_steps_per_geometry: PositiveFloat | None = 2,
    refinement_inside_sim_only: bool = True,
    gap_meshing_iters: NonNegativeInt = 1,
    dl_min_from_gap_width: bool = True,
    **kwargs: Any,
) -> Self:
    """Constructs a :class:`LayerRefinementSpec` that is unbounded in inplane dimensions from bounds along
    layer thickness dimension.

    Parameters
    ----------
    axis : Axis
        Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z).
    bounds : tuple[float, float]
        Minimum and maximum positions of the layer along axis dimension.
    min_steps_along_axis : PositiveFloat = None
        Minimal number of steps along axis.
    bounds_refinement : GridRefinement = None
        Mesh refinement factor around layer bounds.
    bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
        Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
        position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
    corner_finder : CornerFinderSpec = CornerFinderSpec()
        Inplane corner detection specification.
    corner_snapping : bool = True
        Placing grid snapping point at corners.
    corner_refinement : GridRefinement = GridRefinement()
        Inplane mesh refinement factor around corners.
    in_plane_edge_refinement : GridRefinement | Literal["mirror_corner"] | None = "mirror_corner"
        Inplane mesh refinement along axis-unaligned edges. ``"mirror_corner"`` uses
        ``corner_refinement``'s grid size; ``None`` disables edge refinement.
    min_steps_per_geometry : PositiveFloat | None = 2
        Minimum number of grid cells across each small disjoint metal geometry. ``None``
        disables small-geometry resolution.
    refinement_inside_sim_only : bool = True
        Apply refinement only to features inside simulation domain.
    gap_meshing_iters : bool = True
        Number of recursive iterations for resolving thin gaps.
    dl_min_from_gap_width : bool = True
        Take into account autodetected minimal PEC gap width when determining ``dl_min``.


    Example
    -------
    >>> from tidy3d.components.grid.grid_spec import LayerRefinementSpec
    >>> layer = LayerRefinementSpec.from_layer_bounds(axis=2, bounds=(0,1))

    """
    if corner_finder is Undefined:
        corner_finder = CornerFinderSpec()
    if corner_refinement is Undefined:
        corner_refinement = GridRefinement()

    center = Box.unpop_axis((bounds[0] + bounds[1]) / 2, (0, 0), axis)
    size = Box.unpop_axis((bounds[1] - bounds[0]), (inf, inf), axis)

    return cls(
        axis=axis,
        center=center,
        size=size,
        min_steps_along_axis=min_steps_along_axis,
        bounds_refinement=bounds_refinement,
        bounds_snapping=bounds_snapping,
        corner_finder=corner_finder,
        corner_snapping=corner_snapping,
        corner_refinement=corner_refinement,
        in_plane_edge_refinement=in_plane_edge_refinement,
        min_steps_per_geometry=min_steps_per_geometry,
        refinement_inside_sim_only=refinement_inside_sim_only,
        gap_meshing_iters=gap_meshing_iters,
        dl_min_from_gap_width=dl_min_from_gap_width,
        **kwargs,
    )


def from_bounds(
    cls: type[LayerRefinementSpec],
    rmin: Coordinate,
    rmax: Coordinate,
    axis: Axis = None,
    min_steps_along_axis: PositiveFloat = None,
    bounds_refinement: GridRefinement = None,
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
    corner_finder: CornerFinderSpec = Undefined,
    corner_snapping: bool = True,
    corner_refinement: GridRefinement = Undefined,
    in_plane_edge_refinement: GridRefinement | Literal["mirror_corner"] | None = "mirror_corner",
    min_steps_per_geometry: PositiveFloat | None = 2,
    refinement_inside_sim_only: bool = True,
    gap_meshing_iters: NonNegativeInt = 1,
    dl_min_from_gap_width: bool = True,
    **kwargs: Any,
) -> Self:
    """Constructs a :class:`LayerRefinementSpec` from minimum and maximum coordinate bounds.

    Parameters
    ----------
    rmin : tuple[float, float, float]
        (x, y, z) coordinate of the minimum values.
    rmax : tuple[float, float, float]
        (x, y, z) coordinate of the maximum values.
    axis : Axis
        Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z). If ``None``, apply the dimension
        along which the layer thas smallest thickness.
    min_steps_along_axis : PositiveFloat = None
        Minimal number of steps along axis.
    bounds_refinement : GridRefinement = None
        Mesh refinement factor around layer bounds.
    bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
        Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
        position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
    corner_finder : CornerFinderSpec = CornerFinderSpec()
        Inplane corner detection specification.
    corner_snapping : bool = True
        Placing grid snapping point at corners.
    corner_refinement : GridRefinement = GridRefinement()
        Inplane mesh refinement factor around corners.
    in_plane_edge_refinement : GridRefinement | Literal["mirror_corner"] | None = "mirror_corner"
        Inplane mesh refinement along axis-unaligned edges. ``"mirror_corner"`` uses
        ``corner_refinement``'s grid size; ``None`` disables edge refinement.
    min_steps_per_geometry : PositiveFloat | None = 2
        Minimum number of grid cells across each small disjoint metal geometry. ``None``
        disables small-geometry resolution.
    refinement_inside_sim_only : bool = True
        Apply refinement only to features inside simulation domain.
    gap_meshing_iters : bool = True
        Number of recursive iterations for resolving thin gaps.
    dl_min_from_gap_width : bool = True
        Take into account autodetected minimal PEC gap width when determining ``dl_min``.


    Example
    -------
    >>> from tidy3d.components.grid.grid_spec import LayerRefinementSpec
    >>> layer = LayerRefinementSpec.from_bounds(axis=2, rmin=(0,0,0), rmax=(1,1,1))

    """
    if corner_finder is Undefined:
        corner_finder = CornerFinderSpec()
    if corner_refinement is Undefined:
        corner_refinement = GridRefinement()

    box = Box.from_bounds(rmin=rmin, rmax=rmax)
    if axis is None:
        axis = np.argmin(box.size)
    return cls(
        axis=axis,
        center=box.center,
        size=box.size,
        min_steps_along_axis=min_steps_along_axis,
        bounds_refinement=bounds_refinement,
        bounds_snapping=bounds_snapping,
        corner_finder=corner_finder,
        corner_snapping=corner_snapping,
        corner_refinement=corner_refinement,
        in_plane_edge_refinement=in_plane_edge_refinement,
        min_steps_per_geometry=min_steps_per_geometry,
        refinement_inside_sim_only=refinement_inside_sim_only,
        gap_meshing_iters=gap_meshing_iters,
        dl_min_from_gap_width=dl_min_from_gap_width,
        **kwargs,
    )


def from_structures(
    cls: type[LayerRefinementSpec],
    structures: list[Structure],
    axis: Axis = None,
    min_steps_along_axis: PositiveFloat = None,
    bounds_refinement: GridRefinement = None,
    bounds_snapping: Literal["bounds", "lower", "upper", "center"] = "lower",
    corner_finder: CornerFinderSpec = Undefined,
    corner_snapping: bool = True,
    corner_refinement: GridRefinement = Undefined,
    in_plane_edge_refinement: GridRefinement | Literal["mirror_corner"] | None = "mirror_corner",
    min_steps_per_geometry: PositiveFloat | None = 2,
    refinement_inside_sim_only: bool = True,
    gap_meshing_iters: NonNegativeInt = 1,
    dl_min_from_gap_width: bool = True,
    **kwargs: Any,
) -> Self:
    """Constructs a :class:`LayerRefinementSpec` from the bounding box of a list of structures.

    Parameters
    ----------
    structures : list[Structure]
        A list of structures whose overall bounding box is used to define mesh refinement
    axis : Axis
        Specifies dimension of the layer normal axis (0,1,2) -> (x,y,z). If ``None``, apply the dimension
        along which the bounding box of the structures thas smallest thickness.
    min_steps_along_axis : PositiveFloat = None
        Minimal number of steps along axis.
    bounds_refinement : GridRefinement = None
        Mesh refinement factor around layer bounds.
    bounds_snapping : Literal["bounds", "lower", "upper", "center"] = "lower"
        Placing grid snapping point along axis:  ``lower``, ``center``, or ``upper``
        position of the layer; or both ``lower`` and ``upper`` with ``bounds``.
    corner_finder : CornerFinderSpec = CornerFinderSpec()
        Inplane corner detection specification.
    corner_snapping : bool = True
        Placing grid snapping point at corners.
    corner_refinement : GridRefinement = GridRefinement()
        Inplane mesh refinement factor around corners.
    in_plane_edge_refinement : GridRefinement | Literal["mirror_corner"] | None = "mirror_corner"
        Inplane mesh refinement along axis-unaligned edges. ``"mirror_corner"`` uses
        ``corner_refinement``'s grid size; ``None`` disables edge refinement.
    min_steps_per_geometry : PositiveFloat | None = 2
        Minimum number of grid cells across each small disjoint metal geometry. ``None``
        disables small-geometry resolution.
    refinement_inside_sim_only : bool = True
        Apply refinement only to features inside simulation domain.
    gap_meshing_iters : bool = True
        Number of recursive iterations for resolving thin gaps.
    dl_min_from_gap_width : bool = True
        Take into account autodetected minimal PEC gap width when determining ``dl_min``.

    """
    if corner_finder is Undefined:
        corner_finder = CornerFinderSpec()
    if corner_refinement is Undefined:
        corner_refinement = GridRefinement()

    all_bounds = tuple(structure.geometry.bounds for structure in structures)
    rmin = tuple(min(b[i] for b, _ in all_bounds) for i in range(3))
    rmax = tuple(max(b[i] for _, b in all_bounds) for i in range(3))
    box = Box.from_bounds(rmin=rmin, rmax=rmax)
    if axis is None:
        axis = np.argmin(box.size)

    return cls(
        axis=axis,
        center=box.center,
        size=box.size,
        min_steps_along_axis=min_steps_along_axis,
        bounds_refinement=bounds_refinement,
        bounds_snapping=bounds_snapping,
        corner_finder=corner_finder,
        corner_snapping=corner_snapping,
        corner_refinement=corner_refinement,
        in_plane_edge_refinement=in_plane_edge_refinement,
        min_steps_per_geometry=min_steps_per_geometry,
        refinement_inside_sim_only=refinement_inside_sim_only,
        gap_meshing_iters=gap_meshing_iters,
        dl_min_from_gap_width=dl_min_from_gap_width,
        **kwargs,
    )

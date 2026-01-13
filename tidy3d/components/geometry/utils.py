"""Compatibility shim for :mod:`tidy3d._common.components.geometry.utils`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as partially migrated to _common

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import numpy as np

from tidy3d._common.components.geometry.utils import (
    GeometryType,
    SnapBehavior,
    SnapLocation,
    flatten_groups,
    flatten_shapely_geometries,
    from_shapely,
    get_closest_value,
    merging_geometries_on_plane,
    traverse_geometries,
    validate_no_transformed_polyslabs,
    vertices_from_shapely,
)
from tidy3d.components.geometry.base import Box
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import ArrayLike
    from pydantic import NonNegativeInt

    from tidy3d._common.components.geometry.utils import SnappingSpec
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.types.base import (
        Bound,
        Coordinate,
        Direction,
    )


def snap_box_to_grid(grid: Grid, box: Box, snap_spec: SnappingSpec, rtol: float = fp_eps) -> Box:
    """Snaps a :class:`.Box` to the grid, so that the boundaries of the box are aligned with grid centers or boundaries.
    The way in which each dimension of the `box` is snapped to the grid is controlled by ``snap_spec``.
    """

    def _clamp_index(idx: int, length: int) -> int:
        return max(0, min(idx, length - 1))

    def get_lower_bound(
        test: float,
        coords: ArrayLike,
        upper_bound_idx: int,
        rel_tol: float,
        strict_bounds: bool,
        margin: int = 0,
    ) -> float:
        """Choose the lower bound from coords for snapping a test value downward.

        Returns a coordinate value from ``coords`` that satisfies ``result <= test`` when
        ``strict_bounds=False``, or ``result < test`` when ``strict_bounds=True``. The
        ``rel_tol`` parameter is used to determine floating-point equality: if ``test`` is
        close to a grid point (within ``rel_tol``).

        Parameters
        ----------
        test : float
            The value to snap.
        coords : ArrayLike
            Sorted array of coordinate values to snap to.
        upper_bound_idx : int
            Index from ``np.searchsorted(coords, test, side="left")`` - the first index where
            ``coords[upper_bound_idx] >= test``.
        rel_tol : float
            Relative tolerance for floating-point equality comparison.
        strict_bounds : bool
            If ``False``: Return value satisfies ``result <= test`` (using ``rel_tol`` for equality).
            If ``True``: Return value satisfies ``result < test`` (using ``rel_tol`` for equality).
        margin : int, optional
            Additional offset in grid cells (can be negative). Applied after determining the
            snap index. Default is 0.

        Returns
        -------
        float
            The selected coordinate value from ``coords``.
        """
        snap_idx = upper_bound_idx - 1
        if (
            not strict_bounds
            and upper_bound_idx != len(coords)
            and isclose(coords[upper_bound_idx], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx
        elif (
            strict_bounds
            and upper_bound_idx >= 2
            and isclose(coords[upper_bound_idx - 1], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx - 2

        # Apply margin and clamp
        snap_idx += margin
        snap_idx = _clamp_index(snap_idx, len(coords))
        return coords[snap_idx]

    def get_upper_bound(
        test: float,
        coords: ArrayLike,
        upper_bound_idx: int,
        rel_tol: float,
        strict_bounds: bool,
        margin: int = 0,
    ) -> float:
        """Choose the upper bound from coords for snapping a test value upward.

        Returns a coordinate value from ``coords`` that satisfies ``result >= test`` when
        ``strict_bounds=False``, or ``result > test`` when ``strict_bounds=True``. The
        ``rel_tol`` parameter is used to determine floating-point equality: if ``test`` is
        close to a grid point (within ``rel_tol``).

        Parameters
        ----------
        test : float
            The value to snap.
        coords : ArrayLike
            Sorted array of coordinate values to snap to.
        upper_bound_idx : int
            Index from ``np.searchsorted(coords, test, side="left")`` - the first index where
            ``coords[upper_bound_idx] >= test``.
        rel_tol : float
            Relative tolerance for floating-point equality comparison.
        strict_bounds : bool
            If ``False``: Return value satisfies ``result >= test`` (using ``rel_tol`` for equality).
            If ``True``: Return value satisfies ``result > test`` (using ``rel_tol`` for equality).
        margin : int, optional
            Additional offset in grid cells (can be negative). Applied after determining the
            snap index. Default is 0.

        Returns
        -------
        float
            The selected coordinate value from ``coords``.
        """
        snap_idx = upper_bound_idx

        if (
            not strict_bounds
            and upper_bound_idx > 0
            and (isclose(coords[upper_bound_idx - 1], test, rel_tol=rel_tol))
        ):
            snap_idx = upper_bound_idx - 1
        elif (
            strict_bounds
            and upper_bound_idx < len(coords)
            and isclose(coords[upper_bound_idx], test, rel_tol=rel_tol)
        ):
            snap_idx = upper_bound_idx + 1

        # Apply margin and clamp
        snap_idx += margin
        snap_idx = _clamp_index(snap_idx, len(coords))
        return coords[snap_idx]

    def find_snapping_locations(
        interval_min: float,
        interval_max: float,
        coords: np.ndarray,
        snap_type: SnapBehavior,
        snap_margin: NonNegativeInt,
    ) -> tuple[float, float]:
        """Helper that snaps a supplied interval [interval_min, interval_max] to a
        sorted array representing coordinate values.
        """
        # Locate the interval that includes the min and max
        min_upper_bound_idx = np.searchsorted(coords, interval_min, side="left")
        max_upper_bound_idx = np.searchsorted(coords, interval_max, side="left")
        strict_bounds = (
            snap_type == SnapBehavior.StrictExpand or snap_type == SnapBehavior.StrictContract
        )
        if snap_type == SnapBehavior.Closest:
            min_snap = get_closest_value(interval_min, coords, min_upper_bound_idx)
            max_snap = get_closest_value(interval_max, coords, max_upper_bound_idx)
        elif snap_type == SnapBehavior.Expand or snap_type == SnapBehavior.StrictExpand:
            min_snap = get_lower_bound(
                interval_min,
                coords,
                min_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=-snap_margin,
            )
            max_snap = get_upper_bound(
                interval_max,
                coords,
                max_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=+snap_margin,
            )
        else:  # SnapType.Contract
            min_snap = get_upper_bound(
                interval_min,
                coords,
                min_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=+snap_margin,
            )
            max_snap = get_lower_bound(
                interval_max,
                coords,
                max_upper_bound_idx,
                rel_tol=rtol,
                strict_bounds=strict_bounds,
                margin=-snap_margin,
            )
            if max_snap < min_snap:
                raise SetupError("The supplied 'snap_margin' is too large for this contraction.")
        return (min_snap, max_snap)

    # Iterate over each axis and apply the specified snapping behavior.
    min_b, max_b = (list(f) for f in box.bounds)
    grid_bounds = grid.boundaries.to_list
    grid_centers = grid.centers.to_list
    for axis in range(3):
        snap_location = snap_spec.location[axis]
        snap_type = snap_spec.behavior[axis]
        snap_margin = snap_spec.margin[axis]
        if snap_type == SnapBehavior.Off:
            continue
        if snap_location == SnapLocation.Boundary:
            snap_coords = np.array(grid_bounds[axis])
        elif snap_location == SnapLocation.Center:
            snap_coords = np.array(grid_centers[axis])

        box_min = min_b[axis]
        box_max = max_b[axis]

        (new_min, new_max) = find_snapping_locations(
            box_min, box_max, snap_coords, snap_type, snap_margin
        )
        min_b[axis] = new_min
        max_b[axis] = new_max
    return Box.from_bounds(min_b, max_b)


def snap_point_to_grid(
    grid: Grid, point: Coordinate, snap_location: tuple[SnapLocation, SnapLocation, SnapLocation]
) -> Coordinate:
    """Snaps a :class:`.Coordinate` to the grid, so that it is coincident with grid centers or boundaries.
    The way in which each dimension of the ``point`` is snapped to the grid is controlled by ``snap_location``.
    """
    grid_bounds = grid.boundaries.to_list
    grid_centers = grid.centers.to_list
    snapped_point = 3 * [0]
    for axis in range(3):
        if snap_location[axis] == SnapLocation.Boundary:
            snap_coords = np.array(grid_bounds[axis])
        elif snap_location[axis] == SnapLocation.Center:
            snap_coords = np.array(grid_centers[axis])

        # Locate the interval that includes the test point
        min_upper_bound_idx = np.searchsorted(snap_coords, point[axis], side="left")
        snapped_point[axis] = get_closest_value(point[axis], snap_coords, min_upper_bound_idx)

    return tuple(snapped_point)


def _shift_value_signed(
    obj: Box,
    grid: Grid,
    bounds: Bound,
    direction: Direction,
    shift: int,
    name: Optional[str] = None,
) -> float:
    """Calculate the signed distance corresponding to moving the object by ``shift`` number
    of cells in the positive or negative ``direction`` along the dimension given by
    ``obj._normal_axis``.
    """
    if name is None:
        name = f"A '{obj.type}'"

    # get the grid boundaries and sizes along obj normal from the simulation
    normal_axis = obj._normal_axis
    grid_boundaries = grid.boundaries.to_list[normal_axis]
    grid_centers = grid.centers.to_list[normal_axis]

    # get the index of the grid cell where the obj lies
    obj_position = obj.center[normal_axis]
    obj_pos_gt_grid_bounds = np.flatnonzero(obj_position > grid_boundaries)

    # no obj index can be determined
    if len(obj_pos_gt_grid_bounds) == 0 or obj_position > grid_boundaries[-1]:
        raise SetupError(
            f"{name} position '{obj_position}' is outside of simulation bounds '({grid_boundaries[0]}, {grid_boundaries[-1]})' along dimension '{'xyz'[normal_axis]}'."
        )
    obj_index = obj_pos_gt_grid_bounds[-1]
    # shift the obj to the left
    signed_shift = shift if direction == "+" else -shift
    if signed_shift < 0:
        if np.isclose(obj_position, grid_boundaries[obj_index + 1]):
            obj_index += 1
        shifted_index = obj_index + signed_shift
        if shifted_index < 0 or grid_centers[shifted_index] <= bounds[0][normal_axis]:
            raise SetupError(
                f"{name} normal is less than 2 cells to the boundary "
                f"on -{'xyz'[normal_axis]} side. "
                "Please either increase the mesh resolution near the obj or "
                "move the obj away from the boundary."
            )

    # shift the obj to the right
    else:
        shifted_index = obj_index + signed_shift
        if (
            shifted_index >= len(grid_centers)
            or grid_centers[shifted_index] >= bounds[1][normal_axis]
        ):
            raise SetupError(
                f"{name} normal is less than 2 cells to the boundary "
                f"on +{'xyz'[normal_axis]} side."
                "Please either increase the mesh resolution near the obj or "
                "move the obj away from the boundary."
            )

    new_pos = grid_centers[shifted_index]
    return new_pos - obj_position


def _shift_object(obj: Box, grid: Grid, bounds: Bound, direction: Direction, shift: int) -> Box:
    """Move a plane-like object by ``shift`` number
    of cells in the positive or negative ``direction`` along the dimension given by
    ``obj._normal_axis``.
    """
    shift = _shift_value_signed(obj=obj, grid=grid, bounds=bounds, direction=direction, shift=shift)
    new_center = np.array(obj.center)
    new_center[obj._normal_axis] += shift
    # note: if this needs to be generalized beyond absorber, one would probably
    # slightly adjust the code below regarding grid_shift
    return obj.updated_copy(center=tuple(new_center), grid_shift=0)

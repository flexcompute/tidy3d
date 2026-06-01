"""Utilities for point-cloud data validation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .data_array import PointDataArray

if TYPE_CHECKING:
    from collections.abc import Sequence

POINT_CLOUD_STENCIL_CORNERS_PER_FIELD = 8


def _axis_label_to_index(label: object) -> int | None:
    """Map supported point-cloud axis labels to canonical integer axis indices."""

    if isinstance(label, str):
        return {"x": 0, "y": 1, "z": 2}.get(label.lower())

    if label in (0, 1, 2):
        return int(label)

    return None


def canonicalize_point_cloud_points(
    points: PointDataArray,
    *,
    empty_error: str,
    max_num_points: int | None = None,
    require_real: bool = False,
    require_finite: bool = False,
    cast_to_float: bool = False,
) -> PointDataArray:
    """Validate and canonicalize point-cloud coordinate arrays."""

    if points.sizes.get("axis") != 3:
        raise ValueError(
            "Point-cloud coordinates must have exactly three entries along the 'axis' dimension."
        )

    if "axis" in points.coords:
        axis_indices = [_axis_label_to_index(label) for label in points.coords["axis"].values]
        if set(axis_indices) != {0, 1, 2}:
            raise ValueError(
                "Point-cloud coordinate 'axis' labels must be a permutation of "
                "(0, 1, 2) or ('x', 'y', 'z')."
            )
        axis_order = [axis_indices.index(axis) for axis in range(3)]
        if axis_order != [0, 1, 2]:
            points = points.isel(axis=axis_order)

    num_points = points.sizes.get("index", 0)
    if num_points == 0:
        raise ValueError(empty_error)

    if max_num_points is not None and num_points > max_num_points:
        raise ValueError(f"Point-cloud monitors support at most {max_num_points} points.")

    if require_real or require_finite or cast_to_float:
        values = np.asarray(points.values)
        if require_real and np.iscomplexobj(values):
            raise ValueError("Point-cloud coordinates must be real-valued.")

        if require_finite:
            try:
                values_are_finite = np.isfinite(values)
            except TypeError as exc:
                raise ValueError(
                    "Point-cloud coordinates must be finite real numbers. "
                    f"Failed to test finiteness: {exc}"
                ) from exc

            if not np.all(values_are_finite):
                raise ValueError("Point-cloud coordinates must be finite real numbers.")

        if cast_to_float:
            points = points.astype(float, copy=False)

    points = points.assign_coords(index=np.arange(num_points), axis=np.arange(3))
    return PointDataArray(points)


def point_cloud_stencil_grid_num_cells(
    num_cells: Sequence[int], symmetry: Sequence[int]
) -> tuple[int, int, int]:
    """Conservative point-cloud interpolation grid size."""

    grid_num_cells = []
    for num_cells_dim, symmetry_dim in zip(num_cells, symmetry):
        if symmetry_dim != 0:
            grid_num_cells.append(num_cells_dim - num_cells_dim // 2 + 2)
        else:
            grid_num_cells.append(num_cells_dim + 2)

    return tuple(grid_num_cells)


def point_cloud_sampled_cells_upper_bound(
    *,
    num_cells: Sequence[int],
    symmetry: Sequence[int],
    num_points: int,
    num_fields: int,
) -> int:
    """Conservative upper bound on sampled Yee cells for point-cloud interpolation."""

    if num_points == 0 or num_fields == 0:
        return 0

    stencil_rows = POINT_CLOUD_STENCIL_CORNERS_PER_FIELD * num_points * num_fields
    grid_cells = math.prod(point_cloud_stencil_grid_num_cells(num_cells, symmetry))
    return int(min(stencil_rows, grid_cells))

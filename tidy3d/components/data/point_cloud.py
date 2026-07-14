"""Utilities for point-cloud data validation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .data_array import PointDataArray

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

POINT_CLOUD_STENCIL_CORNERS_PER_FIELD = 8
POINT_CLOUD_PERMITTIVITY_COMPONENTS = ("eps_xx", "eps_yy", "eps_zz")


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
    coordinate_name: str = "Point-cloud coordinates",
    preserve_index: bool = False,
) -> PointDataArray:
    """Validate and canonicalize point-cloud coordinate arrays."""

    if points.sizes.get("axis") != 3:
        raise ValueError(
            f"{coordinate_name} must have exactly three entries along the 'axis' dimension."
        )

    if "axis" in points.coords:
        axis_indices = [_axis_label_to_index(label) for label in points.coords["axis"].values]
        if axis_indices != [0, 1, 2]:
            raise ValueError(
                f"{coordinate_name} 'axis' labels must be ordered as (0, 1, 2) or ('x', 'y', 'z')."
            )

    num_points = points.sizes.get("index", 0)
    if num_points == 0:
        raise ValueError(empty_error)

    if max_num_points is not None and num_points > max_num_points:
        raise ValueError(f"Point-cloud monitors support at most {max_num_points} points.")

    if require_real or require_finite or cast_to_float:
        values = np.asarray(points.values)
        if require_real and np.iscomplexobj(values):
            raise ValueError(f"{coordinate_name} must be real-valued.")

        if require_finite:
            try:
                values_are_finite = np.isfinite(values)
            except TypeError as exc:
                raise ValueError(
                    f"{coordinate_name} must be finite real numbers. "
                    f"Failed to test finiteness: {exc}"
                ) from exc

            if not np.all(values_are_finite):
                raise ValueError(f"{coordinate_name} must be finite real numbers.")

        if cast_to_float:
            points = points.astype(float, copy=False)

    index_coords = (
        np.asarray(points.coords["index"].values)
        if preserve_index and "index" in points.coords
        else np.arange(num_points)
    )
    points = points.assign_coords(index=index_coords, axis=np.arange(3))
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
    return min(stencil_rows, grid_cells)


def point_cloud_nearest_sampled_cells_upper_bound(
    *,
    num_cells: Sequence[int],
    symmetry: Sequence[int],
    num_points: int,
    num_components: int,
) -> int:
    """Conservative upper bound on nearest sampled Yee cells for point-cloud data."""

    if num_points == 0 or num_components == 0:
        return 0

    grid_cells = math.prod(point_cloud_stencil_grid_num_cells(num_cells, symmetry))
    return min(num_points * num_components, grid_cells)


def point_cloud_grid_field(field: str) -> str:
    """Return the Yee-grid field sampled for a point-cloud output component."""

    if field[0] == "D":
        return f"E{field[-1]}"
    return field


def point_cloud_num_sampled_grid_fields(fields: Iterable[str]) -> int:
    """Number of distinct component-native grids needed to sample point-cloud raw fields."""

    return len({point_cloud_grid_field(field) for field in fields})

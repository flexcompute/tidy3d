"""Shared helper utilities for source-adjoint gradient processing."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.autograd.derivative_utils import compute_spatial_weights
from tidy3d.components.autograd.path_utils import format_traced_path
from tidy3d.components.autograd.utils import get_static
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.grid.grid import Coords
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tidy3d.components.autograd.types import PathType
    from tidy3d.components.data.monitor_data import ElectromagneticFieldData
    from tidy3d.components.types.base import Bound, Coordinate, Size


def collapse_source_adjoint_to_dataset_frequency(
    fld_adj: ElectromagneticFieldData,
    source_dataset_freq: float,
) -> ElectromagneticFieldData:
    """Accumulate source-adjoint frequencies onto the single source-dataset frequency."""
    update: dict[str, Any] = {}
    target_freq = np.asarray([source_dataset_freq], dtype=float)
    for field_name, field_data in fld_adj.field_components.items():
        collapsed = field_data.sum(dim="f")
        collapsed = collapsed.expand_dims({"f": target_freq}, axis=-1)
        update[field_name] = collapsed.transpose(*field_data.dims)
    return fld_adj.updated_copy(**update)


def split_source_paths(paths: Sequence[PathType]) -> tuple[list[PathType], list[PathType]]:
    """Split source paths into primary and center groups.

    Parameters
    ----------
    paths
        Traced source paths.
    Notes
    -----
    ``center`` paths are always split into the second output list. The setup validator is
    responsible for rejecting unsupported non-center roots.
    """
    primary_paths: list[PathType] = []
    center_paths: list[PathType] = []
    for path in paths:
        path = tuple(path)
        if path and path[0] == "center":
            center_paths.append(path)
        else:
            primary_paths.append(path)
    return primary_paths, center_paths


def _center_path_axes(path: PathType) -> tuple[int, ...]:
    """Return axis indices addressed by a validated traced center path."""
    if len(path) == 1:
        return (0, 1, 2)
    return (int(path[1]),)


def validate_center_path_axes(path: PathType) -> tuple[int, ...]:
    """Validate and return axis indices addressed by a traced center path."""
    path = tuple(path)
    if not path or path[0] != "center":
        raise AdjointError(
            f"Expected 'center' traced source parameter, got '{format_traced_path(path)}'."
        )
    if len(path) == 1:
        return (0, 1, 2)
    if len(path) == 2:
        axis = path[1]
        if not isinstance(axis, Integral) or axis not in (0, 1, 2):
            raise AdjointError(
                f"Unsupported traced source parameter '{format_traced_path(path)}'. "
                "Only center, center[0], center[1], and center[2] are supported."
            )
        return (int(axis),)
    raise AdjointError(
        f"Unsupported traced source parameter '{format_traced_path(path)}'. "
        "Only full-vector paths or single-axis paths are supported for center."
    )


def validate_source_field_component(field_name: str, *, source_name: str) -> None:
    """Validate a source field component name like ``Ex`` or ``Hz``."""
    if (
        len(field_name) != 2
        or field_name[0] not in ("E", "H")
        or field_name[1] not in ("x", "y", "z")
    ):
        raise AdjointError(
            f"Unsupported field component '{field_name}' in {source_name}. "
            "Expected one of Ex, Ey, Ez, Hx, Hy, Hz."
        )


def parse_source_field_component(field_name: str) -> tuple[str, int]:
    """Parse a validated source field component name like ``Ex`` or ``Hz``."""
    return field_name[0], "xyz".index(field_name[1])


def assign_center_path_derivatives(
    derivative_map: dict[PathType, Any],
    center_paths: Sequence[PathType],
    *,
    vjp_center: np.ndarray,
) -> None:
    """Write center derivatives to traced paths."""
    center_vjp_arr = np.asarray(vjp_center, dtype=float).reshape(-1)
    if center_vjp_arr.size != 3:
        raise ValueError(
            f"Expected 3 center derivative components, got shape {center_vjp_arr.shape}."
        )
    center_vjp = tuple(center_vjp_arr.tolist())
    for field_path in center_paths:
        field_path = tuple(field_path)
        axes = _center_path_axes(field_path)
        if len(axes) == 1:
            derivative_map[field_path] = center_vjp[axes[0]]
        else:
            derivative_map[field_path] = center_vjp


def validate_no_zero_dim_center_paths(
    center_paths: Sequence[PathType],
    *,
    source_size: Size,
    source_name: str,
) -> None:
    """Reject center derivatives on collapsed source axes."""
    source_size_arr = np.asarray(get_static(source_size), dtype=float)
    for field_path in center_paths:
        path = tuple(field_path)
        axes = validate_center_path_axes(path)

        for axis in axes:
            if np.isclose(source_size_arr[axis], 0.0):
                raise AdjointError(
                    f"{source_name} does not support derivatives on collapsed axis "
                    f"'{'xyz'[axis]}' for source parameter "
                    f"'{format_traced_path(path)}'."
                )


def _axis_bounds_or_none(
    arr: SpatialDataArray, bounds: Bound, axis: int
) -> tuple[np.ndarray, float, float] | None:
    """Return axis coords and bounds if valid, otherwise ``None``."""
    dim = "xyz"[axis]
    if dim not in arr.coords:
        return None
    coords = arr.coords[dim].values
    if coords.size <= 1 or not np.all(np.isfinite(coords)):
        return None
    bound_min = float(get_static(bounds[0][axis]))
    bound_max = float(get_static(bounds[1][axis]))
    if np.isclose(bound_min, bound_max):
        return None
    return coords, bound_min, bound_max


def _static_bounds(bounds: Bound) -> Bound:
    """Convert bounds entries to static floats."""
    lower = tuple(get_static(value) for value in bounds[0])
    upper = tuple(float(get_static(value)) for value in bounds[1])
    return (lower, upper)


def compute_center_vjp(
    adjoint_field: SpatialDataArray,
    field_on_grid: SpatialDataArray,
    bounds: Bound,
    *,
    component_sign: float,
    dims_to_integrate: tuple[str, ...],
) -> np.ndarray:
    """Compute center VJP from full-profile source/adjoint fields."""
    vjp_center = np.zeros(3, dtype=float)
    bounds_static = _static_bounds(bounds)

    field_on_grid = field_on_grid.transpose(*adjoint_field.dims)
    field_inside = field_on_grid.sel_inside(bounds_static, include_interp_padding=False)
    weights_inside = compute_spatial_weights(field_inside, dims=dims_to_integrate)

    for axis in range(3):
        dim = "xyz"[axis]
        axis_data = _axis_bounds_or_none(adjoint_field, bounds_static, axis)
        if axis_data is None:
            continue
        coords, _, _ = axis_data

        axis_idx = adjoint_field.dims.index(dim)
        grad_adjoint = np.gradient(
            adjoint_field.values,
            coords,
            axis=axis_idx,
            edge_order=1,
        )
        grad_adjoint_da = SpatialDataArray(
            grad_adjoint,
            coords=adjoint_field.coords,
            dims=adjoint_field.dims,
        ).sel_inside(bounds_static, include_interp_padding=False)
        center_density_da = np.real(
            component_sign * grad_adjoint_da * field_inside * weights_inside
        )
        vjp_center[axis] += np.sum(center_density_da.values)

    return vjp_center


def accumulate_center_vjp(
    *,
    field_components: dict[str, SpatialDataArray],
    center: Coordinate,
    bounds: Bound,
    source_size: Size,
    get_adjoint_and_sign: Callable[[str], tuple[SpatialDataArray, float]],
) -> np.ndarray:
    """Accumulate center VJPs across source dataset components."""
    vjp_center = np.zeros(3, dtype=float)
    center = tuple(get_static(value) for value in center)

    for field_name, field_data in field_components.items():
        adjoint_field, component_sign = get_adjoint_and_sign(field_name)
        adjoint_field = adjoint_field.squeeze("f", drop=True)
        field_data = field_data.squeeze("f", drop=True)
        target_grid = Coords(
            x=adjoint_field.coords["x"].values,
            y=adjoint_field.coords["y"].values,
            z=adjoint_field.coords["z"].values,
        )
        field_on_grid = field_data._spatially_sorted.interpolate_to_grid(
            target_grid,
            offset=center,
            method="linear",
            target_dims=tuple(adjoint_field.dims),
        )
        dims_to_integrate = tuple(
            dim
            for axis, dim in enumerate("xyz")
            if dim in adjoint_field.coords and source_size[axis] > 0.0
        )

        center_contrib = compute_center_vjp(
            adjoint_field=adjoint_field,
            field_on_grid=field_on_grid,
            bounds=bounds,
            component_sign=component_sign,
            dims_to_integrate=dims_to_integrate,
        )
        vjp_center += center_contrib

    return vjp_center

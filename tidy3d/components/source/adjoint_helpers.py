"""Shared helper utilities for source-adjoint gradient processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.autograd.derivative_utils import compute_spatial_weights
from tidy3d.components.autograd.utils import get_static
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.grid.grid import Coords
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, Optional

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


def split_source_paths(
    paths: Sequence[PathType], *, dataset_tag: str
) -> tuple[list[PathType], list[PathType]]:
    """Split source paths into dataset and center paths."""
    dataset_paths: list[PathType] = []
    center_paths: list[PathType] = []
    for path in paths:
        path = tuple(path)
        if path and path[0] == "center":
            center_paths.append(path)
        elif path and path[0] == dataset_tag:
            dataset_paths.append(path)
        else:
            raise ValueError(
                f"Unexpected traced source path '{path}'. Paths must be validated before "
                "calling 'split_source_paths'."
            )
    return dataset_paths, center_paths


def parse_source_field_component(field_name: str, *, source_name: str) -> tuple[str, int]:
    """Parse and validate a source field component name like ``Ex`` or ``Hz``."""
    if (
        len(field_name) != 2
        or field_name[0] not in ("E", "H")
        or field_name[1] not in ("x", "y", "z")
    ):
        raise ValueError(
            f"Unsupported field component '{field_name}' in {source_name}. "
            "Expected one of Ex, Ey, Ez, Hx, Hy, Hz."
        )
    return field_name[0], "xyz".index(field_name[1])


def assign_center_path_derivatives(
    derivative_map: dict[PathType, Any],
    center_paths: Sequence[PathType],
    *,
    vjp_center: np.ndarray,
) -> None:
    """Write center derivatives to traced paths."""
    center_vjp = tuple(vjp_center.tolist())
    for field_path in center_paths:
        field_path = tuple(field_path)
        if len(field_path) == 2:
            derivative_map[field_path] = center_vjp[int(field_path[1])]
        else:
            derivative_map[field_path] = center_vjp


def validate_no_zero_dim_center_paths(
    center_paths: Sequence[PathType],
    *,
    source_size: Size,
    source_name: str,
) -> None:
    """Reject center derivatives on collapsed source axes."""
    source_size_arr = np.asarray(source_size, dtype=float)
    for field_path in center_paths:
        path = tuple(field_path)
        if not path or path[0] != "center":
            continue

        if len(path) >= 2:
            axes = (int(path[1]),)
        else:
            axes = (0, 1, 2)

        for axis in axes:
            if np.isclose(source_size_arr[axis], 0.0):
                raise AdjointError(
                    f"{source_name} does not support derivatives on collapsed axis "
                    f"'{'xyz'[axis]}' for traced path {path!r}."
                )


def _axis_bounds_or_none(
    arr: SpatialDataArray, bounds: Bound, axis: int, label: str
) -> Optional[tuple[np.ndarray, float, float]]:
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
        raise AdjointError(
            f"{label}: center derivatives on collapsed axis '{dim}' are not supported."
        )
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
    label: str,
) -> np.ndarray:
    """Compute center VJP from full-profile source/adjoint fields."""
    vjp_center = np.zeros(3, dtype=float)
    bounds_static = _static_bounds(bounds)

    field_on_grid = field_on_grid.transpose(*adjoint_field.dims)
    field_inside = field_on_grid.sel_inside(bounds_static, include_interp_padding=False)
    weights_inside = compute_spatial_weights(field_inside, dims=dims_to_integrate)

    for axis in range(3):
        dim = "xyz"[axis]
        axis_data = _axis_bounds_or_none(adjoint_field, bounds_static, axis, label)
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
    label_prefix: str,
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
            label=f"{label_prefix}:{field_name}",
        )
        vjp_center += center_contrib

    return vjp_center

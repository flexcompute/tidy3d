"""Small helpers shared by monitor-data implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np

from tidy3d.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tidy3d.components.data.data_array import DataArray

    from ._types import DataArrayEntry, SourceT


def _values_in_dim_order(data_array: DataArray, dim_order: tuple[str, ...]) -> np.ndarray:
    """Return data values with axes ordered by ``dim_order``."""
    data_dims = tuple(data_array.dims)
    if set(data_dims) != set(dim_order) or len(data_dims) != len(dim_order):
        raise DataError(f"Expected data dimensions {dim_order}, got {data_dims}.")

    axis_order = tuple(data_dims.index(dim) for dim in dim_order)
    return np.transpose(data_array.values, axes=axis_order)


def _iter_nonzero_data_array_entries(
    data_array: DataArray,
    dim_order: tuple[str, ...],
    *,
    skip_nan: bool,
) -> Iterator[DataArrayEntry]:
    """Iterate nonzero entries with indices and coords ordered by ``dim_order``."""
    values = _values_in_dim_order(data_array, dim_order)
    is_valid = values != 0.0
    if skip_nan:
        is_valid = is_valid & ~np.isnan(values)

    coords = tuple(data_array.coords[dim].values for dim in dim_order)
    for index in zip(*np.nonzero(is_valid)):
        coord_values = tuple(
            dim_coords[axis_index] for dim_coords, axis_index in zip(coords, index)
        )
        yield index, coord_values, complex(values[index])


def _make_adjoint_sources_from_modal_amps(
    amps: DataArray,
    source_from_amp: Callable[[float, str, int, complex], SourceT],
    *,
    skip_nan: bool,
) -> list[SourceT]:
    """Build sources for nonzero ``(f, direction, mode_index)`` amplitudes."""
    return [
        source_from_amp(freq, direction, mode_index, amp_complex)
        for _, (freq, direction, mode_index), amp_complex in _iter_nonzero_data_array_entries(
            amps,
            ("f", "direction", "mode_index"),
            skip_nan=skip_nan,
        )
    ]


__all__ = [
    "_iter_nonzero_data_array_entries",
    "_make_adjoint_sources_from_modal_amps",
    "_values_in_dim_order",
]

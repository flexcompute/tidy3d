from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Optional, Union

import numpy as np

from tidy3d.components.grid.grid import Coords
from tidy3d.components.types import ArrayFloat1D

AXIS_ORDER = ("x", "y", "z")
GridCoords = Union[
    Coords,
    Mapping[str, ArrayFloat1D],
    Sequence[ArrayFloat1D],
]


def coerce_coords_input(coords: GridCoords | None) -> GridCoords | None:
    """Coerce coordinate inputs into concrete sequences for Pydantic parsing."""
    if coords is None or isinstance(coords, Coords):
        return coords
    if isinstance(coords, Mapping):
        return {key: list(val) if isinstance(val, Iterator) else val for key, val in coords.items()}
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes)):
        return tuple(list(val) if isinstance(val, Iterator) else val for val in coords)
    return coords


def normalize_coords(
    coords: GridCoords,
    ndim: int,
    *,
    shape: Optional[tuple[int, ...]] = None,
) -> tuple[np.ndarray, ...]:
    """Normalize coordinate inputs to a tuple of 1D arrays in axis order.

    Parameters
    ----------
    coords : GridCoords
        Coordinate inputs, provided as a :class:`.Coords`, a mapping (e.g. ``data_array.coords``)
        with keys from ``("x", "y", "z")``, or a tuple/list of arrays in axis order.
    ndim : int
        Number of spatial dimensions expected by the target array.
    shape : Optional[tuple[int, ...]]
        Shape of the target array. When supplied, validates coordinate lengths.
    """
    if ndim < 1 or ndim > len(AXIS_ORDER):
        raise ValueError(f"Expected 1D-3D coordinates, got ndim={ndim}.")

    def _as_array(coord: Iterable[float]) -> np.ndarray:
        if isinstance(coord, Iterator):
            coord = list(coord)
        return np.asarray(coord, dtype=float)

    if isinstance(coords, Coords):
        coord_dict = coords.to_dict
    elif isinstance(coords, Mapping):
        coord_dict = coords
    else:
        coord_list = tuple(coords)
        if len(coord_list) != ndim:
            raise ValueError(
                f"Expected {ndim} coordinate arrays, got {len(coord_list)} for coords."
            )
        coords_tuple = tuple(_as_array(c) for c in coord_list)
        _validate_coords(coords_tuple, shape=shape)
        return coords_tuple

    axes = AXIS_ORDER[:ndim]
    missing = [axis for axis in axes if axis not in coord_dict]
    if missing:
        raise ValueError(f"Missing coordinate arrays for axes {missing}.")
    coords_tuple = tuple(_as_array(coord_dict[axis]) for axis in axes)
    _validate_coords(coords_tuple, shape=shape)
    return coords_tuple


def axis_cell_sizes(coord: np.ndarray) -> np.ndarray:
    """Compute per-coordinate cell sizes from coordinate centers."""
    coord = np.asarray(coord, dtype=float)
    if coord.size > 1:
        diff = coord[1:] - coord[:-1]
        diff_left = np.pad(diff, (1, 0), mode="edge")
        diff_right = np.pad(diff, (0, 1), mode="edge")
        return 0.5 * (diff_left + diff_right)
    return np.ones_like(coord, dtype=float)


def axis_min_spacing(coord: np.ndarray) -> float:
    """Return minimum spacing along an axis (defaults to 1.0 for length-1 axes)."""
    coord = np.asarray(coord, dtype=float)
    if coord.size > 1:
        return float(np.min(coord[1:] - coord[:-1]))
    return 1.0


def cell_sizes_from_coords(coords: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Compute per-axis cell sizes from coordinate arrays."""
    return tuple(axis_cell_sizes(coord) for coord in coords)


def uniform_dls_from_coords(
    coords: tuple[np.ndarray, ...],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> Optional[tuple[float, ...]]:
    """Return per-axis uniform spacing if all coords are uniform, else None."""
    dls = []
    for coord in coords:
        coord = np.asarray(coord, dtype=float)
        if coord.size <= 1:
            dls.append(1.0)
            continue
        diffs = coord[1:] - coord[:-1]
        if np.allclose(diffs, diffs[0], rtol=rtol, atol=atol):
            dls.append(float(diffs[0]))
        else:
            return None
    return tuple(dls)


def _validate_coords(coords: tuple[np.ndarray, ...], shape: Optional[tuple[int, ...]]) -> None:
    for coord in coords:
        if coord.ndim != 1:
            raise ValueError("Coordinate arrays must be 1-dimensional.")
        if coord.size > 1 and np.any(np.diff(coord) <= 0):
            raise ValueError("Coordinate arrays must be strictly increasing.")
    if shape is not None and len(shape) == len(coords):
        for axis, (coord, axis_size) in enumerate(zip(coords, shape)):
            if coord.size != axis_size:
                raise ValueError(
                    f"Coordinate length mismatch on axis {axis}: {coord.size} != {axis_size}."
                )

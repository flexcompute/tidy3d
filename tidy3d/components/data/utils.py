"""Utilites for datasets and dataarrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import xarray as xr

from tidy3d.components.types.base import discriminated_union

from .data_array import SpatialDataArray
from .unstructured.base import UnstructuredGridDataset
from .unstructured.tetrahedral import TetrahedralGridDataset
from .unstructured.triangular import TriangularGridDataset

if TYPE_CHECKING:
    from tidy3d.components.types import ArrayLike

    from .data_array import DataArray

UnstructuredGridDatasetType = Union[TriangularGridDataset, TetrahedralGridDataset]

CustomSpatialDataType = Union[SpatialDataArray, UnstructuredGridDatasetType]
CustomSpatialDataTypeAnnotated = Union[
    discriminated_union(UnstructuredGridDatasetType),
    SpatialDataArray,
]

OUTER_DOT_BLOCK_TARGET_BYTES = 64 * 1024**2
OUTER_DOT_BLOCK_MIN_SIZE = 8
OUTER_DOT_BLOCK_MAX_SIZE = 64


def _instantaneous_power_flow_numpy(
    E: tuple[np.ndarray, np.ndarray],
    H: tuple[np.ndarray, np.ndarray],
    dS: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute instantaneous power flow (Poynting vector integral) for real fields.

    Computes (E x H) integrated over the plane. For time-domain (real) fields.

    Parameters
    ----------
    E : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev), each shape ``(..., nu, nv)``.
    H : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv), each shape ``(..., nu, nv)``.
    dS : tuple[np.ndarray, np.ndarray]
        Area elements at the two Yee grid locations, each shape ``(nu, nv)``.
        ``dS[0]`` is for Eu/Hv location, ``dS[1]`` for Ev/Hu location.

    Returns
    -------
    np.ndarray
        Flux values with shape ``(...)``, spatial dimensions integrated out.
    """
    dS_EuHv, dS_EvHu = dS
    Eu, Ev = E
    Hu, Hv = H

    # Instantaneous Poynting vector: S = E x H (normal component)
    # For tangential fields (Eu, Ev) and (Hu, Hv):
    # Sn = Eu * Hv - Ev * Hu
    # Each term is at a different Yee grid location
    term_EuHv = (Eu * Hv) * dS_EuHv
    term_EvHu = (Ev * Hu) * dS_EvHu

    # Sum over spatial dimensions (last two)
    return np.sum(term_EuHv - term_EvHu, axis=(-2, -1))


def _complex_power_flow_numpy(
    E: tuple[np.ndarray, np.ndarray],
    H: tuple[np.ndarray, np.ndarray],
    dS: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute power flow (Poynting vector integral) with Yee grid area elements.

    Computes 0.5 * (E x H*) integrated over the plane.

    Parameters
    ----------
    E : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev), each shape ``(..., nu, nv)``.
    H : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv), each shape ``(..., nu, nv)``.
    dS : tuple[np.ndarray, np.ndarray]
        Area elements at the two Yee grid locations, each shape ``(nu, nv)``.
        ``dS[0]`` is for Eu/Hv location, ``dS[1]`` for Ev/Hu location.

    Returns
    -------
    np.ndarray
        Flux values with shape ``(...)``, spatial dimensions integrated out.
    """
    dS_EuHv, dS_EvHu = dS
    Eu, Ev = E
    Hu, Hv = H

    # Poynting vector: S = 0.5 E x H* (normal component)
    # For tangential fields (Eu, Ev) and (Hu, Hv):
    # Sn = Eu * Hv* - Ev * Hu*
    # Each term is at a different Yee grid location
    term_EuHv = (Eu * np.conj(Hv)) * dS_EuHv
    term_EvHu = (Ev * np.conj(Hu)) * dS_EvHu

    # Sum over spatial dimensions (last two)
    return 0.5 * np.sum(term_EuHv - term_EvHu, axis=(-2, -1))


def _get_broadcast_selection(
    vals_self: np.ndarray, vals_other: np.ndarray
) -> tuple[np.ndarray | slice, np.ndarray | slice, np.ndarray]:
    """Get selection indices with broadcasting (size-1 broadcasts, else intersection).

    Broadcasting rules (matching develop's xarray behavior):
    - If other has size 1: squeeze it out, use self's coords (broadcast)
    - Otherwise: use intersection

    Returns (sel_self, sel_other, final_vals) where sel can be slice(None) for fast path.
    """
    size_self = len(vals_self)
    size_other = len(vals_other)

    if size_other == 1:
        # Broadcast: other squeezed out, use self's coords
        # Use slices when possible to avoid NumPy advanced indexing dimension collapse
        # when multiple array indices are used together
        if size_self == 1:
            return slice(None), slice(None), vals_self
        return slice(None), np.zeros(size_self, dtype=int), vals_self
    elif np.array_equal(vals_self, vals_other):
        # Fast path: exact match, use slices (creates views, not copies)
        return slice(None), slice(None), vals_self
    else:
        # Intersection
        common, idx_self, idx_other = np.intersect1d(vals_self, vals_other, return_indices=True)
        # Preserve order from self
        order = np.argsort(idx_self)
        return idx_self[order], idx_other[order], common[order]


def _get_intersection_selection(
    vals_self: np.ndarray, vals_other: np.ndarray
) -> tuple[np.ndarray | slice, np.ndarray | slice, np.ndarray]:
    """Get selection indices for intersection only (no broadcasting).

    Returns (sel_self, sel_other, common_vals) where sel can be slice(None) for fast path.
    """
    if np.array_equal(vals_self, vals_other):
        return slice(None), slice(None), vals_self
    else:
        common, idx_self, idx_other = np.intersect1d(vals_self, vals_other, return_indices=True)
        order = np.argsort(idx_self)
        return idx_self[order], idx_other[order], common[order]


def _dot_numpy(
    E1: tuple[np.ndarray, np.ndarray],
    H1: tuple[np.ndarray, np.ndarray],
    E2: tuple[np.ndarray, np.ndarray],
    H2: tuple[np.ndarray, np.ndarray],
    dS: tuple[np.ndarray, np.ndarray],
    conjugate: bool = False,
    bidirectional: bool = True,
) -> np.ndarray:
    """Compute modal overlap integral.

    By default computes the bidirectional overlap: 1/4 * integral(E1* x H2 + H1* x E2) dS.
    With bidirectional=False, computes just: 1/2 * integral(E1* x H2) dS.

    Parameters
    ----------
    E1 : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev) from first dataset, each shape ``(..., nu, nv)``.
    H1 : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv) from first dataset, each shape ``(..., nu, nv)``.
    E2 : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev) from second dataset, each shape ``(..., nu, nv)``.
    H2 : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv) from second dataset, each shape ``(..., nu, nv)``.
    dS : tuple[np.ndarray, np.ndarray]
        Area elements at the two Yee grid locations, each shape ``(nu, nv)``.
        ``dS[0]`` is for Eu/Hv location, ``dS[1]`` for Ev/Hu location.
    conjugate : bool
        If True, conjugate the first set of fields (E1, H1) before computing overlap.
    bidirectional : bool
        If True (default), computes symmetric overlap 1/4 * (E1* x H2 + H1* x E2).
        If False, computes just 1/2 * (E1* x H2).

    Returns
    -------
    np.ndarray
        Overlap values with shape ``(...)``, spatial dimensions integrated out.
    """
    dS_EuHv, dS_EvHu = dS

    E1u, E1v = E1
    H1u, H1v = H1
    E2u, E2v = E2
    H2u, H2v = H2

    dS_EuHv_flat = np.asarray(dS_EuHv).reshape(-1)
    dS_EvHu_flat = np.asarray(dS_EvHu).reshape(-1)

    if bidirectional:
        term_specs = (
            (0.25, E1u, H2v, dS_EuHv_flat),
            (0.25, H1v, E2u, dS_EuHv_flat),
            (-0.25, E1v, H2u, dS_EvHu_flat),
            (-0.25, H1u, E2v, dS_EvHu_flat),
        )
    else:
        term_specs = (
            (0.5, E1u, H2v, dS_EuHv_flat),
            (-0.5, E1v, H2u, dS_EvHu_flat),
        )

    batch_shape = np.broadcast_shapes(
        *(left.shape[:-2] for _, left, _, _ in term_specs),
        *(right.shape[:-2] for _, _, right, _ in term_specs),
    )
    dtype = np.result_type(
        *(a.dtype for _, a, _, _ in term_specs),
        *(a.dtype for _, _, a, _ in term_specs),
        *(d.dtype for _, _, _, d in term_specs),
    )
    result = np.zeros(batch_shape, dtype=dtype)

    for coeff, left, right, d_area in term_specs:
        left_flat = left.reshape(*left.shape[:-2], -1)
        right_flat = right.reshape(*right.shape[:-2], -1)
        if conjugate:
            left_flat = np.conj(left_flat)
        result += (
            coeff
            * ((left_flat * d_area)[..., np.newaxis, :] @ right_flat[..., :, np.newaxis])[..., 0, 0]
        )

    return result


def _outer_dot_numpy(
    E1: tuple[np.ndarray, np.ndarray],
    H1: tuple[np.ndarray, np.ndarray],
    E2: tuple[np.ndarray, np.ndarray],
    H2: tuple[np.ndarray, np.ndarray],
    dS: tuple[np.ndarray, np.ndarray],
    conjugate: bool = False,
    bidirectional: bool = True,
) -> np.ndarray:
    """Compute pairwise modal overlap matrix.

    Computes all elements of the overlap matrix S[i,j] = <mode_i | mode_j>.
    By default computes the bidirectional overlap: 1/4 * integral(E1* x H2 + H1* x E2) dS.
    With bidirectional=False, computes just: 1/2 * integral(E1* x H2) dS.

    Parameters
    ----------
    E1 : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev) from first dataset, each shape ``(..., n_modes_1, nu, nv)``.
        Mode index is at -3, spatial dims at -2, -1.
    H1 : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv) from first dataset, each shape ``(..., n_modes_1, nu, nv)``.
    E2 : tuple[np.ndarray, np.ndarray]
        Tangential E-field components (Eu, Ev) from second dataset, each shape ``(..., n_modes_2, nu, nv)``.
    H2 : tuple[np.ndarray, np.ndarray]
        Tangential H-field components (Hu, Hv) from second dataset, each shape ``(..., n_modes_2, nu, nv)``.
    dS : tuple[np.ndarray, np.ndarray]
        Area elements at the two Yee grid locations, each shape ``(nu, nv)``.
        ``dS[0]`` is for Eu/Hv location, ``dS[1]`` for Ev/Hu location.
    conjugate : bool
        If True, conjugate the first set of fields (E1, H1) before computing overlap.
    bidirectional : bool
        If True (default), computes symmetric overlap 1/4 * (E1* x H2 + H1* x E2).
        If False, computes just 1/2 * (E1* x H2).

    Returns
    -------
    np.ndarray
        Overlap matrix with shape ``(..., n_modes_1, n_modes_2)``.
    """
    E1u, E1v = E1
    H1u, H1v = H1
    E2u, E2v = E2
    H2u, H2v = H2

    dS_EuHv, dS_EvHu = dS

    # Get number of modes and broadcast shape
    n_modes_1 = E1u.shape[-3]
    n_modes_2 = E2u.shape[-3]
    broadcast_shape = E1u.shape[:-3]
    num_grid_points = E1u.shape[-2] * E1u.shape[-1]
    dtype = np.result_type(
        E1u.dtype,
        E1v.dtype,
        H1u.dtype,
        H1v.dtype,
        E2u.dtype,
        E2v.dtype,
        H2u.dtype,
        H2v.dtype,
        np.asarray(dS_EuHv).dtype,
        np.asarray(dS_EvHu).dtype,
    )

    # Initialize output matrix
    S = np.zeros((*broadcast_shape, n_modes_1, n_modes_2), dtype=dtype)

    if n_modes_1 == 0 or n_modes_2 == 0:
        return S

    # Conjugate outside loop to avoid repeated copies
    if conjugate:
        E1u, E1v = np.conj(E1u), np.conj(E1v)
        H1u, H1v = np.conj(H1u), np.conj(H1v)

    # Heuristic: choose mode block size targeting bounded temporary allocations.
    itemsize = np.dtype(dtype).itemsize
    if num_grid_points == 0:
        block_size = OUTER_DOT_BLOCK_MAX_SIZE
    else:
        block_size = int(OUTER_DOT_BLOCK_TARGET_BYTES // (num_grid_points * itemsize))
        block_size = max(OUTER_DOT_BLOCK_MIN_SIZE, block_size)
        block_size = min(OUTER_DOT_BLOCK_MAX_SIZE, block_size)

    block_size_left = min(n_modes_1, block_size)
    block_size_right = min(n_modes_2, block_size)

    dS_EuHv_flat = np.asarray(dS_EuHv).reshape(-1)
    dS_EvHu_flat = np.asarray(dS_EvHu).reshape(-1)
    if dS_EuHv_flat.size != num_grid_points or dS_EvHu_flat.size != num_grid_points:
        raise ValueError("Tangential area shape mismatch in blocked outer_dot kernel.")

    num_batches = int(np.prod(broadcast_shape, dtype=int)) if broadcast_shape else 1
    out_flat = S.reshape(num_batches, n_modes_1, n_modes_2)

    if bidirectional:
        term_specs = (
            (0.25, E1u, H2v, dS_EuHv_flat),
            (0.25, H1v, E2u, dS_EuHv_flat),
            (-0.25, E1v, H2u, dS_EvHu_flat),
            (-0.25, H1u, E2v, dS_EvHu_flat),
        )
    else:
        term_specs = (
            (0.5, E1u, H2v, dS_EuHv_flat),
            (-0.5, E1v, H2u, dS_EvHu_flat),
        )

    # Flatten once outside loops so each block only performs views + BLAS matmul.
    flattened_terms = tuple(
        (
            coeff,
            left.reshape(num_batches, n_modes_1, num_grid_points),
            right.reshape(num_batches, n_modes_2, num_grid_points),
            d_area,
        )
        for coeff, left, right, d_area in term_specs
    )

    for batch_idx in range(num_batches):
        out_batch = out_flat[batch_idx]
        for coeff, left_term, right_term, d_area in flattened_terms:
            left_batch = left_term[batch_idx]
            right_batch = right_term[batch_idx]
            for i0 in range(0, n_modes_1, block_size_left):
                i0_end = min(i0 + block_size_left, n_modes_1)
                left_block = left_batch[i0:i0_end]
                for i1 in range(0, n_modes_2, block_size_right):
                    i1_end = min(i1 + block_size_right, n_modes_2)
                    right_block = right_batch[i1:i1_end]
                    weighted_right = right_block * d_area
                    out_batch[i0:i0_end, i1:i1_end] += coeff * (left_block @ weighted_right.T)

    return S


def _get_numpy_array(data_array: Union[ArrayLike, DataArray, UnstructuredGridDataset]) -> ArrayLike:
    """Get numpy representation of dataarray/dataset values."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.values.values
    if isinstance(data_array, xr.DataArray):
        return data_array.values
    return np.array(data_array)


def _zeros_like(
    data_array: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> Union[ArrayLike, xr.DataArray, UnstructuredGridDataset]:
    """Get a zeroed replica of dataarray/dataset."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.updated_copy(values=xr.zeros_like(data_array.values))
    if isinstance(data_array, xr.DataArray):
        return xr.zeros_like(data_array)
    return np.zeros_like(data_array)


def _ones_like(
    data_array: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> Union[ArrayLike, xr.DataArray, UnstructuredGridDataset]:
    """Get a unity replica of dataarray/dataset."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.updated_copy(values=xr.ones_like(data_array.values))
    if isinstance(data_array, xr.DataArray):
        return xr.ones_like(data_array)
    return np.ones_like(data_array)


def _check_same_coordinates(
    a: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
    b: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> bool:
    """Check whether two array are defined at the same coordinates."""

    # we can have xarray.DataArray's of different types but still same coordinates
    # we will deal with that case separately
    both_xarrays = isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray)
    if (not both_xarrays) and type(a) is not type(b):
        return False

    if isinstance(a, UnstructuredGridDataset):
        if not np.allclose(a.points, b.points) or not np.all(a.cells == b.cells):
            return False

        if isinstance(a, TriangularGridDataset):
            if a.normal_axis != b.normal_axis or a.normal_pos != b.normal_pos:
                return False

    elif isinstance(a, xr.DataArray):
        if a.coords.keys() != b.coords.keys() or a.coords != b.coords:
            return False

    else:
        if np.shape(a) != np.shape(b):
            return False

    return True

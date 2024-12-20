"""Utilites for datasets and dataarrays."""

from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr

from ..types import ArrayLike, annotate_type
from .data_array import DataArray, SpatialDataArray
from .unstructured.base import UnstructuredGridDataset
from .unstructured.tetrahedral import TetrahedralGridDataset
from .unstructured.triangular import TriangularGridDataset

UnstructuredGridDatasetType = Union[TriangularGridDataset, TetrahedralGridDataset]

CustomSpatialDataType = Union[SpatialDataArray, UnstructuredGridDatasetType]
CustomSpatialDataTypeAnnotated = Union[SpatialDataArray, annotate_type(UnstructuredGridDatasetType)]


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

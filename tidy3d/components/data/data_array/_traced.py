"""Dataset that keeps traced data arrays intact on item access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from .base import DataArray

if TYPE_CHECKING:
    from collections.abc import Hashable


class _TracedDataset(xr.Dataset):
    """Dataset subclass that preserves traced tidy3d DataArrays when accessed.

    When xr.Dataset constructor is called with tidy3d DataArray objects,
    xarray extracts the data and stores it as Variables internally. When
    items are accessed via __getitem__, xarray wraps these Variables in
    vanilla xr.DataArray, losing the custom .values property that's needed
    for autograd compatibility.

    This subclass overrides _construct_dataarray to return tidy3d DataArray
    objects, preserving the custom .values behavior that returns .data
    directly when tracing (avoiding np.asarray which breaks autodiff).
    """

    __slots__ = ()

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> _TracedDataset:
        """Construct a traced dataset from an existing Dataset instance."""
        return cls._construct_direct(
            variables=dataset._variables.copy(),
            coord_names=dataset._coord_names.copy(),
            dims=dict(dataset._dims),
            attrs=None if dataset._attrs is None else dataset._attrs.copy(),
            indexes=dataset._indexes.copy(),
            encoding=None if dataset._encoding is None else dataset._encoding.copy(),
            close=dataset._close,
        )

    def _construct_dataarray(self, name: Hashable) -> DataArray:
        """Construct a tidy3d DataArray by indexing this dataset."""
        return DataArray(super()._construct_dataarray(name))

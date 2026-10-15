"""HDF5 read and write methods bound onto ``DataArray``."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np

from tidy3d.components.autograd import get_static
from tidy3d.exceptions import FileError

from ._constants import DATA_ARRAY_VALUE_NAME, HDF5_SHUFFLE_MIN_BYTES

if TYPE_CHECKING:
    from os import PathLike

    from .base import DataArray

# Stands for the class a method is bound to, as 'Self' did when these were methods of 'DataArray'.
_DataArrayT = TypeVar("_DataArrayT", bound="DataArray")


def to_hdf5(self: DataArray, fname: PathLike | h5py.File, group_path: str) -> None:
    """Save an ``xr.DataArray`` to the hdf5 file or file handle with a given path to the group."""
    if isinstance(fname, str | pathlib.Path):
        path = pathlib.Path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f_handle:
            self.to_hdf5_handle(f_handle=f_handle, group_path=group_path)
    else:
        self.to_hdf5_handle(f_handle=fname, group_path=group_path)


def to_hdf5_handle(self: DataArray, f_handle: h5py.File, group_path: str) -> None:
    """Save an ``xr.DataArray`` to the hdf5 file handle with a given path to the group."""
    sub_group = f_handle.create_group(group_path)
    values = get_static(self.data)
    sub_group.create_dataset(
        DATA_ARRAY_VALUE_NAME,
        data=values,
        shuffle=values.nbytes >= HDF5_SHUFFLE_MIN_BYTES,
    )
    for key in self._dims:
        if key not in self.coords:
            continue
        val = self.coords[key]
        if val.dtype.kind == "U":
            # Convert Unicode strings to list for HDF5 storage
            sub_group[key] = val.values.tolist()
        else:
            sub_group[key] = val


def _from_hdf5_handle(cls: type[_DataArrayT], f_handle: h5py.File, group_path: str) -> _DataArrayT:
    """Load a DataArray from an open hdf5 file handle with a given group path."""
    sub_group = f_handle[group_path]
    values = np.array(sub_group[DATA_ARRAY_VALUE_NAME])
    dims = cls._dims
    if cls._legacy_dims is not None and values.ndim == len(cls._legacy_dims):
        dims = cls._legacy_dims
    coords = {dim: np.array(sub_group[dim]) for dim in dims if dim in sub_group}
    for key, val in coords.items():
        if val.dtype == "O":
            coords[key] = [byte_string.decode() for byte_string in val.tolist()]
    data_array = cls(values, coords=coords, dims=dims)
    return cls._validate_dims(data_array)


def from_hdf5(cls: type[_DataArrayT], fname: PathLike | h5py.File, group_path: str) -> _DataArrayT:
    """Load a DataArray from an hdf5 file or open file handle with a given group path."""
    if isinstance(fname, h5py.File):
        return cls._from_hdf5_handle(f_handle=fname, group_path=group_path)

    path = pathlib.Path(fname)
    with h5py.File(path, "r") as f_handle:
        return cls._from_hdf5_handle(f_handle=f_handle, group_path=group_path)


def from_file(cls: type[_DataArrayT], fname: PathLike, group_path: str) -> _DataArrayT:
    """Load a DataArray from an hdf5 file with a given path to the group."""
    path = pathlib.Path(fname)
    if not any(suffix.lower() == ".hdf5" for suffix in path.suffixes):
        raise FileError(
            f"'DataArray' objects must be written to '.hdf5' format. Given filename of {path}."
        )
    return cls.from_hdf5(fname=path, group_path=group_path)

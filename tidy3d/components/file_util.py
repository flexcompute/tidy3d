"""File compression utilities"""

from __future__ import annotations

import gzip
import pathlib
import shutil
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from io import BytesIO
    from os import PathLike

JSON_TAG = "JSON_STRING"


def json_string_key(index: int) -> str:
    """Get JSON string dataset key for chunk number ``index``."""
    if index:
        return f"{JSON_TAG}_{index}"
    return JSON_TAG


def json_string_from_hdf5(fname: PathLike | h5py.File) -> bytes:
    """Load the model JSON string from an HDF5 file path or open file handle."""
    if isinstance(fname, h5py.File):
        f_handle = fname
        num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
        json_string = b""
        for ind in range(num_string_parts):
            json_string += f_handle[json_string_key(ind)][()]
        return json_string

    with h5py.File(fname, "r") as f_handle:
        return json_string_from_hdf5(f_handle)


def compress_file_to_gzip(input_file: PathLike, output_gz_file: PathLike | BytesIO) -> None:
    """
    Compress a file using gzip.

    Parameters
    ----------
    input_file : PathLike
        The path to the input file.
    output_gz_file : PathLike | BytesIO
        The path to the output gzip file or an in-memory buffer.
    """
    input_file = pathlib.Path(input_file)
    with input_file.open("rb") as file_in:
        with gzip.open(output_gz_file, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def extract_gzip_file(input_gz_file: PathLike, output_file: PathLike) -> None:
    """
    Extract a gzip-compressed file.

    Parameters
    ----------
    input_gz_file : PathLike
        The path to the gzip-compressed input file.
    output_file : PathLike
        The path to the extracted output file.
    """
    input_path = pathlib.Path(input_gz_file)
    output_path = pathlib.Path(output_file)
    with gzip.open(input_path, "rb") as file_in:
        with output_path.open("wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def replace_values(values: Any, search_value: Any, replace_value: Any) -> Any:
    """
    Create a copy of ``values`` where any elements equal to ``search_value`` are replaced by ``replace_value``.

    Parameters
    ----------
    values : Any
        The input object to iterate through.
    search_value : Any
        An object to match for in ``values``.
    replace_value : Any
        A replacement object for the matched value in ``values``.

    Returns
    -------
    Any
        values type object with ``search_value`` terms replaced by ``replace_value``.
    """
    # np.all allows for arrays to be evaluated
    if np.all(values == search_value):
        return replace_value
    if isinstance(values, dict):
        return {
            key: replace_values(val, search_value, replace_value) for key, val in values.items()
        }
    if isinstance(
        values, (tuple, list)
    ):  # Parts of the nested dict structure include tuples with more dicts
        return type(values)(replace_values(val, search_value, replace_value) for val in values)

    # Used to maintain values that are not search_value or containers
    return values

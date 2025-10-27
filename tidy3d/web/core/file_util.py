"""File compression utilities"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile

import h5py

from tidy3d.web.core.constants import JSON_TAG


def compress_file_to_gzip(input_file: os.PathLike, output_gz_file: os.PathLike) -> None:
    """Compresses a file using gzip.

    Parameters
    ----------
    input_file : PathLike
        The path of the input file.
    output_gz_file : PathLike
        The path of the output gzip file.
    """
    with open(input_file, "rb") as file_in:
        with gzip.open(output_gz_file, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def extract_gzip_file(input_gz_file: os.PathLike, output_file: os.PathLike) -> None:
    """Extract a gzip file.

    Parameters
    ----------
    input_gz_file : PathLike
        The path of the gzip input file.
    output_file : PathLike
        The path of the output file.
    """
    with gzip.open(input_gz_file, "rb") as file_in:
        with open(output_file, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def read_simulation_from_hdf5_gz(file_name: os.PathLike) -> str:
    """read simulation str from hdf5.gz"""

    hdf5_file, hdf5_file_path = tempfile.mkstemp(".hdf5")
    os.close(hdf5_file)
    try:
        extract_gzip_file(file_name, hdf5_file_path)
        # Pass the uncompressed temporary file path to the reader
        json_str = read_simulation_from_hdf5(hdf5_file_path)
    finally:
        os.unlink(hdf5_file_path)
    return json_str


"""TODO: _json_string_key and read_simulation_from_hdf5 are duplicated functions that also exist
as methods in Tidy3dBaseModel. For consistency it would be best if this duplication is avoided."""


def _json_string_key(index: int) -> str:
    """Get json string key for string chunk number ``index``."""
    if index:
        return f"{JSON_TAG}_{index}"
    return JSON_TAG


def read_simulation_from_hdf5(file_name: os.PathLike) -> bytes:
    """read simulation str from hdf5"""
    with h5py.File(file_name, "r") as f_handle:
        num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
        json_string = b""
        for ind in range(num_string_parts):
            json_string += f_handle[_json_string_key(ind)][()]
    return json_string


"""End TODO"""


def read_simulation_from_json(file_name: os.PathLike) -> str:
    """read simulation str from json"""
    with open(file_name) as json_file:
        json_data = json_file.read()
    return json_data

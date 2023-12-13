"""File compression utilities"""

import gzip
import os
import shutil
import tempfile

import h5py

from ..core.constants import JSON_TAG


def compress_file_to_gzip(input_file, output_gz_file):
    """
    Compresses a file using gzip.

    Args:
        input_file (str): The path of the input file.
        output_gz_file (str): The path of the output gzip file.
    """
    with open(input_file, "rb") as file_in:
        with gzip.open(output_gz_file, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def extract_gzip_file(input_gz_file, output_file):
    """
    Extract a gzip file.

    Args:
        input_gz_file (str): The path of the gzip input file.
        output_file (str): The path of the output file.
    """
    with gzip.open(input_gz_file, "rb") as file_in:
        with open(output_file, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)


def read_simulation_from_hdf5_gz(file_name: str) -> str:
    """read simulation str from hdf5.gz"""

    hdf5_file, hdf5_file_path = tempfile.mkstemp(".hdf5")
    os.close(hdf5_file_path)
    try:
        extract_gzip_file(file_name, hdf5_file_path)
        json_str = read_simulation_from_hdf5(file_name)
    finally:
        os.unlink(hdf5_file_path)
    return json_str


"""TODO: _json_string_key and read_simulation_from_hdf5 are duplicated functions that also exist
as methods in Tidy3dBaseModel. For consistency it would be best if this duplication is avoided."""


def _json_string_key(index):
    """Get json string key for string chunk number ``index``."""
    if index:
        return f"{JSON_TAG}_{index}"
    return JSON_TAG


def read_simulation_from_hdf5(file_name: str) -> str:
    """read simulation str from hdf5"""
    with h5py.File(file_name, "r") as f_handle:
        num_string_parts = len([key for key in f_handle.keys() if JSON_TAG in key])
        json_string = b""
        for ind in range(num_string_parts):
            json_string += f_handle[_json_string_key(ind)][()]
    return json_string


"""End TODO"""


def read_simulation_from_json(file_name: str) -> str:
    """read simulation str from json"""
    with open(file_name) as json_file:
        json_data = json_file.read()
    return json_data

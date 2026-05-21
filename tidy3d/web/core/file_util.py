"""File compression utilities"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile

from tidy3d.components.file_util import json_string_from_hdf5


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


def read_simulation_from_hdf5(file_name: os.PathLike) -> bytes:
    """read simulation str from hdf5"""
    return json_string_from_hdf5(file_name)


def read_simulation_from_json(file_name: os.PathLike) -> str:
    """read simulation str from json"""
    with open(file_name, encoding="utf-8") as json_file:
        json_data = json_file.read()
    return json_data

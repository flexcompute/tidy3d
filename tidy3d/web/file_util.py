"""compress and extract file"""

import gzip

import h5py

from tidy3d.components.base import JSON_TAG


def compress_file_to_gzip(input_file, output_gz_file):
    """
    Compresses a file using gzip.

    Args:
        input_file (str): The path of the input file.
        output_gz_file (str): The path of the output gzip file.
    """
    with open(input_file, "rb") as file_in:
        with gzip.open(output_gz_file, "wb") as file_out:
            file_out.writelines(file_in)


def extract_gz_file(input_gz_file, output_file):
    """
    Extract the GZ file

    Args:
        input_gz_file (str): The path of the gzip input file.
        output_file (str): The path of the output file.
    """
    with gzip.open(input_gz_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            f_out.write(f_in.read())


def read_simulation_from_hdf5(file_name: str):
    """read simulation str from hdf5"""

    with h5py.File(file_name, "r") as f_handle:
        json_string = f_handle[JSON_TAG][()]
        return json_string

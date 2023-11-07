"""File compression utilities"""

import gzip
import shutil


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

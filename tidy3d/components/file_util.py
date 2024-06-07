"""File compression utilities"""

import gzip
import shutil
from typing import Any

import numpy as np


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
    elif isinstance(
        values, (tuple, list)
    ):  # Parts of the nested dict structure include tuples with more dicts
        return type(values)(replace_values(val, search_value, replace_value) for val in values)

    # Used to maintain values that are not search_value or containers
    return values

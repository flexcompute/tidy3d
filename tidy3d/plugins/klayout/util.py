from __future__ import annotations

from shutil import which
from typing import Union

import tidy3d as td


def check_installation(raise_error: bool = False) -> Union[str, None]:
    """Checks if KLayout is installed and added to the system PATH.
    If it is, this returns the path to the executable. Otherwise, returns ``None``.
    Equivalent to $which("klayout") in the terminal.

    Parameters
    ----------
    raise_error : bool
        Whether to raise an error if KLayout is not found. If ``False``, a warning is shown.

    Returns
    -------
    Union[str, None]
        The path to the KLayout executable. If KLayout is not found, returns ``None``.
    """
    path = which("klayout")
    msg = "KLayout was not found in the system PATH. Please ensure KLayout is installed and added to your system PATH before running KLayout."
    if path is None:
        if raise_error:
            raise RuntimeError(msg)
        else:
            td.log.warning(msg)
    return path

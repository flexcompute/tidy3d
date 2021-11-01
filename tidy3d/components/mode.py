"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Symmetry


class Mode(Tidy3dBaseModel):
    """Stores Specifications of a Mode to input into mode solver.
    Parameters
    ----------
    mode_index : ``int``
        Return the mode solver output at ``mode_index``.
    target_neff : ``float = None``
        Guess for effective index of mode.
    symmetries : ``(int, int) = (0,0)``
        Symmetries (0, 1,-1) = (none, even, odd) in (x,y) of mode plane.
    num_pml: ``(int, int) = (0,0)``
        number of standard pml layers to add in (x,y) of mode plane.
    """

    mode_index: pd.NonNegativeInt
    num_modes: pd.PositiveInt = None
    target_neff: float = None
    symmetries: Tuple[Symmetry, Symmetry] = (0, 0)
    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = (0, 0)

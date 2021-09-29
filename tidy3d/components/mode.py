""" Defines specification for mode solver """

from typing import Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Symmetry


class Mode(Tidy3dBaseModel):
    """defines mode specification"""

    mode_index: pd.NonNegativeInt
    target_neff: float = None
    symmetries: Tuple[Symmetry, Symmetry] = (0, 0)
    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = (0, 0)

"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Symmetry


class ModeSpec(Tidy3dBaseModel):
    """Stores specifications for the mode solver to find an electromagntic mode.
    Note, the planar axes are found by popping the propagation axis from {x,y,z}.
    For example, if propagation axis is y, the planar axes are ordered {x,z}.

    Parameters
    ----------

    num_modes : int = 1
        Number of modes returned by mode solver.
    target_neff : float = None
        Guess for effective index of mode.
        Must be > 0.
    symmetries : Tuple[int, int] = (0,0)
        Symmetries to apply to mode solver for first two non-propagation axes.
        Values of (0, 1,-1) correspond to (none, even, odd) symmetries, respectvely.
    num_pml: Tuple[int, int] = (0,0)
        Number of standard pml layers to add in the first two non-propagation axes.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3, target_neff=1.5, symmetries=(1,-1))
    """

    num_modes: pd.PositiveInt = 1
    target_neff: pd.PositiveFloat = None
    symmetries: Tuple[Symmetry, Symmetry] = (0, 0)
    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = (0, 0)

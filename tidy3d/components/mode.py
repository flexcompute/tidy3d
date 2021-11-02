"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Symmetry
from ..log import SetupError


class Mode(Tidy3dBaseModel):
    """Stores specifications for the mode solver to find an electromagntic mode.
    Note, the planar axes are found by popping the propagation axis from {x,y,z}.
    For example, if propagation axis is y, the planar axes are ordered {x,z}.

    Parameters
    ----------
    mode_index : int
        Return the mode solver output at ``mode_index``.
        Must be >= 0.
    num_modes : int = None
        Number of modes returned by mode solver before selecting mode at ``mode_index``.
        Must be > ``mode_index`` to accomodate ``mode_index``-th mode.
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
    >>> mode = Mode(mode_index=1, num_modes=3, target_neff=1.5, symmetries=(1,-1))
    """

    mode_index: pd.NonNegativeInt
    num_modes: pd.PositiveInt = None
    target_neff: pd.PositiveFloat = None
    symmetries: Tuple[Symmetry, Symmetry] = (0, 0)
    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = (0, 0)

    @pd.validator("num_modes", always=True)
    def check_num_modes(cls, val, values):
        """Make sure num_modes is > mode_index or None"""
        if val is not None:
            mode_index = values.get("mode_index")
            if not val > mode_index:
                raise SetupError(
                    "`num_modes` must be greater than `mode_index`"
                    f"given {val} and {mode_index}, respectively"
                )
        return val

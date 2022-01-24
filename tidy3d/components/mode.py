"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Symmetry


class ModeSpec(Tidy3dBaseModel):
    """Stores specifications for the mode solver to find an electromagntic mode.
    Note, the planar axes are found by popping the propagation axis from {x,y,z}.
    For example, if propagation axis is y, the planar axes are ordered {x,z}.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3, target_neff=1.5, symmetries=(1,-1))
    """

    num_modes: pd.PositiveInt = pd.Field(
        1, title="Number of modes", description="Number of modes returned by mode solver."
    )

    target_neff: pd.PositiveFloat = pd.Field(
        None, title="Target effective index", description="Guess for effective index of the mode."
    )

    symmetries: Tuple[Symmetry, Symmetry] = pd.Field(
        (0, 0),
        title="Tangential symmetries",
        description="Symmetries to apply to mode solver for first two non-propagation axes.  "
        "Values of (0, 1,-1) correspond to (none, even, odd) symmetries, respectvely.",
    )

    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = pd.Field(
        (0, 0),
        title="Number of PML layers",
        description="Number of standard pml layers to add in the first two non-propagation axes.",
    )

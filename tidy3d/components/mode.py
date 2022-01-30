"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd

from ..constants import MICROMETER
from .base import Tidy3dBaseModel
from .types import Symmetry, Axis2D
from ..log import SetupError


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

    bend_radius: float = pd.Field(
        None,
        title="Bend radius",
        description="A curvature radius for simulation of waveguide bends. Can be negative, in "
        "which case the mode plane center has a smaller value than the curvature center along the "
        "axis that is perpendicular to both the normal axis and the bend axis.",
        units=MICROMETER,
    )

    bend_axis: Axis2D = pd.Field(
        None,
        title="Bend axis",
        description="Index into the first two non-propagating axes defining the normal to the "
        "plane in which the bend lies. This must be provided if ``bend_radius`` is not ``None``.",
    )

    @pd.validator("bend_axis", always=True)
    def bend_axis_given(cls, val, values):
        """check that ``bend_axis`` is provided if ``bend_radius`` is not ``None``"""
        if val is None and values.get("bend_radius") is not None:
            raise SetupError("bend_axis must also be defined if bend_radius is defined.")
        return val

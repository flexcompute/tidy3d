"""Defines specification for mode solver."""

from typing import Tuple

import pydantic as pd
import numpy as np

from ..constants import MICROMETER, RADIAN, GLANCING_CUTOFF
from .base import Tidy3dBaseModel
from .types import Axis2D, Literal
from ..log import SetupError


class ModeSpec(Tidy3dBaseModel):
    """Stores specifications for the mode solver to find an electromagntic mode.
    Note, the planar axes are found by popping the injection axis from {x,y,z}.
    For example, if injection axis is y, the planar axes are ordered {x,z}.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3, target_neff=1.5)
    """

    num_modes: pd.PositiveInt = pd.Field(
        1, title="Number of modes", description="Number of modes returned by mode solver."
    )

    target_neff: pd.PositiveFloat = pd.Field(
        None, title="Target effective index", description="Guess for effective index of the mode."
    )

    num_pml: Tuple[pd.NonNegativeInt, pd.NonNegativeInt] = pd.Field(
        (0, 0),
        title="Number of PML layers",
        description="Number of standard pml layers to add in the two tangential axes.",
    )

    sort_by: Literal["largest_neff", "te_fraction", "tm_fraction"] = pd.Field(
        "largest_neff",
        title="Ordering of the returned modes",
        description="The solver will always compute the ``num_modes`` modes closest to the "
        "``target_neff``, but they can be reordered by the largest ``te_fraction``, defined "
        "as the integral of the intensity of the E-field component parallel to the first plane "
        "axis normalized to the total in-plane E-field intensity. Similarly, ``tm_fraction`` "
        "uses the E field component parallel to the second plane axis.",
    )

    angle_theta: float = pd.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        units=RADIAN,
    )

    angle_phi: float = pd.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        units=RADIAN,
    )

    bend_radius: float = pd.Field(
        None,
        title="Bend radius",
        description="A curvature radius for simulation of waveguide bends. Can be negative, in "
        "which case the mode plane center has a smaller value than the curvature center along the "
        "tangential axis perpendicular to the bend axis.",
        units=MICROMETER,
    )

    bend_axis: Axis2D = pd.Field(
        None,
        title="Bend axis",
        description="Index into the two tangential axes defining the normal to the "
        "plane in which the bend lies. This must be provided if ``bend_radius`` is not ``None``. "
        "For example, for a ring in the global xy-plane, and a mode plane in either the xz or the "
        "yz plane, the ``bend_axis`` is always 1 (the global z axis).",
    )

    @pd.validator("bend_axis", always=True)
    def bend_axis_given(cls, val, values):
        """check that ``bend_axis`` is provided if ``bend_radius`` is not ``None``"""
        if val is None and values.get("bend_radius") is not None:
            raise SetupError("bend_axis must also be defined if bend_radius is defined.")
        return val

    @pd.validator("angle_theta", allow_reuse=True, always=True)
    def glancing_incidence(cls, val):
        """Warn if close to glancing incidence."""
        if np.abs(np.pi / 2 - val) < GLANCING_CUTOFF:
            raise SetupError(
                "Mode propagation axis too close to glancing angle for accurate injection. "
                "For best results, switch the injection axis."
            )
        return val

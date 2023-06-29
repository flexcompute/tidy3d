"""Defines specification for mode solver."""

from typing import Tuple, Union

import pydantic as pd
import numpy as np

from ..constants import MICROMETER, RADIAN, GLANCING_CUTOFF
from .base import Tidy3dBaseModel
from .types import Axis2D, Literal, TrackFreq
from ..log import log
from ..exceptions import SetupError, ValidationError


GROUP_INDEX_STEP = 0.005


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

    filter_pol: Literal["te", "tm"] = pd.Field(
        None,
        title="Polarization filtering",
        description="The solver always computes the ``num_modes`` modes closest to the given "
        "``target_neff``. If ``filter_pol==None``, they are simply sorted in order of decresing "
        "effective index. If a polarization filter is selected, the modes are rearranged such that "
        "the first ``n_pol`` modes in the list are the ones with the selected polarization "
        "fraction larger than or equal to 0.5, while the next ``num_modes - n_pol`` modes are the "
        "ones where it is smaller than 0.5 (i.e. the opposite polarization fraction is larger than "
        "0.5). Within each polarization subset, the modes are still ordered by decreasing "
        "effective index. "
        "``te``-fraction is defined as the integrated intensity of the E-field component parallel "
        "to the first plane axis, normalized to the total in-plane E-field intensity. Conversely, "
        "``tm``-fraction uses the E field component parallel to the second plane axis.",
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

    precision: Literal["single", "double"] = pd.Field(
        "single",
        title="single or double precision in mode solver",
        description="The solver will be faster and using less memory under "
        "single precision, but more accurate under double precision.",
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

    track_freq: Union[TrackFreq, None] = pd.Field(
        "central",
        title="Mode Tracking Frequency",
        description="Parameter that turns on/off mode tracking based on their similarity. "
        "Can take values ``'lowest'``, ``'central'``, or ``'highest'``, which correspond to "
        "mode tracking based on the lowest, central, or highest frequency. "
        "If ``None`` no mode tracking is performed.",
    )

    group_index_step: Union[bool, pd.PositiveFloat] = pd.Field(
        False,
        title="Frequency step for group index computation",
        description="Control the computation of the group index alongside the effective index. If "
        "set to a positive value, it sets the fractional frequency step used in the numerical "
        "differentiation of the effective index to compute the group index. If set to `True`, the "
        f"default of {GROUP_INDEX_STEP} is used.",
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

    @pd.validator("group_index_step")
    def assign_default_on_true(cls, val):
        """Assing the default fractional frequency step value if not provided."""
        if val is True:
            return GROUP_INDEX_STEP
        if val >= 1:
            raise ValidationError("Parameter 'group_index_step' must be less than 1.")
        return val

    @pd.root_validator()
    def check_precision(cls, values):
        """Verify critical ModeSpec settings for group index calculation."""
        if values["group_index_step"] > 0:
            if values["track_freq"] is None:
                log.warning(
                    "Group index calculation without mode tracking can lead to incorrect results "
                    "around mode crossings. Consider setting 'track_freq' to 'central'."
                )
        return values

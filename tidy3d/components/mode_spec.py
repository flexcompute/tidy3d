"""Defines specification for mode solver."""

from math import isclose
from typing import Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ..constants import GLANCING_CUTOFF, MICROMETER, RADIAN, fp_eps
from ..exceptions import SetupError, ValidationError
from ..log import log
from .base import Tidy3dBaseModel, skip_if_fields_missing
from .types import Axis2D, Literal, TrackFreq

GROUP_INDEX_STEP = 0.005


class ModeSpec(Tidy3dBaseModel):
    """
    Stores specifications for the mode solver to find an electromagntic mode.

    Notes
    -----

        The :attr:`angle_theta` and :attr:`angle_phi` parameters define the injection axis as illustrated in the figure
        below, with respect to the axis normal to the mode plane (``x`` in the figure). Note that :attr:`angle_theta`
        must be smaller than :math:`\\frac{pi}{2}`. To inject in the backward direction, we can still use the
        ``direction`` parameter as also shown in the figure. Similarly, the mode amplitudes computed in mode monitors
        are defined w.r.t. the ``forward`` and ``backward`` directions as illustrated. Note, the planar axes are
        found by popping the injection axis from ``{x,y,z}``. For example, if injection axis is ``y``, the planar
        axes are ordered ``{x,z}``.

        .. image:: ../../notebooks/img/ring_modes.png

        The :attr:`bend_axis` is the axis normal to the plane in which the bend lies, (``z`` in the diagram below). In
        the mode specification, it is defined locally for the mode plane as one of the two axes tangential to the
        plane. In the case of bends that lie in the ``xy``-plane, the mode plane would be either in ``xz`` or in
        ``yz``, so in both cases the correct setting is ``bend_axis=1``, selecting the global ``z``. The
        ``bend_radius`` is counted from the center of the mode plane to the center of the curvature,
        along the tangential axis perpendicular to the bend axis. This radius can also be negative, if the center of
        the mode plane is smaller than the center of the bend.

        .. image:: ../../notebooks/img/mode_angled.png

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3, target_neff=1.5)

    See Also
    --------

    **Notebooks**:
        * `Introduction on tidy3d working principles <../../notebooks/Primer.html#Modes>`_
        * `Defining mode sources and monitors <../../notebooks/ModalSourcesMonitors.html>`_
        * `Injecting modes in bent and angled waveguides <../../notebooks/ModesBentAngled.html>`_
        * `Waveguide to ring coupling <../../notebooks/WaveguideToRingCoupling.html>`_

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
        "``target_neff``. If ``filter_pol==None``, they are simply sorted in order of decreasing "
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

    precision: Literal["auto", "single", "double"] = pd.Field(
        "auto",
        title="single, double, or automatic precision in mode solver",
        description="The solver will be faster and using less memory under "
        "single precision, but more accurate under double precision. "
        "With the default ``'auto'``, apply double precision if the simulation contains a good "
        "conductor, single precision otherwise.",
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

    angle_rotation: bool = pd.Field(
        False,
        title="Use fields rotation when angle_theta is not zero",
        description="Defines how modes are computed when angle_theta is not zero. "
        "If 'False', a coordinate transformation is applied through the permittivity and permeability tensors."
        "If 'True', the structures in the simulation are first rotated to compute a mode solution at "
        "a reference plane normal to the structure's azimuthal direction. Then, the fields are rotated "
        "to align with the mode plane, using the 'n_eff' calculated at the reference plane. The second option can "
        "produce more accurate results, but more care must be taken, for example, in ensuring that the "
        "original mode plane intersects the correct geometries in the simulation with rotated structures. "
        "Note: currently only supported when 'angle_phi' is a multiple of 'np.pi'.",
    )

    track_freq: Union[TrackFreq, None] = pd.Field(
        "central",
        title="Mode Tracking Frequency",
        description="Parameter that turns on/off mode tracking based on their similarity. "
        "Can take values ``'lowest'``, ``'central'``, or ``'highest'``, which correspond to "
        "mode tracking based on the lowest, central, or highest frequency. "
        "If ``None`` no mode tracking is performed.",
    )

    group_index_step: Union[pd.PositiveFloat, bool] = pd.Field(
        False,
        title="Frequency step for group index computation",
        description="Control the computation of the group index alongside the effective index. If "
        "set to a positive value, it sets the fractional frequency step used in the numerical "
        "differentiation of the effective index to compute the group index. If set to `True`, the "
        f"default of {GROUP_INDEX_STEP} is used.",
    )

    @pd.validator("bend_axis", always=True)
    @skip_if_fields_missing(["bend_radius"])
    def bend_axis_given(cls, val, values):
        """Check that ``bend_axis`` is provided if ``bend_radius`` is not ``None``"""
        if val is None and values.get("bend_radius") is not None:
            raise SetupError("'bend_axis' must also be defined if 'bend_radius' is defined.")
        return val

    @pd.validator("bend_radius", always=True)
    def bend_radius_not_zero(cls, val, values):
        """Check that ``bend_raidus`` magnitude is not close to zero.`"""
        if val is not None and isclose(val, 0):
            raise SetupError("The magnitude of 'bend_radius' must be larger than 0.")
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

    # Must be executed before type validation by pydantic, otherwise True is converted to 1.0
    @pd.validator("group_index_step", pre=True)
    def assign_default_on_true(cls, val):
        """Assign the default fractional frequency step value if not provided."""
        if val is True:
            return GROUP_INDEX_STEP
        return val

    @pd.validator("group_index_step")
    def check_group_step_size(cls, val):
        """Ensure a reasonable group index step is used."""
        if val >= 1:
            raise ValidationError(
                "Parameter 'group_index_step' is a fractional value. It must be less than 1."
            )
        return val

    @pd.root_validator(skip_on_failure=True)
    def check_precision(cls, values):
        """Verify critical ModeSpec settings for group index calculation."""
        if values["group_index_step"] > 0:
            if values["track_freq"] is None:
                log.warning(
                    "Group index calculation without mode tracking can lead to incorrect results "
                    "around mode crossings. Consider setting 'track_freq' to 'central'."
                )

            # multiply by 5 to be safe
            if values["group_index_step"] < 5 * fp_eps and values["precision"] != "double":
                log.warning(
                    "Group index step is too small! "
                    "The results might be fully corrupted by numerical errors. "
                    "For more accurate results, please consider using 'double' precision, "
                    "or increasing the value of 'group_index_step'."
                )

        return values

    @pd.validator("angle_rotation")
    def angle_rotation_with_phi(cls, val, values):
        """Currently ``angle_rotation`` is only supported with ``angle_phi % (np.pi / 2) == 0``."""
        if val and not isclose(values["angle_phi"] % (np.pi / 2), 0):
            raise ValidationError(
                "Parameter 'angle_phi' must be a multiple of 'np.pi / 2' when 'angle_rotation' is "
                "enabled."
            )
        return val

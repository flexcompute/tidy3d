"""Defines specification for mode solver."""

from __future__ import annotations

from abc import ABC
from math import isclose
from typing import Literal, Optional, Optional, Union

import numpy as np
from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from tidy3d.compat import Self
from tidy3d.constants import GLANCING_CUTOFF, MICROMETER, RADIAN, fp_eps
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .base import Tidy3dBaseModel
from .types import Axis2D, TrackFreq

GROUP_INDEX_STEP = 0.005
MODE_DATA_KEYS = Literal[
    "n_eff",
    "k_eff",
    "TE_fraction",
    "TM_fraction",
    "wg_TE_fraction",
    "wg_TM_fraction",
    "mode_area",
]


class ModeSortSpec(Tidy3dBaseModel):
    """Specification for filtering and sorting modes within each frequency.

    First, an optional filtering step splits the modes into two groups based on a threshold
    applied to ``filter_key``: modes "over" or "under" ``filter_reference`` are placed first,
    with the remaining modes placed next. Second, an optional sorting step orders modes within
    each group according to ``sort_key``, optionally with respect to ``sort_reference`` and in
    the specified ``sort_order``.
    """

    # Filtering stage
    filter_key: Optional[MODE_DATA_KEYS] = Field(
        None,
        title="Filtering key",
        description="Quantity used to filter modes into two groups before sorting.",
    )
    filter_reference: float = Field(
        0.0,
        title="Filtering reference",
        description="Reference value used in the filtering stage.",
    )
    filter_order: Literal["over", "under"] = Field(
        "over",
        title="Filtering order",
        description="Select whether the first group contains values over or under the reference.",
    )

    # Sorting stage
    sort_key: Optional[MODE_DATA_KEYS] = Field(
        None,
        title="Sorting key",
        description="Quantity used to sort modes within each filtered group. If ``None``, "
        "sorting is by descending effective index.",
    )
    sort_reference: Optional[float] = Field(
        None,
        title="Sorting reference",
        description=(
            "If provided, sorting is based on the absolute difference to this reference value."
        ),
    )
    sort_order: Literal["ascending", "descending"] = Field(
        "ascending",
        title="Sorting direction",
        description="Sort order for the selected key or difference to reference value.",
    )

    # Frequency tracking - applied after sorting and filtering
    track_freq: Optional[TrackFreq] = Field(
        "central",
        title="Tracking base frequency",
        description="If provided, enables cross-frequency mode tracking. Can be 'lowest', "
        "'central', or 'highest', which refers to the frequency **index** in the list of "
        "frequencies. The mode sorting would then be exact at the specified frequency, "
        "while at other frequencies it can change depending on the mode tracking.",
    )


class AbstractModeSpec(Tidy3dBaseModel, ABC):
    """
    Abstract base for mode specification data.
    """

    num_modes: PositiveInt = Field(
        1,
        title="Number of modes",
        description="Number of modes returned by mode solver.",
    )

    target_neff: Optional[PositiveFloat] = Field(
        None,
        title="Target effective index",
        description="Guess for effective index of the mode.",
    )

    num_pml: tuple[NonNegativeInt, NonNegativeInt] = Field(
        (0, 0),
        title="Number of PML layers",
        description="Number of standard pml layers to add in the two tangential axes.",
    )

    filter_pol: Optional[Literal["te", "tm"]] = Field(
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

    angle_theta: float = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        units=RADIAN,
    )

    angle_phi: float = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        units=RADIAN,
    )

    precision: Literal["auto", "single", "double"] = Field(
        "double",
        title="single, double, or automatic precision in mode solver",
        description="The solver will be faster and using less memory under "
        "single precision, but more accurate under double precision. "
        "Choose ``'auto'`` to apply double precision if the simulation contains a good "
        "conductor, single precision otherwise.",
    )

    bend_radius: Optional[float] = Field(
        None,
        title="Bend radius",
        description="A curvature radius for simulation of waveguide bends. Can be negative, in "
        "which case the mode plane center has a smaller value than the curvature center along the "
        "tangential axis perpendicular to the bend axis.",
        units=MICROMETER,
    )

    bend_axis: Optional[Axis2D] = Field(
        None,
        title="Bend axis",
        description="Index into the two tangential axes defining the normal to the "
        "plane in which the bend lies. This must be provided if ``bend_radius`` is not ``None``. "
        "For example, for a ring in the global xy-plane, and a mode plane in either the xz or the "
        "yz plane, the ``bend_axis`` is always 1 (the global z axis).",
    )

    angle_rotation: bool = Field(
        False,
        title="Use fields rotation when ``angle_theta`` is not zero",
        description="Defines how modes are computed when ``angle_theta`` is not zero. "
        "If ``False``, a coordinate transformation is applied through the permittivity and permeability tensors."
        "If ``True``, the structures in the simulation are first rotated to compute a mode solution at "
        "a reference plane normal to the structure's azimuthal direction. Then, the fields are rotated "
        "to align with the mode plane, using the ``n_eff`` calculated at the reference plane. The second option can "
        "produce more accurate results, but more care must be taken, for example, in ensuring that the "
        "original mode plane intersects the correct geometries in the simulation with rotated structures. "
        "Note: currently only supported when ``angle_phi`` is a multiple of ``np.pi``.",
    )

    track_freq: Optional[TrackFreq] = Field(
        None,
        title="Mode Tracking Frequency (deprecated)",
        description="Deprecated. Use 'sort_spec.track_freq' instead.",
    )

    group_index_step: Union[PositiveFloat, bool] = Field(
        False,
        title="Frequency step for group index computation",
        description="Control the computation of the group index alongside the effective index. If "
        "set to a positive value, it sets the fractional frequency step used in the numerical "
        "differentiation of the effective index to compute the group index. If set to `True`, the "
        f"default of {GROUP_INDEX_STEP} is used.",
    )

    sort_spec: ModeSortSpec = Field(
        default_factory=ModeSortSpec,
        title="Mode filtering and sorting specification",
        description="Defines how to filter and sort modes within each frequency. If ``track_freq`` "
        "is not ``None``, the sorting is only exact at the specified frequency, while at other "
        "frequencies it can change depending on the mode tracking.",
    )

    @field_validator("group_index_step", mode="before")
    def _validate_group_index_step_default(val):
        """If ``True``, replace with default fractional step."""
        if val is True:
            return GROUP_INDEX_STEP
        return val

    @field_validator("group_index_step")
    def _validate_group_index_step_size(val):
        """Ensure group-index step is < 1."""
        if val is not False and val >= 1:
            raise ValidationError(
                "Parameter 'group_index_step' must be a fractional value less than 1."
            )
        return val

    @field_validator("bend_radius")
    def _validate_bend_radius_not_zero(v):
        """`bend_radius` magnitude must be non-zero."""
        if v is not None and isclose(v, 0):
            raise SetupError("The magnitude of 'bend_radius' must be larger than 0.")
        return v

    @field_validator("angle_theta")
    def _validate_angle_theta_glancing(val):
        """Disallow incidence too close to glancing."""
        if abs(np.pi / 2 - val) < GLANCING_CUTOFF:
            raise SetupError(
                "Mode propagation axis too close to glancing angle for accurate injection. "
                "For best results, switch the injection axis."
            )
        return val

    @model_validator(mode="after")
    def _check_bend_axis_given(self) -> Self:
        """``bend_axis`` must be provided when ``bend_radius`` is set."""
        if self.bend_radius is not None and self.bend_axis is None:
            raise SetupError("'bend_axis' must also be defined if 'bend_radius' is defined.")
        return self

    @model_validator(mode="after")
    def _check_angle_rotation_with_phi(self) -> Self:
        """``angle_rotation`` requires ``angle_phi`` % (π/2) == 0."""
        if self.angle_rotation and not isclose(self.angle_phi % (np.pi / 2), 0):
            raise ValidationError(
                "'angle_phi' must be a multiple of 'π/2' when 'angle_rotation' is enabled."
            )
        return self
    
    @model_validator(mode="after")
    def check_precision(self):
        """Verify critical ModeSpec settings for group index calculation."""
        if self.group_index_step > 0:
            tf = self._track_freq
            if tf is None:
                log.warning(
                    "Group index calculation without mode tracking can lead to incorrect results "
                    "around mode crossings. Consider setting 'sort_spec.track_freq' to 'central'."
                )

            # multiply by 5 to be safe
            if self.group_index_step < 5 * fp_eps and self.precision != "double":
                log.warning(
                    "Group index step is too small! "
                    "The results might be fully corrupted by numerical errors. "
                    "For more accurate results, please consider using 'double' precision, "
                    "or increasing the value of 'group_index_step'."
                )

        return self

    @model_validator(mode="after")
    def _filter_pol_and_sort_spec_exclusive(self):
        """Ensure that 'filter_pol' and 'sort_spec' are not used together."""
        sort_spec = self.sort_spec
        sort_or_filter = sort_spec.filter_key is not None or sort_spec.sort_key is not None
        if self.filter_pol is not None and sort_or_filter:
            raise SetupError(
                "'filter_pol' cannot be used simultaneously with sorting or filtering "
                "defined in 'sort_spec'. Define the filtering in 'sort_spec' exclusively."
            )
        return self

    @field_validator("filter_pol")
    def _filter_pol_deprecated(cls, val):
        """Warn that 'filter_pol' is deprecated in favor of 'sort_spec'."""
        if val is not None:
            log.warning(
                "'filter_pol' is deprecated and will be removed in future versions. "
                "Please use 'sort_spec' instead."
            )
        return val

    @field_validator("track_freq")
    def _track_freq_deprecated(cls, val):
        """Warn that 'track_freq' on ModeSpec is deprecated in favor of 'sort_spec.track_freq'."""
        if val is not None:
            log.warning(
                "'ModeSpec.track_freq' is deprecated and will be removed in future versions. "
                "Please use 'sort_spec.track_freq' instead."
            )
        return val

    @property
    def _track_freq(self) -> Optional[TrackFreq]:
        """Private resolver for tracking frequency: prefers ModeSpec.track_freq if set,
        otherwise falls back to ModeSortSpec.track_freq."""
        if self.track_freq is not None:
            return self.track_freq
        if self.sort_spec is not None:
            return self.sort_spec.track_freq
        return None


class ModeSpec(AbstractModeSpec):
    """
    Stores specifications for the mode solver to find an electromagnetic mode.

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

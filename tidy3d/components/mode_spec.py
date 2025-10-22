"""Defines specification for mode solver."""

from __future__ import annotations

from abc import ABC
from math import isclose
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
import pydantic.v1 as pd

from tidy3d.constants import GLANCING_CUTOFF, MICROMETER, RADIAN, fp_eps
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .base import Tidy3dBaseModel, skip_if_fields_missing
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
    filter_key: Optional[MODE_DATA_KEYS] = pd.Field(
        None,
        title="Filtering key",
        description="Quantity used to filter modes into two groups before sorting.",
    )
    filter_reference: float = pd.Field(
        0.0,
        title="Filtering reference",
        description="Reference value used in the filtering stage.",
    )
    filter_order: Literal["over", "under"] = pd.Field(
        "over",
        title="Filtering order",
        description="Select whether the first group contains values over or under the reference.",
    )

    # Sorting stage
    sort_key: Optional[MODE_DATA_KEYS] = pd.Field(
        None,
        title="Sorting key",
        description="Quantity used to sort modes within each filtered group. If ``None``, "
        "sorting is by descending effective index.",
    )
    sort_reference: Optional[float] = pd.Field(
        None,
        title="Sorting reference",
        description=(
            "If provided, sorting is based on the absolute difference to this reference value."
        ),
    )
    sort_order: Literal["ascending", "descending"] = pd.Field(
        "ascending",
        title="Sorting direction",
        description="Sort order for the selected key or difference to reference value.",
    )

    # Frequency tracking - applied after sorting and filtering
    track_freq: Optional[TrackFreq] = pd.Field(
        "central",
        title="Tracking base frequency",
        description="If provided, enables cross-frequency mode tracking. Can be 'lowest', "
        "'central', or 'highest', which refers to the frequency **index** in the list of "
        "frequencies. The mode sorting would then be exact at the specified frequency, "
        "while at other frequencies it can change depending on the mode tracking.",
    )


class ModeInterpSpec(Tidy3dBaseModel):
    """Specification for mode frequency interpolation.

    Allows computing modes at a reduced set of frequencies and interpolating
    to obtain results at all requested frequencies. This can significantly
    reduce computational cost for broadband simulations where modes vary
    smoothly with frequency.

    Note
    ----
        Requires frequency tracking to be enabled (``mode_spec.sort_spec.track_freq``
        must not be ``None``) to ensure mode ordering is consistent across frequencies.

    Example
    -------
    >>> interp_spec = ModeInterpSpec(num_points=10, method='linear')

    See Also
    --------

    :class:`ModeSolver`:
        Mode solver that can use this specification for efficient broadband computation.

    :class:`ModeSolverMonitor`:
        Monitor that can use this specification to reduce mode computation cost.

    :class:`ModeMonitor`:
        Monitor that can use this specification to reduce mode computation cost.
    """

    num_points: int = pd.Field(
        ...,
        title="Number of Frequency Points",
        description="Number of frequency points at which to actually compute modes. "
        "Must be at least 2 and less than the total number of frequencies requested. "
        "The mode solver will compute modes at these sampling frequencies "
        "and interpolate to obtain results at all requested frequencies. "
        "For 'linear' and 'cubic' methods, points are uniformly spaced. "
        "For 'cheb' method, Chebyshev nodes are used.",
        ge=2,
    )

    method: Literal["linear", "cubic", "cheb"] = pd.Field(
        "linear",
        title="Interpolation Method",
        description="Method for interpolating mode data between computed frequencies. "
        "'linear' uses linear interpolation (faster, requires 2+ points). "
        "'cubic' uses cubic spline interpolation (smoother, more accurate, requires 4+ points). "
        "'cheb' uses Chebyshev polynomial interpolation with barycentric formula "
        "(optimal for smooth functions, requires 3+ points, samples at Chebyshev nodes). "
        "For complex-valued data, real and imaginary parts are interpolated independently.",
    )

    @pd.validator("method", always=True)
    @skip_if_fields_missing(["num_points"])
    def _validate_method_needs_points(cls, val, values):
        """Validate that the method has enough points."""
        num_points = values.get("num_points", 0)
        if val == "cubic" and num_points < 4:
            raise ValidationError(
                "Cubic interpolation requires at least 4 frequency points. "
                f"Got num_points={num_points}. "
                "Use method='linear' or increase num_points."
            )
        if val == "cheb" and num_points < 3:
            raise ValidationError(
                "Chebyshev interpolation requires at least 3 frequency points. "
                f"Got num_points={num_points}. "
                "Use method='linear' or increase num_points."
            )
        return val

    def sampling_points(self, freqs: ArrayLike) -> np.ndarray:
        """Compute frequency sampling points based on the interpolation method.

        Parameters
        ----------
        freqs : ArrayLike
            Target frequency array. The sampling points will span from min(freqs) to max(freqs).

        Returns
        -------
        np.ndarray
            Array of ``num_points`` frequency sampling points.
            For 'linear' and 'cubic' methods: uniformly spaced points.
            For 'cheb' method: Chebyshev nodes of the second kind.

        Example
        -------
        >>> import numpy as np
        >>> freqs = np.linspace(1e14, 2e14, 100)
        >>> interp_spec = ModeInterpSpec(num_points=10, method='cheb')
        >>> sampling_freqs = interp_spec.sampling_points(freqs)
        """
        freqs_array = np.asarray(freqs)
        f_min, f_max = float(freqs_array.min()), float(freqs_array.max())

        if self.method in ("linear", "cubic"):
            # Uniformly spaced points
            return np.linspace(f_min, f_max, self.num_points)
        elif self.method == "cheb":
            # Chebyshev nodes of the second kind: x_k = cos(k*pi/(n-1)) for k=0,...,n-1
            # Map from [-1, 1] to [f_min, f_max]
            k = np.arange(self.num_points)
            nodes_normalized = np.cos(k * np.pi / (self.num_points - 1))
            # Map from [-1, 1] to [f_min, f_max]
            return 0.5 * (f_min + f_max) + 0.5 * (f_max - f_min) * nodes_normalized
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")


class AbstractModeSpec(Tidy3dBaseModel, ABC):
    """
    Abstract base for mode specification data.
    """

    num_modes: pd.PositiveInt = pd.Field(
        1, title="Number of modes", description="Number of modes returned by mode solver."
    )

    target_neff: pd.PositiveFloat = pd.Field(
        None, title="Target effective index", description="Guess for effective index of the mode."
    )

    num_pml: tuple[pd.NonNegativeInt, pd.NonNegativeInt] = pd.Field(
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
        "double",
        title="single, double, or automatic precision in mode solver",
        description="The solver will be faster and using less memory under "
        "single precision, but more accurate under double precision. "
        "Choose ``'auto'`` to apply double precision if the simulation contains a good "
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

    track_freq: Optional[TrackFreq] = pd.Field(
        None,
        title="Mode Tracking Frequency (deprecated)",
        description="Deprecated. Use 'sort_spec.track_freq' instead.",
    )

    group_index_step: Union[pd.PositiveFloat, bool] = pd.Field(
        False,
        title="Frequency step for group index computation",
        description="Control the computation of the group index alongside the effective index. If "
        "set to a positive value, it sets the fractional frequency step used in the numerical "
        "differentiation of the effective index to compute the group index. If set to `True`, the "
        f"default of {GROUP_INDEX_STEP} is used.",
    )

    sort_spec: ModeSortSpec = pd.Field(
        ModeSortSpec(),
        title="Mode filtering and sorting specification",
        description="Defines how to filter and sort modes within each frequency. If ``track_freq`` "
        "is not ``None``, the sorting is only exact at the specified frequency, while at other "
        "frequencies it can change depending on the mode tracking.",
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
            # prefer explicit track_freq on ModeSpec, else fall back to sort_spec.track_freq
            # TODO: can be replaced with self._track_freq in pydantic v2
            tf = values.get("track_freq")
            if tf is None:
                sort_spec = values.get("sort_spec")
                tf = None if sort_spec is None else sort_spec.track_freq
            if tf is None:
                log.warning(
                    "Group index calculation without mode tracking can lead to incorrect results "
                    "around mode crossings. Consider setting 'sort_spec.track_freq' to 'central'."
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

    @pd.root_validator(skip_on_failure=True)
    def _filter_pol_and_sort_spec_exclusive(cls, values):
        """Ensure that 'filter_pol' and 'sort_spec' are not used together."""
        sort_spec = values.get("sort_spec")
        sort_or_filter = sort_spec.filter_key is not None or sort_spec.sort_key is not None
        if values.get("filter_pol") is not None and sort_or_filter:
            raise SetupError(
                "'filter_pol' cannot be used simultaneously with sorting or filtering "
                "defined in 'sort_spec'. Define the filtering in 'sort_spec' exclusively."
            )
        return values

    @pd.validator("filter_pol", always=True)
    def _filter_pol_deprecated(cls, val):
        """Warn that 'filter_pol' is deprecated in favor of 'sort_spec'."""
        if val is not None:
            log.warning(
                "'filter_pol' is deprecated and will be removed in future versions. "
                "Please use 'sort_spec' instead."
            )
        return val

    @pd.validator("track_freq", always=True)
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

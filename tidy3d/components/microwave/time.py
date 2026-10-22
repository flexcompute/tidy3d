"""Baseband (unmodulated) source time classes for transient RF simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from flexcompute.core._migration.em.numerical.raw import (
    baseband_source_time as baseband_numerics,
)
from pydantic import Field, PositiveFloat, field_validator, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import TimeDataArray
from tidy3d.components.data.dataset import TimeDataset
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.source.time import DEFAULT_SIGMA, END_TIME_FACTOR_GAUSSIAN, SourceTime
from tidy3d.components.validators import warn_if_dataset_none
from tidy3d.constants import HERTZ
from tidy3d.exceptions import SetupError, ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types import ArrayFloat1D, FreqBound


class BasebandSourceTime(SourceTime, ABC):
    """Abstract base class for baseband (unmodulated) source time profiles.

    Notes
    -----
    These source times have no carrier frequency and are intended for transient
    RF simulations (TDR, step excitations, arbitrary time-domain profiles).
    The ``phase`` parameter is locked to 0 since baseband signals have no carrier.

    The fastest 10%-to-90% transition sets both the characteristic frequency and the band.
    The characteristic frequency is ``freq0`` when given, which must lie within the band, and
    otherwise the inverse of that transition; ``freq0`` lists what it sets. The band,
    :meth:`frequency_range`, runs from DC to the band of an error-function edge with the
    same transition, so it follows the signal's fastest content rather than where its energy
    sits and does not narrow as a pulse lengthens. Without ``freq0`` the characteristic
    frequency is not a spectral centroid: it lies at about 0.6 of the top of the band, about
    2.9 times the ``0.35 / rise_time`` bandwidth of an edge, so that an automatic grid
    resolves the transitions. Lowering ``freq0`` to a frequency of interest, for a mode
    source or an angled plane wave, coarsens an automatic grid as well; set the grid's
    ``wavelength`` to keep the transitions resolved.
    """

    phase: Literal[0] = Field(
        default=0,
        frozen=True,
        title="Phase",
        description="Phase is locked to 0 for baseband source times (no carrier frequency).",
    )

    freq0: PositiveFloat | None = Field(
        default=None,
        title="Frequency of Interest",
        description="Frequency of interest, used wherever the simulation needs a single "
        "frequency for the source: among others, the wavelength of an automatic grid that has "
        "no ``wavelength``, the frequency at which mode sources solve their mode and media are "
        "evaluated, and the reference frequency of angled plane waves. It does not modulate the "
        "signal, and must lie within :meth:`frequency_range`. If ``None``, the inverse of the "
        "fastest 10%-to-90% transition is used.",
        json_schema_extra={"units": HERTZ},
    )

    @property
    @abstractmethod
    def _fastest_transition_time(self) -> float:
        """Shortest 10%-to-90% transition of the signal in seconds."""

    @cached_property
    def _freq0(self) -> float:
        """Characteristic frequency: ``freq0``, or the inverse of the fastest transition."""
        if self.freq0 is not None:
            return self.freq0
        return 1.0 / self._fastest_transition_time

    @cached_property
    def _freq0_sigma_centroid(self) -> float:
        """The characteristic frequency, since a baseband signal has no useful centroid.

        Sources validate their central frequency against this property rather than against
        ``_freq0``, so it answers with the same value. Left to the base class it would average
        the one-sigma band, which for a slow signal falls below the lowest frequency a source
        accepts.
        """
        return self._freq0

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Band from DC to ``num_fwidth`` spectral widths of an error-function edge with the
        signal's fastest 10%-to-90% transition."""
        sigma = baseband_numerics.sigma_from_rise_time(self._fastest_transition_time)
        return (0.0, num_fwidth * baseband_numerics.gaussian_fwidth(sigma))

    @model_validator(mode="after")
    def _validate_freq0_in_band(self) -> Self:
        """Reject a ``freq0`` above the band, where the source carries no energy."""
        if self.freq0 is not None:
            fmax = self.frequency_range()[1]
            if self.freq0 > fmax:
                self._raise_validation_error_at_loc(
                    f"'freq0' of {self.freq0:.3e} Hz lies above the source's band, which ends at "
                    f"{fmax:.3e} Hz. Set 'freq0' within the band, or set the grid's 'wavelength' "
                    "to refine an automatic grid.",
                    "freq0",
                )
        return self


class BasebandStep(BasebandSourceTime):
    """Step function source time profile using an error function (erf).

    Notes
    -----
    The signal ramps from 0 to ``amplitude`` with ``rise_time`` specifying the
    10%-to-90% rise time.

    Example
    -------
    >>> step = BasebandStep(rise_time=1e-9)
    """

    rise_time: PositiveFloat = Field(
        title="Rise Time",
        description="The 10%-to-90% rise time in seconds. "
        "Related to the Gaussian standard deviation (sigma) by ``rise_time ≈ 2.563 * sigma``.",
    )

    offset: float = Field(
        default=5.0,
        title="Offset",
        description="Delay of the step center in units of ``rise_time``.",
        ge=2.5,
    )

    @cached_property
    def _sigma(self) -> float:
        """Gaussian standard deviation derived from the 10-90% rise time."""
        return baseband_numerics.sigma_from_rise_time(self.rise_time)

    @cached_property
    def t_center(self) -> float:
        """Time of the step center in seconds."""
        return self.offset * self.rise_time

    @property
    def _fastest_transition_time(self) -> float:
        """The step's 10%-to-90% rise time."""
        return self.rise_time

    @cached_property
    def fwidth_equiv(self) -> float:
        """Equivalent frequency-domain standard deviation (sigma_f) in Hz.

        Defined as the frequency where the Gaussian spectral envelope drops
        to ``exp(-1/2)`` of its peak value."""
        return self.frequency_range(num_fwidth=1)[1]

    def amp_time(self, time: float | ArrayFloat1D) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        return baseband_numerics.baseband_step_amp_time(
            time, amplitude=self.amplitude, t_center=self.t_center, sigma=self._sigma
        )

    def end_time(self) -> float | None:
        """Step never decays, so no end time."""
        return None


class BasebandGaussianPulse(BasebandSourceTime):
    """Unmodulated Gaussian pulse source time profile.

    Notes
    -----
    The signal is a Gaussian envelope centered at ``offset * twidth`` with
    temporal width ``twidth``. Its fastest 10%-to-90% transition is the rise of its leading
    half, about ``1.687 * twidth``. Its band is its own Gaussian spectrum, narrower than that
    of an error-function edge with the same transition.

    Example
    -------
    >>> gauss = BasebandGaussianPulse(twidth=1e-9)
    """

    twidth: PositiveFloat = Field(
        title="Temporal Width",
        description="Temporal width (standard deviation) of the Gaussian pulse in seconds.",
    )

    offset: float = Field(
        default=5.0,
        title="Offset",
        description="Delay of the pulse peak in units of ``twidth``.",
        ge=2.5,
    )

    @cached_property
    def t_center(self) -> float:
        """Time of the pulse peak in seconds."""
        return self.offset * self.twidth

    @cached_property
    def fwidth(self) -> float:
        """Standard deviation of the frequency content of the Gaussian pulse."""
        return baseband_numerics.gaussian_fwidth(self.twidth)

    @property
    def _fastest_transition_time(self) -> float:
        """The 10%-to-90% rise of the pulse's leading half."""
        return baseband_numerics.GAUSSIAN_PULSE_10_90_FACTOR * self.twidth

    def amp_time(self, time: float | ArrayFloat1D) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        return baseband_numerics.baseband_gaussian_pulse_amp_time(
            time, amplitude=self.amplitude, t_center=self.t_center, twidth=self.twidth
        )

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Band from DC to ``num_fwidth`` widths of the pulse's Gaussian spectrum."""
        return (0.0, num_fwidth * self.fwidth)

    def end_time(self) -> float | None:
        """Time after which the source is effectively zero."""
        return self.t_center + END_TIME_FACTOR_GAUSSIAN * self.twidth


class BasebandRectangularPulse(BasebandSourceTime):
    """Smoothed rectangular pulse source time profile.

    Notes
    -----
    The signal is a flat-top pulse constructed from two error functions,
    producing a smooth rise and fall. The ``rise_time`` parameter specifies
    the 10%-to-90% rise/fall time.

    Example
    -------
    >>> pulse = BasebandRectangularPulse(rise_time=1e-10, twidth=1e-9)
    """

    rise_time: PositiveFloat = Field(
        title="Rise Time",
        description="The 10%-to-90% rise/fall time in seconds. "
        "Related to the Gaussian standard deviation (sigma) by ``rise_time ≈ 2.563 * sigma``.",
    )

    twidth: PositiveFloat = Field(
        title="Temporal Width",
        description="Duration of the flat-top region in seconds.",
    )

    offset: float = Field(
        default=5.0,
        title="Offset",
        description="Delay of the pulse start (rising edge center) in units of ``rise_time``.",
        ge=2.5,
    )

    @cached_property
    def _sigma(self) -> float:
        """Gaussian standard deviation derived from the 10-90% rise time."""
        return baseband_numerics.sigma_from_rise_time(self.rise_time)

    @cached_property
    def t_start(self) -> float:
        """Center of the rising edge in seconds."""
        return self.offset * self.rise_time

    @cached_property
    def t_stop(self) -> float:
        """Center of the falling edge in seconds."""
        return self.t_start + self.twidth

    @property
    def _fastest_transition_time(self) -> float:
        """The pulse's 10%-to-90% rise and fall time."""
        return self.rise_time

    @cached_property
    def fwidth_equiv(self) -> float:
        """Equivalent frequency-domain standard deviation (sigma_f) in Hz.

        Defined as the frequency where the Gaussian spectral envelope drops
        to ``exp(-1/2)`` of its peak value."""
        return self.frequency_range(num_fwidth=1)[1]

    def amp_time(self, time: float | ArrayFloat1D) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        return baseband_numerics.baseband_rectangular_pulse_amp_time(
            time,
            amplitude=self.amplitude,
            t_start=self.t_start,
            t_stop=self.t_stop,
            sigma=self._sigma,
        )

    def end_time(self) -> float | None:
        """Time after which the source is effectively zero."""
        return self.t_stop + END_TIME_FACTOR_GAUSSIAN * self._sigma


class BasebandCustomSourceTime(BasebandSourceTime):
    """Custom baseband source time profile from a user-provided time-domain dataset.

    Notes
    -----
    The signal is defined by a ``source_time_dataset`` containing the time-domain
    envelope. The band, and the characteristic frequency unless ``freq0`` is given, follow
    the fastest 10%-to-90% transition measured across the envelope's full swing, so the
    record must hold at least one. Noise of more than a fraction of a percent of the swing
    lengthens that measurement, so give ``freq0`` for a noisy record.

    Example
    -------
    >>> cst = BasebandCustomSourceTime.from_values(values=np.linspace(0, 1, 100), dt=1e-10)
    """

    source_time_dataset: TimeDataset | None = Field(
        default=None,
        title="Source time dataset",
        description="Dataset for storing the baseband source time envelope. "
        "If ``None``, the source produces no signal and must be populated before use.",
    )

    _no_nans_dataset = validate_no_nans("source_time_dataset")
    _source_time_dataset_none_warning = warn_if_dataset_none("source_time_dataset")

    @field_validator("source_time_dataset")
    @classmethod
    def _validate_time_coords(cls, val: TimeDataset | None) -> TimeDataset | None:
        """Time coordinates must have more than one point and be strictly increasing."""
        if val is None:
            return val
        if val.values.size <= 1:
            raise ValidationError(
                "'BasebandCustomSourceTime' must have more than one time coordinate."
            )
        times = val.values.coords["t"].values
        if not np.all(np.diff(times) > 0):
            raise ValidationError(
                "'BasebandCustomSourceTime' time coordinates must be strictly monotonically"
                " increasing."
            )
        return val

    @model_validator(mode="after")
    def _validate_freq0_in_band(self) -> Self:
        """Check ``freq0`` once there is a record to take the band from."""
        if self.source_time_dataset is None:
            return self
        return super()._validate_freq0_in_band()

    @classmethod
    def from_values(
        cls, values: ArrayFloat1D, dt: float, **kwargs: Any
    ) -> BasebandCustomSourceTime:
        """Create a :class:`.BasebandCustomSourceTime` from a numpy array.

        Parameters
        ----------
        values : ArrayFloat1D
            Real-valued source envelope samples.
        dt : float
            Time step for the ``values`` array.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        BasebandCustomSourceTime
            Source time with envelope given by ``values``.
        """
        if dt <= 0:
            raise ValidationError("'dt' must be positive.")
        times = np.arange(len(values)) * dt
        source_time_dataarray = TimeDataArray(values, coords={"t": times})
        source_time_dataset = TimeDataset(values=source_time_dataarray)
        return cls(source_time_dataset=source_time_dataset, **kwargs)

    @property
    def data_times(self) -> ArrayFloat1D:
        """Times of envelope definition."""
        if self.source_time_dataset is None:
            return np.array([], dtype=float)
        return self.source_time_dataset.values.coords["t"].values.squeeze()

    def amp_time(self, time: float | ArrayFloat1D) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        if self.source_time_dataset is None:
            raise SetupError("'source_time_dataset' must be provided to use this method.")

        return baseband_numerics.baseband_custom_amp_time(
            time,
            amplitude=self.amplitude,
            data_times=self.data_times,
            data_values=self.source_time_dataset.values.to_numpy().squeeze(),
        )

    @cached_property
    def _fastest_transition_time(self) -> float:
        """Shortest 10%-to-90% transition of the sampled envelope.

        Measured on the linear interpolant of the samples across the envelope's full swing,
        so settled samples appended to the record do not change it.
        """
        if self.source_time_dataset is None:
            raise SetupError("'source_time_dataset' must be provided to use this method.")

        fastest = baseband_numerics.fastest_transition_time(
            self.data_times, self.source_time_dataset.values.to_numpy().squeeze()
        )
        if fastest is None:
            raise SetupError(
                "'source_time_dataset' holds a single level and no transition, so the source "
                "has no characteristic frequency or band."
            )
        return fastest

    def end_time(self) -> float | None:
        """Time of the last non-zero value in the dataset."""
        if self.source_time_dataset is None:
            raise SetupError("'source_time_dataset' must be provided to use this method.")

        data_array = self.source_time_dataset.values
        t_coords = data_array.coords["t"]
        source_is_non_zero = ~np.isclose(abs(data_array), 0)
        t_non_zero = t_coords[source_is_non_zero]

        if len(t_non_zero) == 0:
            return None

        return float(np.max(t_non_zero))

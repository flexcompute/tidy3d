"""Baseband (unmodulated) source time classes for transient RF simulations."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
from pydantic import Field, PositiveFloat, field_validator

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import TimeDataArray
from tidy3d.components.data.dataset import TimeDataset
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.source.time import DEFAULT_SIGMA, END_TIME_FACTOR_GAUSSIAN, SourceTime
from tidy3d.components.validators import warn_if_dataset_none
from tidy3d.exceptions import SetupError, ValidationError

if TYPE_CHECKING:
    from typing import Union

    from tidy3d.components.types import ArrayFloat1D, FreqBound

# Factor converting 10%-to-90% rise time to Gaussian sigma (standard deviation).
# rise_time = RISE_TIME_10_90_FACTOR * sigma
# Derived from: 2 * sqrt(2) * erfinv(0.8) ≈ 2.5631
RISE_TIME_10_90_FACTOR = 2.5631031310892006


class BasebandSourceTime(SourceTime, ABC):
    """Abstract base class for baseband (unmodulated) source time profiles.

    These source times have no carrier frequency and are intended for transient
    RF simulations (TDR, step excitations, arbitrary time-domain profiles).
    The ``phase`` parameter is locked to 0 since baseband signals have no carrier.
    """

    phase: Literal[0] = Field(
        0,
        frozen=True,
        title="Phase",
        description="Phase is locked to 0 for baseband source times (no carrier frequency).",
    )


class BasebandStep(BasebandSourceTime):
    """Step function source time profile using an error function (erf).

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
        5.0,
        title="Offset",
        description="Delay of the step center in units of ``rise_time``.",
        ge=2.5,
    )

    @cached_property
    def _sigma(self) -> float:
        """Gaussian standard deviation derived from the 10-90% rise time."""
        return self.rise_time / RISE_TIME_10_90_FACTOR

    @cached_property
    def t_center(self) -> float:
        """Time of the step center in seconds."""
        return self.offset * self.rise_time

    @cached_property
    def fwidth_equiv(self) -> float:
        """Equivalent frequency-domain standard deviation (sigma_f) in Hz.

        Defined as the frequency where the Gaussian spectral envelope drops
        to ``exp(-1/2)`` of its peak value."""
        return 1.0 / (2 * np.pi * self._sigma)

    def amp_time(self, time: Union[float, ArrayFloat1D]) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        from scipy.special import erf

        time = np.atleast_1d(np.asarray(time, dtype=float))
        return (
            self.amplitude * 0.5 * (1.0 + erf((time - self.t_center) / (np.sqrt(2) * self._sigma)))
        )

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range based on equivalent bandwidth."""
        return (0.0, num_fwidth * self.fwidth_equiv)

    def end_time(self) -> Optional[float]:
        """Step never decays, so no end time."""
        return None


class BasebandGaussianPulse(BasebandSourceTime):
    """Unmodulated Gaussian pulse source time profile.

    The signal is a Gaussian envelope centered at ``offset * twidth`` with
    temporal width ``twidth``.

    Example
    -------
    >>> gauss = BasebandGaussianPulse(twidth=1e-9)
    """

    twidth: PositiveFloat = Field(
        title="Temporal Width",
        description="Temporal width (standard deviation) of the Gaussian pulse in seconds.",
    )

    offset: float = Field(
        5.0,
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
        return 1.0 / (2 * np.pi * self.twidth)

    def amp_time(self, time: Union[float, ArrayFloat1D]) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        time = np.atleast_1d(np.asarray(time, dtype=float))
        return self.amplitude * np.exp(-((time - self.t_center) ** 2) / (2 * self.twidth**2))

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range based on equivalent bandwidth."""
        return (0.0, num_fwidth * self.fwidth)

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively zero."""
        return self.t_center + END_TIME_FACTOR_GAUSSIAN * self.twidth


class BasebandRectangularPulse(BasebandSourceTime):
    """Smoothed rectangular pulse source time profile.

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
        5.0,
        title="Offset",
        description="Delay of the pulse start (rising edge center) in units of ``rise_time``.",
        ge=2.5,
    )

    @cached_property
    def _sigma(self) -> float:
        """Gaussian standard deviation derived from the 10-90% rise time."""
        return self.rise_time / RISE_TIME_10_90_FACTOR

    @cached_property
    def t_start(self) -> float:
        """Center of the rising edge in seconds."""
        return self.offset * self.rise_time

    @cached_property
    def t_stop(self) -> float:
        """Center of the falling edge in seconds."""
        return self.t_start + self.twidth

    @cached_property
    def fwidth_equiv(self) -> float:
        """Equivalent frequency-domain standard deviation (sigma_f) in Hz.

        Defined as the frequency where the Gaussian spectral envelope drops
        to ``exp(-1/2)`` of its peak value."""
        return 1.0 / (2 * np.pi * self._sigma)

    def amp_time(self, time: Union[float, ArrayFloat1D]) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        from scipy.special import erf

        time = np.atleast_1d(np.asarray(time, dtype=float))
        rise = erf((time - self.t_start) / (np.sqrt(2) * self._sigma))
        fall = erf((time - self.t_stop) / (np.sqrt(2) * self._sigma))
        return self.amplitude * 0.5 * (rise - fall)

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range based on equivalent bandwidth (determined by rise time)."""
        return (0.0, num_fwidth * self.fwidth_equiv)

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively zero."""
        return self.t_stop + END_TIME_FACTOR_GAUSSIAN * self._sigma


class BasebandCustomSourceTime(BasebandSourceTime):
    """Custom baseband source time profile from a user-provided time-domain dataset.

    The signal is defined by a ``source_time_dataset`` containing the time-domain
    envelope.

    Example
    -------
    >>> cst = BasebandCustomSourceTime.from_values(values=np.linspace(0, 1, 100), dt=1e-10)
    """

    source_time_dataset: Optional[TimeDataset] = Field(
        None,
        title="Source time dataset",
        description="Dataset for storing the baseband source time envelope. "
        "If ``None``, the source produces no signal and must be populated before use.",
    )

    _no_nans_dataset = validate_no_nans("source_time_dataset")
    _source_time_dataset_none_warning = warn_if_dataset_none("source_time_dataset")

    @field_validator("source_time_dataset")
    @classmethod
    def _validate_time_coords(cls, val: Optional[TimeDataset]) -> Optional[TimeDataset]:
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

    def amp_time(self, time: Union[float, ArrayFloat1D]) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        if self.source_time_dataset is None:
            raise SetupError("'source_time_dataset' must be provided to use this method.")

        times = np.atleast_1d(np.asarray(time, dtype=float))
        data_times = self.data_times

        # Mask for times outside dataset range; these use endpoint clamping
        mask = (times < min(data_times)) | (times > max(data_times))

        # Interpolate envelope
        envelope = np.zeros(len(times), dtype=float)
        values = self.source_time_dataset.values
        envelope[mask] = values.sel(t=times[mask], method="nearest").to_numpy()
        if not all(mask):
            envelope[~mask] = values.interp(t=times[~mask]).to_numpy()

        return self.amplitude * envelope

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range computed from FFT of the source envelope.

        Note: only the real part of the envelope is used for frequency estimation.
        """
        if self.source_time_dataset is None:
            raise SetupError("'source_time_dataset' must be provided to use this method.")

        data_times = self.data_times
        values = np.real(self.source_time_dataset.values.to_numpy().squeeze())
        dts = np.diff(data_times)
        dt = float(np.min(dts))

        # Resample to uniform grid if times are non-uniform
        if not np.allclose(dts, dt):
            uniform_times = np.arange(data_times[0], data_times[-1] + dt * 0.5, dt)
            values = np.interp(uniform_times, data_times, values)

        return self._frequency_range_from_fft(dt, values, num_fwidth)

    def end_time(self) -> Optional[float]:
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

"""Defines time dependencies of injected electromagnetic sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pydantic.v1 as pydantic

from ...constants import HERTZ
from ...exceptions import ValidationError
from ..data.data_array import TimeDataArray
from ..data.dataset import TimeDataset
from ..data.validators import validate_no_nans
from ..time import AbstractTimeDependence
from ..types import (
    ArrayComplex1D,
    ArrayFloat1D,
    Ax,
    FreqBound,
    PlotVal,
)
from ..validators import warn_if_dataset_none
from ..viz import add_ax_if_none

# how many units of ``twidth`` from the ``offset`` until a gaussian pulse is considered "off"
END_TIME_FACTOR_GAUSSIAN = 10


class SourceTime(AbstractTimeDependence):
    """Base class describing the time dependence of a source."""

    @add_ax_if_none
    def plot_spectrum(
        self,
        times: ArrayFloat1D,
        num_freqs: int = 101,
        val: PlotVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.
        Note: Only the real part of the time signal is used.

        Parameters
        ----------
        times : np.ndarray
            Array of evenly-spaced times (seconds) to evaluate source time-dependence at.
            The spectrum is computed from this value and the source time frequency content.
            To see source spectrum for a specific :class:`Simulation`,
            pass ``simulation.tmesh``.
        num_freqs : int = 101
            Number of frequencies to plot within the SourceTime.frequency_range.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        fmin, fmax = self.frequency_range()
        return self.plot_spectrum_in_frequency_range(
            times, fmin, fmax, num_freqs=num_freqs, val=val, ax=ax
        )

    @abstractmethod
    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within plus/minus ``num_fwidth * fwidth`` of the central frequency."""

    @abstractmethod
    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""


class Pulse(SourceTime, ABC):
    """A source time that ramps up with some ``fwidth`` and oscillates at ``freq0``."""

    freq0: pydantic.PositiveFloat = pydantic.Field(
        ..., title="Central Frequency", description="Central frequency of the pulse.", units=HERTZ
    )
    fwidth: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="",
        description="Standard deviation of the frequency content of the pulse.",
        units=HERTZ,
    )

    offset: float = pydantic.Field(
        5.0,
        title="Offset",
        description="Time delay of the maximum value of the "
        "pulse in units of 1 / (``2pi * fwidth``).",
        ge=2.5,
    )

    @property
    def twidth(self) -> float:
        """Width of pulse in seconds."""
        return 1.0 / (2 * np.pi * self.fwidth)

    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency.

        Parameters
        ----------
        num_fwidth : float = 4.
            Frequency range defined as plus/minus ``num_fwidth * self.fwdith``.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum frequencies of the :class:`GaussianPulse` or :class:`ContinuousWave`
            power.
        """

        freq_width_range = num_fwidth * self.fwidth
        freq_min = max(0, self.freq0 - freq_width_range)
        freq_max = self.freq0 + freq_width_range
        return (freq_min, freq_max)


class GaussianPulse(Pulse):
    """Source time dependence that describes a Gaussian pulse.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    """

    remove_dc_component: bool = pydantic.Field(
        True,
        title="Remove DC Component",
        description="Whether to remove the DC component in the Gaussian pulse spectrum. "
        "If ``True``, the Gaussian pulse is modified at low frequencies to zero out the "
        "DC component, which is usually desirable so that the fields will decay. However, "
        "for broadband simulations, it may be better to have non-vanishing source power "
        "near zero frequency. Setting this to ``False`` results in an unmodified Gaussian "
        "pulse spectrum which can have a nonzero DC component.",
    )

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * self.twidth

        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted**2) / 2 / self.twidth**2) * self.amplitude

        pulse_amp = offset * oscillation * amp

        # subtract out DC component
        if self.remove_dc_component:
            pulse_amp = pulse_amp * (1j + time_shifted / self.twidth**2 / omega0)
        else:
            # 1j to make it agree in large omega0 limit
            pulse_amp = pulse_amp * 1j

        return pulse_amp

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""

        # TODO: decide if we should continue to return an end_time if the DC component remains
        # if not self.remove_dc_component:
        #     return None

        return self.offset * self.twidth + END_TIME_FACTOR_GAUSSIAN * self.twidth

    @property
    def amp_complex(self) -> complex:
        """Grab the complex amplitude from a ``GaussianPulse``."""
        phase = np.exp(1j * self.phase)
        return self.amplitude * phase

    @classmethod
    def from_amp_complex(cls, amp: complex, **kwargs) -> GaussianPulse:
        """Set the complex amplitude of a ``GaussianPulse``.

        Parameters
        ----------
        amp : complex
            Complex-valued amplitude to set in the returned ``GaussianPulse``.
        kwargs : dict
            Keyword arguments passed to ``GaussianPulse()``, excluding ``amplitude`` & ``phase``.
        """
        amplitude = abs(amp)
        phase = np.angle(amp)
        return cls(amplitude=amplitude, phase=phase, **kwargs)

    @classmethod
    def from_frequency_range(
        cls, fmin: pydantic.PositiveFloat, fmax: pydantic.PositiveFloat, **kwargs
    ) -> GaussianPulse:
        """Create a ``GaussianPulse`` that maximizes its amplitude in the frequency range [fmin, fmax].

        Parameters
        ----------
        fmin : float
            Lower bound of frequency of interest.
        fmax : float
            Upper bound of frequency of interest.
        kwargs : dict
            Keyword arguments passed to ``GaussianPulse()``, excluding ``freq0`` & ``fwidth``.

        Returns
        -------
        GaussianPulse
            A ``GaussianPulse`` that maximizes its amplitude in the frequency range [fmin, fmax].
        """
        # validate that fmin and fmax must positive, and fmax > fmin
        if fmin <= 0:
            raise ValidationError("'fmin' must be positive.")
        if fmax <= fmin:
            raise ValidationError("'fmax' must be greater than 'fmin'.")

        # frequency range and center
        freq_range = fmax - fmin
        freq_center = (fmax + fmin) / 2.0

        # If remove_dc_component=False, simply return the standard GaussianPulse parameters
        if kwargs.get("remove_dc_component", True) is False:
            return cls(freq0=freq_center, fwidth=freq_range / 2.0, **kwargs)

        # If remove_dc_component=True, the Gaussian pulse is distorted
        kwargs.update({"remove_dc_component": True})
        log_ratio = np.log(fmax / fmin)
        coeff = ((1 + log_ratio**2) ** 0.5 - 1) / 2.0
        freq0 = freq_center - coeff / log_ratio * freq_range
        fwidth = freq_range / log_ratio * coeff**0.5
        return cls(freq0=freq0, fwidth=fwidth, **kwargs)


class ContinuousWave(Pulse):
    """Source time dependence that ramps up to continuous oscillation
    and holds until end of simulation.

    Note
    ----
    Field decay will not occur, so the simulation will run for the full ``run_time``.
    Also, source normalization of frequency-domain monitors is not meaningful.

    Example
    -------
    >>> cw = ContinuousWave(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1.0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = 1 / (1 + np.exp(-time_shifted / twidth)) * self.amplitude

        return const * offset * oscillation * amp

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""
        return None


class CustomSourceTime(Pulse):
    """Custom source time dependence consisting of a real or complex envelope
    modulated at a central frequency, as shown below.

    Note
    ----
    .. math::

        amp\\_time(t) = amplitude \\cdot \\
                e^{i \\cdot phase - 2 \\pi i \\cdot freq0 \\cdot t} \\cdot \\
                envelope(t - offset / (2 \\pi \\cdot fwidth))

    Note
    ----
    Depending on the envelope, field decay may not occur.
    If field decay does not occur, then the simulation will run for the full ``run_time``.
    Also, if field decay does not occur, then source normalization of frequency-domain
    monitors is not meaningful.

    Note
    ----
    The source time dependence is linearly interpolated to the simulation time steps.
    The sampling rate should be sufficiently fast that this interpolation does not
    introduce artifacts. The source time dependence should also start at zero and ramp up smoothly.
    The first and last values of the envelope will be used for times that are out of range
    of the provided data.

    Example
    -------
    >>> cst = CustomSourceTime.from_values(freq0=1, fwidth=0.1,
    ...     values=np.linspace(0, 9, 10), dt=0.1)

    """

    offset: float = pydantic.Field(
        0.0,
        title="Offset",
        description="Time delay of the envelope in units of 1 / (``2pi * fwidth``).",
    )

    source_time_dataset: Optional[TimeDataset] = pydantic.Field(
        ...,
        title="Source time dataset",
        description="Dataset for storing the envelope of the custom source time. "
        "This envelope will be modulated by a complex exponential at frequency ``freq0``.",
    )

    _no_nans_dataset = validate_no_nans("source_time_dataset")
    _source_time_dataset_none_warning = warn_if_dataset_none("source_time_dataset")

    @pydantic.validator("source_time_dataset", always=True)
    def _more_than_one_time(cls, val):
        """Must have more than one time to interpolate."""
        if val is None:
            return val
        if val.values.size <= 1:
            raise ValidationError("'CustomSourceTime' must have more than one time coordinate.")
        return val

    @classmethod
    def from_values(
        cls, freq0: float, fwidth: float, values: ArrayComplex1D, dt: float
    ) -> CustomSourceTime:
        """Create a :class:`.CustomSourceTime` from a numpy array.

        Parameters
        ----------
        freq0 : float
            Central frequency of the source. The envelope provided will be modulated
            by a complex exponential at this frequency.
        fwidth : float
            Estimated frequency width of the source.
        values: ArrayComplex1D
            Complex values of the source envelope.
        dt: float
            Time step for the ``values`` array. This value should be sufficiently small
            that the interpolation to simulation time steps does not introduce artifacts.

        Returns
        -------
        CustomSourceTime
            :class:`.CustomSourceTime` with envelope given by ``values``, modulated by a complex
            exponential at frequency ``freq0``. The time coordinates are evenly spaced
            between ``0`` and ``dt * (N-1)`` with a step size of ``dt``, where ``N`` is the length of
            the values array.
        """

        times = np.arange(len(values)) * dt
        source_time_dataarray = TimeDataArray(values, coords=dict(t=times))
        source_time_dataset = TimeDataset(values=source_time_dataarray)
        return CustomSourceTime(
            freq0=freq0,
            fwidth=fwidth,
            source_time_dataset=source_time_dataset,
        )

    @property
    def data_times(self) -> ArrayFloat1D:
        """Times of envelope definition."""
        if self.source_time_dataset is None:
            return []
        data_times = self.source_time_dataset.values.coords["t"].values.squeeze()
        return data_times

    def _all_outside_range(self, run_time: float) -> bool:
        """Whether all times are outside range of definition."""

        # can't validate if data isn't loaded
        if self.source_time_dataset is None:
            return False

        # make time a numpy array for uniform handling
        data_times = self.data_times

        # shift time
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        max_time_shifted = run_time - self.offset * twidth
        min_time_shifted = -self.offset * twidth

        return (max_time_shifted < min(data_times)) | (min_time_shifted > max(data_times))

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at that time.
        """

        if self.source_time_dataset is None:
            return None

        # make time a numpy array for uniform handling
        times = np.array([time] if isinstance(time, (int, float)) else time)
        data_times = self.data_times

        # shift time
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        time_shifted = times - self.offset * twidth

        # mask times that are out of range
        mask = (time_shifted < min(data_times)) | (time_shifted > max(data_times))

        # get envelope
        envelope = np.zeros(len(time_shifted), dtype=complex)
        values = self.source_time_dataset.values
        envelope[mask] = values.sel(t=time_shifted[mask], method="nearest").to_numpy()
        if not all(mask):
            envelope[~mask] = values.interp(t=time_shifted[~mask]).to_numpy()

        # modulation, phase, amplitude
        omega0 = 2 * np.pi * self.freq0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * times)
        amp = self.amplitude

        return offset * oscillation * amp * envelope

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""

        if self.source_time_dataset is None:
            return None

        data_array = self.source_time_dataset.values

        t_coords = data_array.coords["t"]
        source_is_non_zero = ~np.isclose(abs(data_array), 0)
        t_non_zero = t_coords[source_is_non_zero]

        return np.max(t_non_zero)


SourceTimeType = Union[GaussianPulse, ContinuousWave, CustomSourceTime]

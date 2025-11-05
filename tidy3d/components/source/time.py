"""Defines time dependencies of injected electromagnetic sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pydantic.v1 as pydantic
from pyroots import Brentq

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import TimeDataArray
from tidy3d.components.data.dataset import TimeDataset
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.time import AbstractTimeDependence
from tidy3d.components.types import ArrayComplex1D, ArrayFloat1D, Ax, FreqBound, PlotVal
from tidy3d.components.validators import warn_if_dataset_none
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import HERTZ
from tidy3d.exceptions import ValidationError
from tidy3d.log import log
from tidy3d.packaging import check_tidy3d_extras_licensed_feature, tidy3d_extras

# how many units of ``twidth`` from the ``offset`` until a gaussian pulse is considered "off"
END_TIME_FACTOR_GAUSSIAN = 10

# warn if source amplitude is too small at the endpoints of frequency range
WARN_SOURCE_AMPLITUDE = 0.1
# used in Brentq
_ROOTS_TOL = 1e-10
# Default sigma value in frequency_range
DEFAULT_SIGMA = 4.0
# Offset in fwidth in finding frequency_range_sigma[1] to ensure the interval brackets the root
OFFSET_FWIDTH_FMAX = 100


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
            To see source spectrum for a specific :class:`.Simulation`,
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

        fmin, fmax = self.frequency_range_sigma()
        return self.plot_spectrum_in_frequency_range(
            times, fmin, fmax, num_freqs=num_freqs, val=val, ax=ax
        )

    @abstractmethod
    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range within plus/minus ``num_fwidth * fwidth`` of the central frequency."""

    def frequency_range_sigma(self, sigma: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range where the source amplitude is within ``exp(-sigma**2/2)`` of the peak amplitude."""
        return self.frequency_range(num_fwidth=sigma)

    @cached_property
    def _frequency_range_sigma_cached(self) -> FreqBound:
        """Cached `frequency_range_sigma` for the default sigma value."""
        return self.frequency_range_sigma(sigma=DEFAULT_SIGMA)

    @abstractmethod
    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""

    @cached_property
    def _freq0(self) -> float:
        """Central frequency. If not present in input parameters, returns `_freq0_sigma_centroid`."""
        return self._freq0_sigma_centroid

    @cached_property
    def _freq0_sigma_centroid(self) -> float:
        """Central of frequency range at 1-sigma drop from the peak amplitude."""
        return np.mean(self.frequency_range_sigma(sigma=1))


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

    @cached_property
    def _freq0(self) -> float:
        """Central frequency."""
        return self.freq0

    @property
    def offset_time(self) -> float:
        """Offset time in seconds."""
        return self.offset * self.twidth

    @property
    def twidth(self) -> float:
        """Width of pulse in seconds."""
        return 1.0 / (2 * np.pi * self.fwidth)

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
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

    @property
    def peak_time(self) -> float:
        """Peak time in seconds, defined by ``offset``."""
        return self.offset * self.twidth

    @property
    def _peak_time_shift(self) -> float:
        """In the case of DC removal, correction to offset_time so that ``offset`` indeed defines time delay
        of pulse peak.
        """
        if self.remove_dc_component and self.fwidth > self.freq0:
            return self.twidth * np.sqrt(1 - self.freq0**2 / self.fwidth**2)
        return 0

    @property
    def offset_time(self) -> float:
        """Offset time in seconds. Note that in the case of DC removal, the maximal value of pulse can be shifted."""
        return self.peak_time + self._peak_time_shift

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset_time

        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted**2) / 2 / self.twidth**2) * self.amplitude

        pulse_amp = offset * oscillation * amp

        # subtract out DC component
        if self.remove_dc_component:
            pulse_amp = pulse_amp * (1j * omega0 + time_shifted / self.twidth**2)
            # normalize by peak frequency instead of omega0, as for small omega0, omega0 approaches 0 faster
            pulse_amp /= 2 * np.pi * self.peak_frequency
        else:
            # 1j to make it agree in large omega0 limit
            pulse_amp = pulse_amp * 1j

        return pulse_amp

    def end_time(self) -> Optional[float]:
        """Time after which the source is effectively turned off / close to zero amplitude."""

        # TODO: decide if we should continue to return an end_time if the DC component remains
        # if not self.remove_dc_component:
        #     return None

        end_time = self.offset_time + END_TIME_FACTOR_GAUSSIAN * self.twidth

        # for derivative Gaussian that contains two peaks, add time interval between them
        if self.remove_dc_component and self.fwidth > self.freq0:
            end_time += 2 * self._peak_time_shift
        return end_time

    def amp_freq(self, freq: float) -> complex:
        """Complex-valued source spectrum in frequency domain."""
        phase = np.exp(1j * self.phase + 1j * 2 * np.pi * (freq - self.freq0) * self.offset_time)
        envelope = np.exp(-((freq - self.freq0) ** 2) / 2 / self.fwidth**2)
        amp = 1j * self.amplitude / self.fwidth * phase * envelope
        if not self.remove_dc_component:
            return amp

        # derivative of Gaussian when DC is removed
        return freq * amp / (2 * np.pi * self.peak_frequency)

    def _rel_amp_freq(self, freq: float) -> complex:
        """Complex-valued source spectrum in frequency domain normalized by peak amplitude."""
        return self.amp_freq(freq) / self._peak_freq_amp

    @property
    def peak_frequency(self) -> float:
        """Frequency at which the source time dependence has its peak amplitude in the frequency domain."""
        if not self.remove_dc_component:
            return self.freq0
        return 0.5 * (self.freq0 + np.sqrt(self.freq0**2 + 4 * self.fwidth**2))

    @property
    def _peak_freq_amp(self) -> complex:
        """Peak amplitude in frequency domain"""
        return self.amp_freq(self.peak_frequency)

    @property
    def _peak_time_amp(self) -> complex:
        """Peak amplitude in time domain"""
        return self.amp_time(self.peak_time)

    def frequency_range_sigma(self, sigma: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range where the source amplitude is within ``exp(-sigma**2/2)`` of the peak amplitude."""
        if not self.remove_dc_component:
            return self.frequency_range(num_fwidth=sigma)

        # With dc removed, we'll need to solve for the transcendental equation to find the frequency range
        def equation_for_sigma_frequency(freq):
            """computes A / A_p - exp(-sigma)"""
            return np.abs(self._rel_amp_freq(freq)) - np.exp(-(sigma**2) / 2)

        logger = logging.getLogger("pyroots")
        logger.setLevel(logging.CRITICAL)
        root_scalar = Brentq(raise_on_fail=False, epsilon=_ROOTS_TOL)
        fmin_data = root_scalar(equation_for_sigma_frequency, xa=0, xb=self.peak_frequency)
        fmax_data = root_scalar(
            equation_for_sigma_frequency,
            xa=self.peak_frequency,
            xb=self.peak_frequency
            + self.fwidth
            * (
                OFFSET_FWIDTH_FMAX + 2 * sigma**2
            ),  # offset slightly to make sure that it flips sign
        )
        fmin, fmax = fmin_data.x0, fmax_data.x0

        # if unconverged, fall back to `frequency_range`
        if not (fmin_data.converged and fmax_data.converged and fmax > fmin):
            return self.frequency_range(num_fwidth=sigma)

        # converged
        return fmin.item(), fmax.item()

    @property
    def amp_complex(self) -> complex:
        """Grab the complex amplitude from a ``GaussianPulse``."""
        phase = np.exp(1j * self.phase)
        return self.amplitude * phase

    @classmethod
    def from_amp_complex(cls, amp: complex, **kwargs: Any) -> GaussianPulse:
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

    @staticmethod
    def _minimum_source_bandwidth(
        fmin: float, fmax: float, minimum_source_bandwidth: float
    ) -> tuple[float, float]:
        """Define a source bandwidth based on fmin and fmax, but enforce a minimum bandwidth."""
        if minimum_source_bandwidth <= 0:
            raise ValidationError("'minimum_source_bandwidth' must be positive")
        if minimum_source_bandwidth >= 1:
            raise ValidationError("'minimum_source_bandwidth' must less than or equal to 1")

        f_difference = fmax - fmin
        f_middle = 0.5 * (fmin + fmax)

        full_width = minimum_source_bandwidth * f_middle
        if f_difference < full_width:
            half_width = 0.5 * full_width
            fmin = f_middle - half_width
            fmax = f_middle + half_width

        return fmin, fmax

    @classmethod
    def from_frequency_range(
        cls,
        fmin: pydantic.PositiveFloat,
        fmax: pydantic.PositiveFloat,
        minimum_source_bandwidth: pydantic.PositiveFloat = None,
        **kwargs: Any,
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

        if minimum_source_bandwidth is not None:
            fmin, fmax = cls._minimum_source_bandwidth(fmin, fmax, minimum_source_bandwidth)

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
        pulse = cls(freq0=freq0, fwidth=fwidth, **kwargs)
        if np.abs(pulse._rel_amp_freq(fmin)) < WARN_SOURCE_AMPLITUDE:
            log.warning(
                "Source amplitude is not sufficiently large throughout the specified frequency range, "
                "which can result in inaccurate simulation results. Please decrease the frequency range.",
            )
        return pulse


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
        time_shifted = time - self.offset_time

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
        source_time_dataarray = TimeDataArray(values, coords={"t": times})
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
        max_time_shifted = run_time - self.offset_time
        min_time_shifted = -self.offset_time

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


class BroadbandPulse(SourceTime):
    """A source time injecting significant energy in the entire custom frequency range."""

    freq_range: FreqBound = pydantic.Field(
        ...,
        title="Frequency Range",
        description="Frequency range where the pulse should have significant energy.",
        units=HERTZ,
    )
    minimum_amplitude: float = pydantic.Field(
        0.3,
        title="Minimum Amplitude",
        description="Minimum amplitude of the pulse relative to the peak amplitude in the frequency range.",
        gt=0.05,
        lt=0.5,
    )
    offset: float = pydantic.Field(
        0.0,
        title="Offset",
        description="An automatic time delay of the peak value of the pulse has been applied under the hood "
        "to ensure smooth ramping up of the pulse at time = 0. This offfset is added on top of the automatic time delay "
        "in units of 1 / [``2pi * (freq_range[1] - freq_range[0])``].",
    )

    @pydantic.validator("freq_range", always=True)
    def _validate_freq_range(cls, val):
        """Validate that freq_range is positive and properly ordered."""
        if val[0] <= 0 or val[1] <= 0:
            raise ValidationError("Both elements of 'freq_range' must be positive.")
        if val[1] <= val[0]:
            raise ValidationError(
                f"'freq_range[1]' ({val[1]}) must be greater than 'freq_range[0]' ({val[0]})."
            )
        return val

    @pydantic.root_validator()
    def _check_broadband_pulse_available(cls, values):
        """Check if BroadbandPulse is available."""
        check_tidy3d_extras_licensed_feature("BroadbandPulse")
        return values

    @cached_property
    def _source(self):
        """Implementation of broadband pulse."""
        return tidy3d_extras["mod"].extension.BroadbandPulse(
            fmin=self.freq_range[0],
            fmax=self.freq_range[1],
            minRelAmp=self.minimum_amplitude,
            amp=self.amplitude,
            phase=self.phase,
            offset=self.offset,
        )

    def end_time(self) -> float:
        """Time after which the source is effectively turned off / close to zero amplitude."""
        return self._source.end_time(END_TIME_FACTOR_GAUSSIAN)

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""
        return self._source.amp_time(time)

    def amp_freq(self, freq: float) -> complex:
        """Complex-valued source amplitude as a function of frequency."""
        return self._source.amp_freq(freq)

    def frequency_range_sigma(self, sigma: float = DEFAULT_SIGMA) -> FreqBound:
        """Frequency range where the source amplitude is within ``exp(-sigma**2/2)`` of the peak amplitude."""
        return self._source.frequency_range(sigma)

    def frequency_range(self, num_fwidth: float = DEFAULT_SIGMA) -> FreqBound:
        """Delegated to `frequency_range_sigma(sigma=num_fwidth)` for computing the frequency range where the source amplitude
        is within ``exp(-num_fwidth**2/2)`` of the peak amplitude.
        """
        return self.frequency_range_sigma(num_fwidth)


SourceTimeType = Union[GaussianPulse, ContinuousWave, CustomSourceTime, BroadbandPulse]

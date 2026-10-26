"""Baseband (unmodulated) source time classes for transient RF simulations."""

from __future__ import annotations

import math
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
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types import ArrayFloat1D, FreqBound

EdgeShapeType = Literal["erf", "logistic", "raised_cosine", "linear"]

# Longest bit sequence accepted by ``DigitalSourceTime``, which bounds the cost of evaluating
# the waveform: every transition contributes to every sample, so an unbounded pattern makes
# that quadratic.
MAX_BITS = 64


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


# Not exported from ``tidy3d``: ``flexcompute.rf.tidy3d`` is its public home.
class DigitalSourceTime(BasebandSourceTime):
    """Digital (bit-sequence) source time profile with smooth transitions.

    Notes
    -----
    Each bit of ``bits`` occupies a slot of ``bit_duration``, held at ``amplitude`` for a
    '1' and at zero for a '0', joined by transitions of independent ``rise_time`` and
    ``fall_time`` whose functional form is set by ``edge_shape``. The line code is
    selected by ``duty_cycle``.

    As for every baseband source, the faster of ``rise_time`` and ``fall_time`` sets the band
    and, unless ``freq0`` is given, the characteristic frequency, so that an automatic grid
    resolves the transitions; see :class:`.BasebandSourceTime`.

    Example
    -------
    >>> digital = DigitalSourceTime(bits='0110', bit_duration=1e-9, rise_time=1e-10)
    >>> rc_edges = DigitalSourceTime(
    ...     bits='0110', bit_duration=1e-9, rise_time=1e-10, edge_shape='raised_cosine'
    ... )
    >>> mode_at_1ghz = digital.updated_copy(freq0=1e9)
    """

    bits: str = Field(
        title="Bit Sequence",
        description="Sequence of '0' and '1' characters defining the signal, e.g. "
        "``'011010'``. Each bit occupies one slot of duration ``bit_duration``, held at "
        "``amplitude`` for a '1' and at zero for a '0'. The sequence is padded with a "
        "virtual '0' at both ends, so the signal starts and ends at zero, and must contain "
        "at least one '1', since a pattern of only '0' bits produces no signal at all.",
        max_length=MAX_BITS,
    )

    bit_duration: PositiveFloat = Field(
        title="Bit Duration",
        description="Duration of a single bit slot in seconds.",
        allow_inf_nan=False,
    )

    duty_cycle: float = Field(
        default=1.0,
        title="Duty Cycle",
        description="Fraction of each bit slot for which a '1' bit is held high, which "
        "selects the line code. The default of 1 gives non-return-to-zero (NRZ) "
        "signaling, whereas a value below 1 gives return-to-zero (RZ) signaling.",
        gt=0,
        le=1,
    )

    rise_time: PositiveFloat = Field(
        title="Rise Time",
        description="The 10%-to-90% rise time of '0' -> '1' transitions in seconds.",
        allow_inf_nan=False,
    )

    fall_time: PositiveFloat | None = Field(
        default=None,
        title="Fall Time",
        description="The 90%-to-10% fall time of '1' -> '0' transitions in seconds. "
        "If ``None``, ``rise_time`` is used.",
        allow_inf_nan=False,
    )

    offset: float = Field(
        default=0.5,
        title="Offset",
        description="Delay of the first bit slot in units of ``bit_duration``, at least half a "
        "slot, which is the default. The source must rise from zero after ``t = 0``: a "
        "transition too slow to settle within the offset is rejected with the smallest "
        "``offset`` that fits it.",
        ge=0.5,
        allow_inf_nan=False,
    )

    edge_shape: EdgeShapeType = Field(
        default="erf",
        title="Edge Shape",
        description="Functional form of each bit transition: ``'erf'`` a Gaussian integral, "
        "``'logistic'`` a sigmoid, ``'raised_cosine'`` a half cosine, and ``'linear'`` a "
        "straight ramp. ``'raised_cosine'`` and ``'linear'`` reach each level in a finite "
        "time, so the signal is exactly flat between transitions; ``'erf'`` and "
        "``'logistic'`` approach their levels asymptotically instead, but are smooth to all "
        "orders. The 50% crossing lies on the bit-slot boundary for every shape, so "
        "changing the shape does not shift the bit timing.",
    )

    @field_validator("bits")
    @classmethod
    def _validate_bits(cls, value: str) -> str:
        """Bit sequence must be a string of '0' and '1' characters holding at least one '1'."""
        if set(value) - {"0", "1"}:
            raise ValidationError(
                f"'bits' must contain only '0' and '1' characters, got '{value}'."
            )
        if "1" not in value:
            raise ValidationError("'bits' must contain at least one '1' to produce a signal.")
        return value

    @model_validator(mode="after")
    def _validate_source_starts_from_zero(self) -> DigitalSourceTime:
        """Reject setups whose first transition is already under way at ``t = 0``.

        A transition cut off at the start of the simulation switches the source on with a
        step, injecting content that no grid sized to the transitions resolves. Half a slot
        fits a transition that settles within its share of a level's hold, but unequal
        ``rise_time`` and ``fall_time`` may give one of them nearly the whole hold. Each
        transition is held at ``t = 0`` to the tolerance of every settled level, one at a
        time as ``end_time`` treats them, so the suggested ``offset``, which starts every
        settling window after ``t = 0`` and is rounded up, is accepted.
        """
        step = self._edge_shape_spec.step
        level = max(step(-t_edge / edge_time) for t_edge, _, edge_time in self._transitions)
        if level > baseband_numerics.EDGE_LEVEL_TOLERANCE:
            start = baseband_numerics.digital_start_time(
                transitions=self._transitions, edge_shape=self.edge_shape
            )
            # rounded up to a ten-thousandth of a slot, so never below what the check needs
            required = math.ceil((self.offset - start / self.bit_duration) * 1e4) / 1e4
            self._raise_validation_error_at_loc(
                "With these transition times a transition is already under way at t = 0, so "
                f"the source would switch on abruptly. Set 'offset' to at least {required:.4f}.",
                "offset",
            )
        return self

    @model_validator(mode="after")
    def _warn_edges_exceed_hold(self) -> DigitalSourceTime:
        """Warn about setups whose bits cannot reach their level.

        Every level is held from the 50% crossing of the transition into it to the 50%
        crossing of the transition out of it, so the hold has to fit the settling half of
        one edge and the approaching half of the other: the rise and fall times together,
        not the slower of the two alone. The allowance is twice the single-edge one, which
        the shapes are calibrated against with equal edges.
        """
        edge_times = self.rise_time + self._fall_time
        allowed = 2 * self._edge_shape_spec.max_edge_fraction * self._hold_window
        if edge_times > allowed:
            log.warning(
                f"'DigitalSourceTime' has a 'rise_time' of {self.rise_time:.3e} s and a "
                f"'fall_time' of {self._fall_time:.3e} s, which together are too long for the "
                f"shortest level hold of {self._hold_window:.3e} s (set by 'bit_duration' "
                "and 'duty_cycle'), so the signal will not settle at 'amplitude' or zero "
                "before the next transition. Keep 'rise_time' plus 'fall_time' below "
                f"{allowed:.3e} s for '{self.edge_shape}' edges, or lengthen 'bit_duration'."
            )
        return self

    @cached_property
    def _fall_time(self) -> float:
        """Fall time, defaulting to the rise time."""
        return self.rise_time if self.fall_time is None else self.fall_time

    @cached_property
    def _edge_shape_spec(self) -> baseband_numerics.EdgeShape:
        """Step function and calibration of the selected ``edge_shape``."""
        return baseband_numerics.EDGE_SHAPES[self.edge_shape]

    @cached_property
    def _hold_window(self) -> float:
        """Shortest time the waveform holds a level, in seconds."""
        return baseband_numerics.digital_hold_window(
            bit_duration=self.bit_duration, duty_cycle=self.duty_cycle
        )

    @cached_property
    def t_start(self) -> float:
        """Start time of the first bit slot in seconds."""
        return self.offset * self.bit_duration

    @cached_property
    def _transitions(self) -> list[tuple[float, float, float]]:
        """``(time, direction, edge_time)`` of each transition, in time order."""
        return baseband_numerics.digital_transitions(
            bits=self.bits,
            bit_duration=self.bit_duration,
            duty_cycle=self.duty_cycle,
            rise_time=self.rise_time,
            fall_time=self._fall_time,
            t_start=self.t_start,
        )

    @property
    def _fastest_transition_time(self) -> float:
        """The faster of the 10%-to-90% rise and fall times."""
        return min(self.rise_time, self._fall_time)

    def _envelope(self, time: ArrayFloat1D) -> ArrayFloat1D:
        """Signed sum of the transition steps, running between 0 and 1."""
        return baseband_numerics.digital_envelope(
            time, transitions=self._transitions, edge_shape=self.edge_shape
        )

    def amp_time(self, time: float | ArrayFloat1D) -> ArrayFloat1D:
        """Real-valued source amplitude as a function of time."""
        return self.amplitude * self._envelope(time)

    def end_time(self) -> float:
        """Time after which the source is effectively zero."""
        return baseband_numerics.digital_end_time(
            transitions=self._transitions, edge_shape=self.edge_shape
        )

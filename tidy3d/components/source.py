"""Defines electric current sources for injecting light into simulation."""

from abc import ABC, abstractmethod
from typing import Union

import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import Direction, Polarization, Ax, FreqBound, Array
from .types import inf  # pylint:disable=unused-import
from .validators import assert_plane, validate_name_str
from .geometry import Box
from .mode import ModeSpec
from .viz import add_ax_if_none, SourceParams, equal_aspect
from ..constants import RADIAN, HERTZ, MICROMETER
from ..log import SetupError

# TODO: change directional source to something signifying its intent is to create a specific field.

# width of pulse frequency range defition in units of standard deviation.
WIDTH_STD = 5


class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source."""

    amplitude: pydantic.NonNegativeFloat = pydantic.Field(
        1.0, title="Amplitude", description="Real-valued maximum amplitude of the time dependence."
    )

    phase: float = pydantic.Field(
        0.0, title="Phase", description="Phase shift of the time dependence.", units=RADIAN
    )

    @abstractmethod
    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at that time..
        """

    def spectrum(self, times: Array[float], freqs: Array[float], dt: float) -> complex:
        """Complex-valued source spectrum as a function of frequency

        Parameters
        ----------
        times : np.ndarray
            Times to use to evaluate spectrum Fourier transform.
            (Typically the simulation time mesh).
        freqs : np.ndarray
            Frequencies in Hz to evaluate spectrum at.
        dt : float or np.ndarray
            Time step to weight FT integral with.
            If array, use to weigh each of the time intervals in ``times``.

        Returns
        -------
        np.ndarray
            Complex-valued array (of len(freqs)) containing spectrum at those frequencies.
        """

        # (Nf, Nt) matrix that gives DFT when matrix multiplied with signal
        dft_matrix = np.exp(2j * np.pi * freqs[:, None] * times) / np.sqrt(2 * np.pi)
        return dt * dft_matrix @ np.real(self.amp_time(times))

    @add_ax_if_none
    def plot(self, times: Array[float], ax: Ax = None) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.

        Parameters
        ----------
        times : np.ndarray
            Array of times (seconds) to plot source at.
            To see source time amplitude for a specific :class:`Simulation`,
            pass ``simulation.tmesh``.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        times = np.array(times)
        amp_complex = self.amp_time(times)

        ax.plot(times, amp_complex.real, color="blueviolet", label="real")
        ax.plot(times, amp_complex.imag, color="crimson", label="imag")
        ax.plot(times, np.abs(amp_complex), color="k", label="abs")
        ax.set_xlabel("time (s)")
        ax.set_title("source amplitude")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @add_ax_if_none
    def plot_spectrum(self, times: Array[float], num_freqs: int = 101, ax: Ax = None) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.

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
        times = np.array(times)

        dts = np.diff(times)
        if not np.allclose(dts, dts[0] * np.ones_like(dts)):
            raise SetupError("Supplied times not evenly spaced.")

        dt = np.mean(dts)

        fmin, fmax = self.frequency_range
        freqs = np.linspace(fmin, fmax, num_freqs)

        spectrum = self.spectrum(times=times, dt=dt, freqs=freqs)

        ax.plot(freqs, spectrum.real, color="blueviolet", label="real")
        ax.plot(freqs, spectrum.imag, color="crimson", label="imag")
        ax.plot(freqs, np.abs(spectrum), color="k", label="abs")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title("source spectrum")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @property
    @abstractmethod
    def frequency_range(self) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency."""


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
        description="Time delay of the maximum value of the pulse in units of 1 / ``fwidth``.",
        ge=2.5,
    )

    @property
    def frequency_range(self) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum frequencies of the
            :class:`GaussianPulse` or :class:`ContinuousWave` power
            within 6 standard deviations.
        """
        width_std = 6
        freq_width_range = width_std * self.fwidth
        freq_min = max(0, self.freq0 - freq_width_range)
        freq_max = self.freq0 + freq_width_range
        return (freq_min, freq_max)


class GaussianPulse(Pulse):
    """Source time dependence that describes a Gaussian pulse.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1j + time_shifted / twidth ** 2 / omega0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted ** 2) / 2 / twidth ** 2)

        return const * offset * oscillation * amp


class ContinuousWave(Pulse):
    """Source time dependence that ramps up to continuous oscillation
    and holds until end of simulation.

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
        amp = 1 / (1 + np.exp(-time_shifted / twidth))

        return const * offset * oscillation * amp


SourceTimeType = Union[GaussianPulse, ContinuousWave]

""" Source objects """


class Source(Box, ABC):
    """Abstract base class for all sources."""

    source_time: SourceTimeType = pydantic.Field(
        ..., title="Source Time", description="Specification of the source time-dependence."
    )

    name: str = pydantic.Field(None, title="Name", description="Optional name for the source.")

    _name_validator = validate_name_str()

    @equal_aspect
    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:

        kwargs = SourceParams().update_params(**kwargs)
        ax = self.geometry.plot(x=x, y=y, z=z, ax=ax, **kwargs)
        return ax

    @property
    def geometry(self):
        """:class:`Box` representation of source."""

        return Box(center=self.center, size=self.size)


class VolumeSource(Source):
    """Source spanning a rectangular volume with uniform time dependence.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_source = VolumeSource(size=(0,0,0), source_time=pulse, polarization='Ex')
    """

    polarization: Polarization = pydantic.Field(
        ...,
        title="Polarization",
        description="Specifies the direction and type of current component.",
    )


class FieldSource(Source, ABC):
    """A planar Source defined by the desired E and H fields at a plane. The sources are created
    such that the propagation is uni-directional."""

    direction: Direction = pydantic.Field(
        ...,
        title="Direction",
        description="Specifies propagation in positive or negative direction of the normal axis.",
    )

    _plane_validator = assert_plane()


class ModeSource(FieldSource):
    """Injects current source to excite modal profile on finite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> mode_spec = ModeSpec(target_neff=2.)
    >>> mode_source = ModeSource(
    ...     size=(10,10,0),
    ...     source_time=pulse,
    ...     mode_spec=mode_spec,
    ...     mode_index=1,
    ...     direction='-')
    """

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    mode_index: pydantic.NonNegativeInt = pydantic.Field(
        0,
        title="Mode Index",
        description="Index into the collection of modes returned by mode solver. "
        " Specifies which mode to inject using this source. "
        "If larger than ``mode_spec.num_modes``, "
        "``num_modes`` in the solver will be set to ``mode_index + 1``.",
    )

    # @pydantic.validator("mode_index", always=True)
    # def mode_index_in_bounds(cls, val, values):
    #     """Ensures number of modes in mode spec can support mode index."""
    #     mode_spec = values.get("mode_spec")
    #     if mode_spec is None:
    #         raise SetupError("ModeSpec not found.")
    #     num_modes = mode_spec.num_modes
    #     if num_modes <= val:
    #         raise SetupError(
    #             f"ModeSpec contains {num_modes} modes, but mode index is {val}. "
    #             "Either increase number of modes in the specifications or decrease mode index."
    #         )
    #     return val


class AngledFieldSource(FieldSource):
    """Field Source with a polarization angle."""

    pol_angle: float = pydantic.Field(
        0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the normal axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        units=RADIAN,
    )


class PlaneWave(AngledFieldSource):
    """Uniform current distribution on an infinite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw_source = PlaneWave(size=(inf,0,inf), source_time=pulse, pol_angle=0.1, direction='+')
    """

    # Note: this is replaced by `pol_angle`
    # polarization: Polarization = "Ex"


class GaussianBeam(AngledFieldSource):
    """Guassian distribution on finite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = GaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_radius=1.0)
    """

    waist_radius: pydantic.PositiveFloat = pydantic.Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        units=MICROMETER,
    )

    waist_distance: float = pydantic.Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction.",
        units=MICROMETER,
    )

    angle_theta: float = pydantic.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle from the normal axis.",
        units=RADIAN,
    )

    angle_phi: float = pydantic.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle in the plane orthogonal to the normal axis.",
        units=RADIAN,
    )


# sources allowed in Simulation.sources
SourceType = Union[VolumeSource, PlaneWave, ModeSource, GaussianBeam]

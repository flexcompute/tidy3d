"""Defines electric current sources for injecting light into simulation."""

from abc import ABC, abstractmethod
from typing import Union, Tuple
import logging

from typing_extensions import Annotated
import pydantic
import numpy as np

from .base import Tidy3dBaseModel, TYPE_TAG_STR
from .types import Direction, Polarization, Ax, FreqBound, Array, Axis
from .validators import assert_plane, validate_name_str
from .geometry import Box
from .mode import ModeSpec
from .viz import add_ax_if_none, SourceParams, equal_aspect
from .viz import ARROW_COLOR_SOURCE, ARROW_ALPHA, ARROW_COLOR_POLARIZATION
from ..constants import RADIAN, HERTZ, MICROMETER, GLANCING_CUTOFF
from ..constants import inf  # pylint:disable=unused-import
from ..log import SetupError

# in spectrum computation, discard amplitudes with relative magnitude smaller than cutoff
DFT_CUTOFF = 1e-8


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

        time_amps = np.real(self.amp_time(times))

        # Cut to only relevant times
        count_times = np.where(np.abs(time_amps) / np.amax(np.abs(time_amps)) > DFT_CUTOFF)
        time_amps = time_amps[count_times]
        times_cut = times[count_times]

        # (Nf, Nt_cut) matrix that gives DFT when matrix multiplied with signal
        dft_matrix = np.exp(2j * np.pi * freqs[:, None] * times_cut) / np.sqrt(2 * np.pi)
        return dt * dft_matrix @ time_amps

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

        fmin, fmax = self.frequency_range()
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

    @abstractmethod
    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within plus/minus ``num_fwidth * fwidth`` of the central frequency."""


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

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1j + time_shifted / twidth**2 / omega0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted**2) / 2 / twidth**2)

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
    """Source in a rectangular volume with uniform time dependence. size=(0,0,0) gives point source.

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


class PlanarSource(Source, ABC):
    """A source defined on a 2D plane."""

    _plane_validator = assert_plane()

    @property
    def injection_axis(self):
        """Injection axis of the source."""
        return self.size.index(0.0)

    @injection_axis.setter
    def injection_axis(self):
        raise ValueError("PlanarSource injection axis is defined by the axis normal to the plane.")


class FieldSource(Source, ABC):
    """A Source defined by the desired E and/or H fields."""


class DirectionalSource(FieldSource, ABC):
    """A FieldSource defined with a direction of propagation. The direction is defined by
    the polar and azimuth angles w.r.t. an injection axis, as well as forward ``+`` or
    backward ``-``. This base class only defines the ``direction`` and ``injection_axis``
    attributes, but it must be composed with a class that also defines ``angle_theta`` and
    ``angle_phi``."""

    direction: Direction = pydantic.Field(
        ...,
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the injection "
        "axis.",
    )

    injection_axis: Axis = pydantic.Field(
        None,
        title="Injection Axis",
        description="Specifies injection axis. The popagation axis is defined with respect to "
        "the injection axis by ``angle_theta`` and ``angle_phi``. Must be ``None`` for planar "
        "directional sources, as it is taken automatically from the plane size.",
    )

    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:

        # call the `Source.plot()` function first.
        ax = super().plot(x=x, y=y, z=z, ax=ax, **kwargs)

        # then add the arrow based on the propagation direction
        ax = self._plot_arrow(
            x=x,
            y=y,
            z=z,
            ax=ax,
            direction=self._dir_arrow,
            color=ARROW_COLOR_SOURCE,
            alpha=ARROW_ALPHA,
            both_dirs=False,
        )
        return ax

    # pylint: disable=no-member
    @property
    def _dir_arrow(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self.injection_axis)


class ModeSource(PlanarSource, DirectionalSource):
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

    @property
    def angle_theta(self):
        """Polar angle of propagation."""
        return self.mode_spec.angle_theta

    @property
    def angle_phi(self):
        """Azimuth angle of propagation."""
        return self.mode_spec.angle_phi


class AngledDirectionalSource(DirectionalSource, ABC):
    """Directional source with explicitly defined ``angle_theta`` and ``angle_phi`` fields."""

    angle_theta: float = pydantic.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        units=RADIAN,
    )

    angle_phi: float = pydantic.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        units=RADIAN,
    )

    @pydantic.validator("angle_theta", allow_reuse=True, always=True)
    def glancing_incidence(cls, val):
        """Warn if close to glancing incidence."""
        if np.abs(np.pi / 2 - val) < GLANCING_CUTOFF:
            logging.warning(
                "Angled source propagation axis close to glancing angle. "
                "For best results, switch the injection axis."
            )
        return val


class PolarizedSource(AngledDirectionalSource, ABC):
    """AngledDirectionalSource with a polarization angle."""

    pol_angle: float = pydantic.Field(
        0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the injection axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        units=RADIAN,
    )

    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:

        # call the `DirectionalSource.plot()` function first (including dir arrow)
        ax = super().plot(x=x, y=y, z=z, ax=ax, **kwargs)

        # then add another arrow based on the polarization direction
        ax = self._plot_arrow(
            x=x,
            y=y,
            z=z,
            ax=ax,
            direction=self._pol_arrow,
            color=ARROW_COLOR_POLARIZATION,
            alpha=ARROW_ALPHA,
            both_dirs=False,
        )
        return ax

    @property
    def _pol_arrow(self) -> Tuple[float, float, float]:
        """Source polarization normal vector in cartesian coordinates."""
        normal_dir = [0.0, 0.0, 0.0]
        normal_dir[self.injection_axis] = 1.0
        propagation_dir = list(self._dir_arrow)
        pol_vector = np.cross(normal_dir, propagation_dir)
        if np.all(pol_vector == 0.0):
            pol_vector = np.array((0, 1, 0)) if self.injection_axis == 0 else np.array((1, 0, 0))
        return self.rotate_points(pol_vector, propagation_dir, angle=self.pol_angle)


class PlaneWave(PlanarSource, PolarizedSource):
    """Uniform current distribution on an infinite extent plane.  One element of size must be zero.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw_source = PlaneWave(size=(inf,0,inf), source_time=pulse, pol_angle=0.1, direction='+')
    """

    @pydantic.validator("angle_theta", always=True)
    def normal_incidence(cls, val):
        """Raise not implemented error if not normal incidence."""
        if val > 0.0:
            raise NotImplementedError(
                "Plane wave with off-normal incidence requires Bloch "
                "boundary conditions, which are not yet implemented."
            )
        return val


class GaussianBeam(PlanarSource, PolarizedSource):
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


# sources allowed in Simulation.sources
SourceType = Annotated[
    Union[VolumeSource, GaussianBeam, ModeSource, PlaneWave],
    pydantic.Field(discriminator=TYPE_TAG_STR),
]

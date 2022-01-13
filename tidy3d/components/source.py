"""Defines electric current sources for injecting light into simulation."""

from abc import ABC, abstractmethod
from typing import Union

import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import Direction, Polarization, Ax, FreqBound, Array, Literal
from .validators import assert_plane, validate_name_str
from .geometry import Box
from .mode import Mode
from .viz import add_ax_if_none, SourceParams
from ..constants import inf  # pylint:disable=unused-import

# TODO: change directional source to something signifying its intent is to create a specific field.

# width of pulse frequency range defition in units of standard deviation.
WIDTH_STD = 5


class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source."""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0

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
        return dt * dft_matrix @ self.amp_time(times)

    @add_ax_if_none
    def plot(self, times: Array[float], ax: Ax = None) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.

        Parameters
        ----------
        times : np.ndarray
            Array of times to plot source at in seconds.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        times = np.array(times)
        amp_complex = self.amp_time(times)

        times_ps = times / 1e-12
        ax.plot(times_ps, amp_complex.real, color="blueviolet", label="real")
        ax.plot(times_ps, amp_complex.imag, color="crimson", label="imag")
        ax.plot(times_ps, np.abs(amp_complex), color="k", label="abs")
        ax.set_xlabel("time (ps)")
        ax.set_title("source amplitude")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @property
    @abstractmethod
    def frequency_range(self) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency."""


class Pulse(SourceTime, ABC):
    """A source time that ramps up with some ``fwidth`` and oscillates at ``freq0``."""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat  # currently standard deviation
    offset: pydantic.confloat(ge=2.5) = 5.0

    @property
    def frequency_range(self) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum frequencies of the
            :class:`GaussianPulse` or :class:`ContinuousWave` power
            within 5 standard deviations.
        """
        width_std = 5
        freq_min = max(0, self.freq0 - width_std * self.fwidth)
        freq_max = self.freq0 + width_std * self.fwidth
        return (freq_min, freq_max)


class GaussianPulse(Pulse):
    """Source time dependence that describes a Gaussian pulse.

    Parameters
    ----------
        freq0 : float
            Central oscillating frequency in Hz.
            Must be positive.
        fwidth : float
            Standard deviation width of the Gaussian pulse in Hz.
            Must be positive.
        offset : float = 5.0
            Time of the maximum value of the pulse
            in units of 1/fwidth.
            Must be greater than 2.5.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at supplied time.
        """

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

    Parameters
    ----------
        freq0 : float
            Central oscillating frequency in Hz.
            Must be positive.
        fwidth : float
            Standard deviation width of the Gaussian pulse in Hz.
            Must be positive.
        offset : float = 5.0
            Time of the maximum value of the pulse
            in units of 1/fwidth.
            Must be greater than 2.5.

    Example
    -------
    >>> cw = ContinuousWave(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at supplied time.
        """
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
    """Abstract base class for all sources.

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of source in x,y,z.
    size : Tuple[float, float, float]
        Size of source in x,y,z.
        All elements must be non-negative.
    source_time : :class:`GaussianPulse` or :class:`ContinuousWave`
        Specification of time-dependence of source.
    name : str = None
        Optional name for source.
    """

    source_time: SourceTimeType
    name: str = None

    _name_validator = validate_name_str()

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot the source geometry on a cross section plane.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs = SourceParams().update_params(**kwargs)
        ax = self.geometry.plot(x=x, y=y, z=z, ax=ax, **kwargs)
        return ax

    @property
    def geometry(self):
        """:class:`Box` representation of source.

        Returns
        -------
        :class:`Box`
            Representation of the source geometry as a :class:`Box`.
        """
        return Box(center=self.center, size=self.size)


class VolumeSource(Source):
    """Source spanning a rectangular volume with uniform time dependence.

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of source in x,y,z.
    size : Tuple[float, float, float]
        Size of source in x,y,z.
        All elements must be non-negative.
    source_time : :class:`GaussianPulse` or :class:`ContinuousWave`
        Specification of time-dependence of source.
    polarization : str
        Specifies the direction and type of current component.
        Must be in ``{'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'}``.
        For example, ``'Ez'`` specifies electric current source polarized along the z-axis.
    name : str = None
        Optional name for source.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_source = VolumeSource(size=(0,0,0), source_time=pulse, polarization='Ex')
    """

    polarization: Polarization
    type: Literal["VolumeSource"] = "VolumeSource"


class FieldSource(Source, ABC):
    """A planar Source defined by the desired E and H fields at a plane. The sources are created
    such that the propagation is uni-directional."""

    direction: Direction
    _plane_validator = assert_plane()


class ModeSource(FieldSource):
    """Modal profile on finite extent plane

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of source in x,y,z.
    size : Tuple[float, float, float]
        Size of source in x,y,z.
        One component must be 0.0 to define plane.
        All elements must be non-negative.
    source_time : :class:`GaussianPulse` or :class:`ContinuousWave`
        Specification of time-dependence of source.
    direction : str
        Specifies propagation in the positive or negative direction of the normal axis. Must be in
        ``{'+', '-'}``.
    mode : :class:`Mode`
        Specification of the mode being injected by source.
    name : str = None
        Optional name for source.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> mode = Mode(mode_index=1, num_modes=3)
    >>> mode_source = ModeSource(size=(10,10,0), source_time=pulse, mode=mode, direction='-')
    """

    type: Literal["ModeSource"] = "ModeSource"
    mode: Mode


class PlaneWave(FieldSource):
    """Uniform distribution on infinite extent plane.

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of source in x,y,z.
    size : Tuple[float, float, float]
        Size of source in x,y,z.
        One component must be 0.0 to define plane.
        All elements must be non-negative.
    source_time : :class:`GaussianPulse` or :class:`ContinuousWave`
        Specification of time-dependence of source.
    pol_angle : float, optional
        Specifies the angle between the electric field polarization of the source and the plane
        defined by the normal axis and the propagation axis (rad). ``pol_angle=0`` (default)
        specifies P polarization, while ``pol_angle=np.pi/2`` specifies S polarization. At normal
        incidence when S and P are undefined, ``pol_angle=0`` defines, respectively:
         - ``Ey`` polarization for propagation along ``x``.
         - ``Ex`` polarization for propagation along ``y``.
         - ``Ex`` polarization for propagation along ``z``.
    direction : str
        Specifies propagation in the positive or negative direction of the normal axis. Must be in
        ``{'+', '-'}``.
    name : str = None
        Optional name for source.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw_source = PlaneWave(size=(inf,0,inf), source_time=pulse, polarization='Ex', direction='+')
    """

    type: Literal["PlaneWave"] = "PlaneWave"
    pol_angle: float = 0
    # TODO: this is only needed so that the convert path still works. Remove eventually.
    polarization: Polarization = "Ex"


class GaussianBeam(FieldSource):
    """guassian distribution on finite extent plane

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of source in x,y,z.
    size : Tuple[float, float, float]
        Size of source in x,y,z.
        One component must be 0.0 to define plane.
        All elements must be non-negative.
    source_time : :class:`GaussianPulse` or :class:`ContinuousWave`
        Specification of time-dependence of source.
    direction : str
        Specifies propagation in the positive or negative direction of the normal axis. Must be in
        ``{'+', '-'}``.
    waist_radius: float = 1.0
        Radius of the beam at the waist (um).
        Must be positive.
    waist_distance: float = 0.0
        Distance (um) from the beam waist along the propagation direction.
        Must be non-negative.
    angle_theta : float, optional
        Polar angle from the normal axis (rad).
    angle_phi : float, optional
        Azimuth angle in the plane orthogonal to the normal axis (rad).
    pol_angle : float, optional
        Specifies the angle between the electric field polarization of the source and the plane
        defined by the normal axis and the propagation axis (rad). ``pol_angle=0`` (default)
        specifies P polarization, while ``pol_angle=np.pi/2`` specifies S polarization. At normal
        incidence when S and P are undefined, ``pol_angle=0`` defines, respectively:
         - ``Ey`` polarization for propagation along ``x``.
         - ``Ex`` polarization for propagation along ``y``.
         - ``Ex`` polarization for propagation along ``z``.
    name : str = None
        Optional name for source.

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

    waist_radius: pydantic.PositiveFloat = 1.0
    waist_distance: pydantic.NonNegativeFloat = 0.0
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    type: Literal["GaussianBeam"] = "GaussianBeam"


# sources allowed in Simulation.sources
SourceType = Union[VolumeSource, PlaneWave, ModeSource, GaussianBeam]
